"""Optional exact CUDA implementation for the fixed diagonal Bayesian core.

The extension is deliberately loaded lazily.  Importing :mod:`bayes_ssm` must
remain usable on CPU-only machines and on CUDA installations without a
compiler.  The kernel is an exact dense-covariance recurrence specialized to
diagonal transitions and scalar process noise; it is not a low-rank or
approximate state representation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


_EXTENSIONS: dict[int, object] = {}
_LOAD_ERRORS: dict[int, BaseException] = {}


def _load_extension(key_dim: int):
    if key_dim in _EXTENSIONS:
        return _EXTENSIONS[key_dim]
    if key_dim in _LOAD_ERRORS:
        return None
    if os.environ.get("BAYES_SSM_DISABLE_FUSED_CUDA", "0") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        if key_dim == 64:
            source = Path(__file__).with_name("persistent_bayes_chunked.cu")
        elif key_dim == 8:
            source = Path(__file__).with_name("persistent_bayes_chunked_d8.cu")
        elif key_dim == 4:
            source = Path(__file__).with_name("persistent_bayes_chunked_d4m8.cu")
        elif key_dim == 2:
            source = Path(__file__).with_name("persistent_bayes_chunked_d2m8.cu")
        elif key_dim == 1:
            source = Path(__file__).with_name("persistent_bayes_chunked_d1m8.cu")
        else:
            return None
        _EXTENSIONS[key_dim] = load(
            name=f"bayes_ssm_persistent_chunked_d{key_dim}",
            sources=[str(source)],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("BAYES_SSM_VERBOSE_BUILD", "0") == "1",
        )
    except BaseException as exc:  # optional acceleration must not break fallback
        _LOAD_ERRORS[key_dim] = exc
        return None
    return _EXTENSIONS[key_dim]


class _ChunkedBayesianFunction(torch.autograd.Function):
    """Autograd boundary for the exact chunked CUDA recurrence."""

    @staticmethod
    def forward(
        ctx,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        decay: Tensor,
        process: Tensor,
        observation: Tensor,
        initial_mean: Tensor,
        initial_covariance: Tensor,
        chunk_length: int,
        denominator_floor: float,
        fixed_dynamics: bool,
    ):
        extension = _load_extension(keys.shape[-1])
        if extension is None:
            raise RuntimeError("the fused Bayesian CUDA extension is unavailable")
        n, t, _ = keys.shape
        # D=1 has a particularly cheap recurrent state, so retain its exact
        # token replay values while scanning one full sequence chunk.  The
        # reverse pass can consume these values directly and avoid launching
        # the separate recompute kernel.  All other dimensions and partial
        # chunks keep the boundary/recompute implementation.
        # D1's scalar covariance makes a full-sequence replay cheaper than
        # launching one recompute kernel per chunk.  The recurrence itself is
        # chunk-independent, so use the exact full-history path for every
        # chunk_length; callers retain identical outputs and gradients while
        # avoiding the known multi-chunk reverse-state synchronization path.
        use_saved_history = (
            # The opt-out remains available for full-chunk microbenchmarks,
            # but never routes D1 partial chunks through the known-unsafe
            # generic recompute path.
            (os.environ.get("BAYES_SSM_DISABLE_SAVED_HISTORY", "0") != "1"
             or int(chunk_length) < int(t))
            and not fixed_dynamics
            and keys.shape[-1] == 1
            and hasattr(extension, "forward_chunked_saved")
        )
        if use_saved_history:
            result = extension.forward_chunked_saved(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, int(t),
                float(denominator_floor),
            )
            (output, final_mean, final_covariance, boundary_mean,
             boundary_covariance, mp_history, pp_history, u_history,
             beta_history, e_history) = result
        else:
            forward_impl = (
                extension.forward_chunked_fixed
                if fixed_dynamics else extension.forward_chunked
            )
            result = forward_impl(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, int(chunk_length),
                float(denominator_floor),
            )
            (output, final_mean, final_covariance, boundary_mean,
             boundary_covariance) = result
        # The extension returns ordinary PyTorch tensors allocated on the
        # current stream.  Do not clone them here: a clone adds a full
        # read/write pass over the [N,T,M] readout on every training step (and
        # another pass over the final state), which is especially expensive for
        # the large six-layer envelope.  CUDA-graph replay is safe because the
        # forward output is consumed by the captured backward before the next
        # replay, and the boundary/history tensors below already remain in the
        # extension-owned graph pool.  Keeping the original tensors also makes
        # this path consistent with the saved-history ABI.
        # Save exactly the contiguous buffers passed to the extension.  The
        # projected [B,H,T,D] tensors are commonly reshaped views; saving the
        # original views would make the reverse kernel's flat indexing read
        # unrelated elements even though the forward result was correct.
        if use_saved_history:
            ctx.save_for_backward(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, boundary_mean,
                boundary_covariance, mp_history, pp_history, u_history,
                beta_history, e_history,
            )
        else:
            ctx.save_for_backward(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, boundary_mean,
                boundary_covariance,
            )
        ctx.use_saved_history = use_saved_history
        ctx.chunk_length = int(t if use_saved_history else chunk_length)
        ctx.denominator_floor = float(denominator_floor)
        ctx.fixed_dynamics = bool(fixed_dynamics)
        # The production Bayesian mixer uses fixed decay/process/observation
        # buffers and discards the recurrent final states.  Keep this bit of
        # metadata so the reverse kernel can omit the unused dynamics-gradient
        # epilogue while retaining the full path for callers that learn decay.
        ctx.need_dynamics_grad = any(
            value.requires_grad for value in (
                decay, process, observation, initial_mean, initial_covariance,
            )
        )
        # The production mixer starts from constant zero/identity state.  Keep
        # this separate from ``need_dynamics_grad`` so the reverse extension
        # can omit only the two final initial-state gradient allocations while
        # still returning exact learned transition/noise gradients.
        ctx.need_initial_grad = (
            initial_mean.requires_grad or initial_covariance.requires_grad
        )
        ctx.output_shape = tuple(output.shape)
        ctx.mean_shape = tuple(final_mean.shape)
        ctx.covariance_shape = tuple(final_covariance.shape)
        # The model consumes only the token readout.  Without this setting
        # autograd materializes zero tensors for unused final-state outputs,
        # making ``lean_backward`` indistinguishable from a caller that really
        # differentiates the returned mean/covariance.  Keeping unused output
        # gradients as ``None`` lets the exact lean reverse kernels skip all
        # state/dynamics-gradient work while preserving full gradients when a
        # caller uses either final state.
        ctx.set_materialize_grads(False)
        return output, final_mean, final_covariance

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_covariance):
        saved = ctx.saved_tensors
        (
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
        ) = saved[:10]
        if ctx.use_saved_history:
            ctx.noinitial_backward = False
            _, _, _, _, _, _, _, _, _, _, mp_history, pp_history, u_history, beta_history, e_history = saved
        extension = _load_extension(keys.shape[-1])
        if extension is None:  # pragma: no cover - extension loaded in forward
            raise RuntimeError("the fused Bayesian CUDA extension is unavailable")
        device = keys.device
        # In the production mixer only the token readout is consumed.  The
        # recurrent final mean/covariance are returned for the public API but
        # have no downstream users, so autograd supplies ``None`` for them.
        # Let the extension omit all unused dynamics/state-gradient buffers
        # in this common fixed-parameter case.
        lean_backward = (
            not ctx.need_dynamics_grad
            and grad_mean is None
            and grad_covariance is None
        )
        grad_output = (
            torch.zeros(ctx.output_shape, device=device, dtype=keys.dtype)
            if grad_output is None else grad_output.contiguous()
        )
        if not lean_backward:
            grad_mean = (
                torch.zeros(ctx.mean_shape, device=device, dtype=keys.dtype)
                if grad_mean is None else grad_mean.contiguous()
            )
            grad_covariance = (
                torch.zeros(ctx.covariance_shape, device=device, dtype=keys.dtype)
                if grad_covariance is None else grad_covariance.contiguous()
            )
        if ctx.use_saved_history:
            if (
                os.environ.get("BAYES_SSM_DISABLE_D1_SPECIALIZED", "0") != "1"
                and not lean_backward
                and keys.shape[-1] == 1
                and hasattr(extension, "backward_chunked_saved_d1")
            ):
                backward = extension.backward_chunked_saved_d1
            else:
                backward = (
                    extension.backward_chunked_saved_lean
                    if lean_backward else extension.backward_chunked_saved
                )
        elif ctx.fixed_dynamics:
            # The fixed recurrence ignores the decay/noise tensors by
            # contract.  Use the matching FIXED reverse kernel even when a
            # caller differentiates the initial state; falling through to
            # the dynamic reverse kernel would backpropagate through inputs
            # that the forward pass deliberately did not read.
            if not lean_backward and not ctx.need_initial_grad and hasattr(
                extension, "backward_chunked_noinitial_fixed"
            ):
                backward = extension.backward_chunked_noinitial_fixed
                ctx.noinitial_backward = True
            else:
                backward = (
                    extension.backward_chunked_lean_fixed
                    if lean_backward else extension.backward_chunked_fixed
                )
        else:
            if not lean_backward and not ctx.need_initial_grad and hasattr(
                extension, "backward_chunked_noinitial"
            ):
                backward = extension.backward_chunked_noinitial
                ctx.noinitial_backward = True
            else:
                backward = (extension.backward_chunked
                            if not lean_backward
                            else getattr(extension, "backward_chunked_lean", extension.backward_chunked))
        if ctx.use_saved_history:
            # The saved-history wrappers retain the final-state gradient
            # arguments in their ABI so the same kernel can serve both the
            # lean and full-dynamics cases.  Lean mode supplies zero state
            # gradients; the kernel omits all corresponding output tensors.
            saved_gm = (
                torch.zeros(ctx.mean_shape, device=device, dtype=keys.dtype)
                if grad_mean is None else grad_mean
            )
            saved_gc = (
                torch.zeros(ctx.covariance_shape, device=device, dtype=keys.dtype)
                if grad_covariance is None else grad_covariance
            )
            gradients = backward(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, boundary_mean,
                boundary_covariance, mp_history, pp_history, u_history,
                beta_history, e_history, grad_output, saved_gm, saved_gc,
                ctx.chunk_length, ctx.denominator_floor,
            )
            if lean_backward:
                # The forward has eight tensor inputs followed by three
                # scalar metadata arguments.  Lean mode returns gradients
                # only for keys/values/queries; the other five tensor inputs
                # and three metadata values still need exactly eight slots.
                return (*gradients, None, None, None, None, None, None, None, None)
            # The no-initial ABI returns only six tensor gradients (keys,
            # values, queries, decay, process, observation); the regular full
            # ABI returns two additional initial-state gradients.
            if ctx.noinitial_backward:
                return (*gradients, None, None, None, None, None)
            return (*gradients, None, None, None)
        if lean_backward:
            # The CUDA lean entry points still keep the full ABI: they accept
            # final-mean/final-covariance gradients even though the specialized
            # kernel does not propagate them or return dynamics/state grads.
            # Supplying no tensors here used to shift ``chunk_length`` into
            # the ``gmf`` slot and raised a pybind argument error whenever a
            # caller used the common readout-only path with detached dynamics.
            # Zero buffers are cheap relative to replay and make the ABI
            # explicit for both dynamic and fixed kernels.
            zero_gm = torch.zeros(ctx.mean_shape, device=device, dtype=keys.dtype)
            zero_gc = torch.zeros(ctx.covariance_shape, device=device, dtype=keys.dtype)
            gradients = backward(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, boundary_mean, boundary_covariance,
                grad_output, zero_gm, zero_gc,
                ctx.chunk_length, ctx.denominator_floor,
            )
            # Three tensor gradients plus five tensor and three scalar
            # non-differentiable inputs = eleven backward slots.
            return (*gradients, None, None, None, None, None, None, None, None)
        gradients = backward(
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
            grad_output, grad_mean, grad_covariance, ctx.chunk_length,
            ctx.denominator_floor,
        )
        # 8 tensor gradients + 3 non-differentiable inputs.
        return (*gradients, None, None, None)


def fused_diagonal_bayesian(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    decay: Tensor,
    process: Tensor,
    observation: Tensor,
    initial_mean: Tensor,
    initial_covariance: Tensor,
    *,
    chunk_length: int,
    denominator_floor: float,
    fixed_dynamics: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Run the exact CUDA core when its fixed-shape contract is satisfied.

    ``None`` means the caller should use the regular PyTorch/Inductor path.
    Inputs are made contiguous here because expanded module buffers have
    stride-zero batch/time dimensions, while the CUDA kernel uses ordinary
    row-major indexing.
    """
    if (
        keys.device.type != "cuda"
        or keys.ndim != 3
        or values.ndim != 3
        or queries.ndim != 3
        or keys.shape[-1] not in (1, 2, 4, 8, 64)
        or (keys.shape[-1], values.shape[-1]) not in ((1, 8), (2, 8), (4, 8), (8, 8), (64, 64))
        or queries.shape[-1] != keys.shape[-1]
        or keys.shape != queries.shape
        or values.shape[:2] != keys.shape[:2]
        or not all(x.dtype == torch.float32 for x in (
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance,
        ))
    ):
        return None
    key_dim = keys.shape[-1]
    extension = _load_extension(key_dim)
    if extension is None:
        return None
    n, t, _ = keys.shape
    if decay.shape != (n, t, key_dim) or process.shape != (n, t) or observation.shape != (n, t):
        return None
    value_dim = values.shape[-1]
    if initial_mean.shape != (n, key_dim, value_dim) or initial_covariance.shape != (n, key_dim, key_dim):
        return None
    # Materialize every broadcasted or view input before both forward and
    # backward.  In particular, module decay/noise buffers usually have
    # stride-zero batch/time dimensions, which the flat CUDA address arithmetic
    # intentionally does not interpret.
    keys = keys.contiguous()
    values = values.contiguous()
    queries = queries.contiguous()
    decay = decay.contiguous()
    process = process.contiguous()
    observation = observation.contiguous()
    initial_mean = initial_mean.contiguous()
    initial_covariance = initial_covariance.contiguous()
    return _ChunkedBayesianFunction.apply(
        keys, values, queries, decay, process, observation,
        initial_mean, initial_covariance,
        int(chunk_length), float(denominator_floor), bool(fixed_dynamics),
    )


__all__ = ["fused_diagonal_bayesian"]
