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
        forward_impl = (
            extension.forward_chunked_fixed
            if fixed_dynamics else extension.forward_chunked
        )
        result = forward_impl(
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, int(chunk_length),
            float(denominator_floor),
        )
        output, final_mean, final_covariance, boundary_mean, boundary_covariance = result
        # The clones make the returned tensors independent of extension-owned
        # graph-pool storage during CUDA graph replay.  Boundary tensors remain
        # saved and are consumed by the matching reverse kernel in the same
        # captured step.
        # Save exactly the contiguous buffers passed to the extension.  The
        # projected [B,H,T,D] tensors are commonly reshaped views; saving the
        # original views would make the reverse kernel's flat indexing read
        # unrelated elements even though the forward result was correct.
        ctx.save_for_backward(
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
        )
        ctx.chunk_length = int(chunk_length)
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
        ctx.output_shape = tuple(output.shape)
        ctx.mean_shape = tuple(final_mean.shape)
        ctx.covariance_shape = tuple(final_covariance.shape)
        return output.clone(), final_mean.clone(), final_covariance.clone()

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_covariance):
        (
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
        ) = ctx.saved_tensors
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
        if ctx.fixed_dynamics:
            # The fixed recurrence ignores the decay/noise tensors by
            # contract.  Use the matching FIXED reverse kernel even when a
            # caller differentiates the initial state; falling through to
            # the dynamic reverse kernel would backpropagate through inputs
            # that the forward pass deliberately did not read.
            backward = (
                extension.backward_chunked_lean_fixed
                if lean_backward else extension.backward_chunked_fixed
            )
        else:
            backward = (extension.backward_chunked
                        if not lean_backward
                        else getattr(extension, "backward_chunked_lean", extension.backward_chunked))
        if lean_backward:
            gradients = backward(
                keys, values, queries, decay, process, observation,
                initial_mean, initial_covariance, boundary_mean, boundary_covariance,
                grad_output, ctx.chunk_length, ctx.denominator_floor,
            )
            # 3 tensor gradients + 8 non-differentiable inputs.
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
        or keys.shape[-1] not in (4, 8, 64)
        or (keys.shape[-1], values.shape[-1]) not in ((4, 8), (8, 8), (64, 64))
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
