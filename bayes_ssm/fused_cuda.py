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


_EXTENSION = None
_LOAD_ERROR: Optional[BaseException] = None


def _load_extension():
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        return None
    if os.environ.get("BAYES_SSM_DISABLE_FUSED_CUDA", "0") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("persistent_bayes_chunked.cu")
        _EXTENSION = load(
            name="bayes_ssm_persistent_chunked",
            sources=[str(source)],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("BAYES_SSM_VERBOSE_BUILD", "0") == "1",
        )
    except BaseException as exc:  # optional acceleration must not break fallback
        _LOAD_ERROR = exc
        return None
    return _EXTENSION


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
    ):
        extension = _load_extension()
        if extension is None:
            raise RuntimeError("the fused Bayesian CUDA extension is unavailable")
        n, t, _ = keys.shape
        result = extension.forward_chunked(
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
        ctx.output_shape = tuple(output.shape)
        ctx.mean_shape = tuple(final_mean.shape)
        ctx.covariance_shape = tuple(final_covariance.shape)
        return output.clone(), final_mean.clone(), final_covariance.clone()

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_covariance):
        extension = _load_extension()
        if extension is None:  # pragma: no cover - extension loaded in forward
            raise RuntimeError("the fused Bayesian CUDA extension is unavailable")
        (
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
        ) = ctx.saved_tensors
        device = keys.device
        grad_output = (
            torch.zeros(ctx.output_shape, device=device, dtype=keys.dtype)
            if grad_output is None else grad_output.contiguous()
        )
        grad_mean = (
            torch.zeros(ctx.mean_shape, device=device, dtype=keys.dtype)
            if grad_mean is None else grad_mean.contiguous()
        )
        grad_covariance = (
            torch.zeros(ctx.covariance_shape, device=device, dtype=keys.dtype)
            if grad_covariance is None else grad_covariance.contiguous()
        )
        gradients = extension.backward_chunked(
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance, boundary_mean, boundary_covariance,
            grad_output, grad_mean, grad_covariance, ctx.chunk_length,
            ctx.denominator_floor,
        )
        return (*gradients, None, None)


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
        or keys.shape[-1] != 64
        or values.shape[-1] != 64
        or queries.shape[-1] != 64
        or keys.shape != queries.shape
        or values.shape[:2] != keys.shape[:2]
        or not all(x.dtype == torch.float32 for x in (
            keys, values, queries, decay, process, observation,
            initial_mean, initial_covariance,
        ))
        or process.requires_grad
        or observation.requires_grad
        or initial_mean.requires_grad
        or initial_covariance.requires_grad
    ):
        return None
    extension = _load_extension()
    if extension is None:
        return None
    n, t, _ = keys.shape
    if decay.shape != (n, t, 64) or process.shape != (n, t) or observation.shape != (n, t):
        return None
    if initial_mean.shape != (n, 64, 64) or initial_covariance.shape != (n, 64, 64):
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
        int(chunk_length), float(denominator_floor),
    )


__all__ = ["fused_diagonal_bayesian"]
