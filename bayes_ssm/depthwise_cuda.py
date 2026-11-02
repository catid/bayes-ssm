"""Optional SM103 depthwise causal-convolution kernels.

The Bayesian paper path uses independent length-four causal convolutions for
the projected Q/K/V streams.  TransformerEngine produces BF16 activations on
the FP8 path, so this module provides a small BF16 forward/backward kernel for
that exact operation.  It accumulates each output in FP32 and keeps the
parameter and activation dtypes unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

_EXT = None
_ERROR: BaseException | None = None


def _load_extension():
    global _EXT, _ERROR
    if _EXT is not None:
        return _EXT
    if _ERROR is not None or os.environ.get("BAYES_SSM_DISABLE_DEPTHWISE_CUDA", "0") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("depthwise_causal.cu")
        _EXT = load(
            name="bayes_ssm_depthwise_causal",
            sources=[str(source)],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("BAYES_SSM_VERBOSE_BUILD", "0") == "1",
        )
    except BaseException as exc:  # Optional acceleration must never break import.
        _ERROR = exc
        return None
    return _EXT


class _DepthwiseCausalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        ext = _load_extension()
        if ext is None:
            # The caller only selects this function for CUDA BF16 tensors.  A
            # fallback here keeps direct API use safe if compilation fails.
            y = torch.nn.functional.conv1d(
                torch.nn.functional.pad(x.transpose(1, 2), (3, 0)),
                weight[:, None, :], groups=x.shape[-1],
            ).transpose(1, 2)
            ctx.fallback = True
            ctx.save_for_backward(x, weight)
            return y
        x = x.contiguous()
        weight = weight.contiguous()
        y = ext.forward(x, weight)[0]
        ctx.fallback = False
        ctx.save_for_backward(x, weight)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight = ctx.saved_tensors
        if os.environ.get("BAYES_SSM_DEBUG_DEPTHWISE", "0") == "1":
            print("depthwise grad", torch.isfinite(grad_output).all().item(),
                  float(grad_output.detach().abs().nan_to_num().max()),
                  flush=True)
        if ctx.fallback:
            with torch.enable_grad():
                xx = x.detach().requires_grad_()
                ww = weight.detach().requires_grad_()
                yy = torch.nn.functional.conv1d(
                    torch.nn.functional.pad(xx.transpose(1, 2), (3, 0)),
                    ww[:, None, :], groups=xx.shape[-1],
                ).transpose(1, 2)
                gx, gw = torch.autograd.grad(yy, (xx, ww), grad_output)
            return gx, gw
        grad_output = grad_output.contiguous()
        # The custom reverse scan is substantially faster for dX.  For dW,
        # PyTorch's grouped-convolution reduction uses a tuned reduction
        # strategy and avoids the one-block-per-channel kernel in the
        # extension.  Both results have the same BF16 accumulation boundary
        # as the reference Conv1d backward.
        ext = _load_extension()
        if hasattr(ext, "backward_input"):
            gx = ext.backward_input(x, weight, grad_output)[0]
            padded = torch.nn.functional.pad(x.transpose(1, 2), (3, 0))
            gw = torch.nn.grad.conv1d_weight(
                padded, (weight.shape[0], 1, weight.shape[1]),
                grad_output.transpose(1, 2), groups=x.shape[-1]
            ).squeeze(1)
        else:  # pragma: no cover - extensions built before the split entry point
            gx, gw = ext.backward(x, weight, grad_output)
        return gx, gw


def depthwise_causal(x: Tensor, weight: Tensor) -> Tensor:
    """Apply a causal length-four depthwise convolution.

    ``x`` is ``[B,T,C]`` and ``weight`` is ``[C,4]``.  The custom kernel is
    selected only for CUDA BF16 inputs; all other inputs use PyTorch's exact
    reference operation.
    """
    if (
        x.device.type == "cuda" and weight.device.type == "cuda"
        and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
        and x.ndim == 3 and weight.shape == (x.shape[-1], 4)
    ):
        return _DepthwiseCausalFunction.apply(x, weight)
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.transpose(1, 2), (3, 0)),
        weight[:, None, :], groups=x.shape[-1],
    ).transpose(1, 2)


__all__ = ["depthwise_causal"]
