"""ReplaySSM boundary for the exact dense Bayesian Layer.

ReplaySSM uses the same state equations as :func:`bayesian_layer`, including
the dense covariance update.  It changes the autograd policy: each fixed-size
chunk is evaluated without retaining token-level intermediate tensors and is
replayed under autograd in the reverse pass.  This is the checkpoint/replay
schedule used by recurrent SSM kernels.  It is an exact memory optimization,
not the diagonal-covariance approximation.

The Bayesian update is not an associative elementwise scan: the Kalman gain
depends on the transported covariance and all previous rank-one updates. The
existing chunk transport routine therefore remains the reference for the
forward state transition. A future fused implementation may replace that
routine with a block scan, but this module deliberately does not change the
math to obtain a parallel approximation.
"""
from __future__ import annotations

from numbers import Real
import os
from typing import Optional

import torch
from torch import Tensor, nn

from .bayesian_layer import (
    bayesian_layer,
    _broadcastable,
    _check_finite,
    _validate_covariance,
)


class ReplaySSM(nn.Module):
    """Exact dense Bayesian recurrence with Mamba-style replay backward.

    ``key_dim`` and ``value_dim`` describe the latent matrix state. Inputs are
    ``[..., time, key_dim]`` keys/queries and ``[..., time, value_dim]``
    values. State is reset at each call; callers that stream across batches
    should pass the returned mean/covariance to a lower-level
    :func:`bayesian_layer` call.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        *,
        process_variance: float = 0.05,
        observation_variance: float = 0.2,
        initial_covariance: float = 1.0,
        chunk_length: int = 64,
        denominator_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(key_dim, int) or isinstance(key_dim, bool) or key_dim <= 0:
            raise ValueError("key_dim must be a positive integer")
        if not isinstance(value_dim, int) or isinstance(value_dim, bool) or value_dim <= 0:
            raise ValueError("value_dim must be a positive integer")
        if not isinstance(chunk_length, int) or isinstance(chunk_length, bool) or chunk_length <= 0:
            raise ValueError("chunk_length must be a positive integer")
        for name, value in (
            ("process_variance", process_variance),
            ("observation_variance", observation_variance),
            ("initial_covariance", initial_covariance),
            ("denominator_floor", denominator_floor),
        ):
            if not isinstance(value, Real) or not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
        if process_variance < 0 or observation_variance <= 0 or initial_covariance <= 0:
            raise ValueError("process variance must be nonnegative; observation and initial covariance positive")
        if denominator_floor <= 0:
            raise ValueError("denominator_floor must be positive")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.chunk_length = chunk_length
        self.denominator_floor = float(denominator_floor)
        self.register_buffer("process_variance", torch.tensor(float(process_variance)), persistent=False)
        self.register_buffer("observation_variance", torch.tensor(float(observation_variance)), persistent=False)
        self.register_buffer("initial_covariance", torch.eye(key_dim) * float(initial_covariance), persistent=False)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        *,
        transitions: Optional[Tensor] = None,
        transition_diagonals: Optional[Tensor] = None,
        process_variances: Optional[Tensor | Real] = None,
        observation_variances: Optional[Tensor | Real] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if keys.shape[-1] != self.key_dim or queries.shape[-1] != self.key_dim:
            raise ValueError("keys and queries must end in key_dim")
        if values.shape[-1] != self.value_dim:
            raise ValueError("values must end in value_dim")
        process = self.process_variance if process_variances is None else process_variances
        observation = self.observation_variance if observation_variances is None else observation_variances
        # ``optimized=True`` selects the compiled chunk core; ``replay_ssm``
        # routes each chunk through the custom no-tape/replay autograd
        # boundary. Dense covariance is intentional here.
        return bayesian_layer(
            keys,
            values,
            queries,
            transitions=transitions,
            transition_diagonals=transition_diagonals,
            process_variances=process,
            observation_variances=observation,
            initial_covariance=self.initial_covariance.to(device=keys.device),
            denominator_floor=self.denominator_floor,
            chunk_length=self.chunk_length,
            optimized=True,
            replay_ssm=True,
        )
    def associative_scan(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        *,
        observation_variances: Optional[Tensor | Real] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the CUDA scan for the exact static Bayesian special case.

        The scan is valid only when ``A=I`` and ``Q=0``.  ReplaySSM instances
        configured with process noise reject this path instead of silently
        changing the recurrence.  On CUDA the implementation uses a compiled
        cumsum/prefix-solve graph; the eager information-form scan remains the
        reference for CPU and FP64 checks.
        """
        if bool(self.process_variance.detach().item() != 0.0):
            raise ValueError("associative_scan requires process_variance=0")
        if observation_variances is None:
            observation_variances = self.observation_variance
        scan = (
            associative_static_bayesian_cuda
            if keys.is_cuda else associative_static_bayesian
        )
        return scan(
            keys, values, queries,
            observation_variances=observation_variances,
            initial_covariance=self.initial_covariance.to(device=keys.device),
            denominator_floor=self.denominator_floor,
        )


__all__ = [
    "ReplaySSM",
    "associative_static_bayesian",
    "associative_static_bayesian_cuda",
]


def associative_static_bayesian(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    *,
    observation_variances: Tensor | Real = 1.0,
    initial_mean: Optional[Tensor] = None,
    initial_covariance: Optional[Tensor] = None,
    denominator_floor: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact associative prefix scan for the static, no-process-noise case.

    This is the one dense Bayesian setting where token updates compose as an
    associative operator: dynamics must be identity and ``Q=0``. In
    information form, each observation adds ``k kᵀ / r²`` to precision and
    ``k vᵀ / r²`` to the natural parameter. Prefix sums then recover every
    causal posterior. The result is mathematically equivalent to
    ``bayesian_layer(..., process_variances=0, transitions=None)`` up to solve
    roundoff. General dynamics or nonzero process noise are intentionally not
    routed here because they break this associative decomposition.
    ``observation_variances`` must be finite and at least
    ``denominator_floor`` so the dense recurrence's innovation clamp is
    inactive; otherwise this exact information-form scan raises ``ValueError``.

    The implementation materializes ``[..., T, D, D]`` precision prefixes and
    is a correctness/scan prototype; a production kernel should tile the
    prefix solve or use a block scan.
    """
    if keys.ndim < 2 or values.ndim < 2 or queries.ndim < 2:
        raise ValueError("keys, values and queries must have time and feature dimensions")
    if not all(value.is_floating_point() for value in (keys, values, queries)):
        raise TypeError("keys, values and queries must have floating-point dtype")
    if not (keys.device == values.device == queries.device):
        raise ValueError("keys, values and queries must be on the same device")
    if keys.shape[:-1] != queries.shape[:-1] or keys.shape[:-2] != values.shape[:-2]:
        raise ValueError("keys/queries batch-time dimensions must match values")
    t, d = keys.shape[-2:]
    if values.shape[-2] != t:
        raise ValueError("values and keys must have the same sequence length")
    m = values.shape[-1]
    # FP8 promotion is not implemented on every PyTorch build.  The
    # CroissAttn state boundary is FP32 in that case; retain FP64 when the
    # caller explicitly asks for a high-precision oracle.
    try:
        # Promote every projection input at the CroissAttn state boundary. This
        # keeps FP8/BF16 scans in FP32 while preserving a caller's FP64
        # reference precision when any of K/V/Q is double.
        dtype = torch.promote_types(
            torch.promote_types(keys.dtype, values.dtype),
            torch.promote_types(queries.dtype, torch.float32),
        )
    except (RuntimeError, TypeError):
        dtype = torch.float32
    keys = keys.to(dtype)
    values = values.to(dtype)
    queries = queries.to(dtype)
    batch = keys.shape[:-2]
    device = keys.device
    if initial_covariance is None:
        covariance0 = torch.eye(d, device=device, dtype=dtype).expand(*batch, d, d)
    else:
        covariance0 = initial_covariance.to(device=device, dtype=dtype)
    if covariance0.shape[-2:] != (d, d):
        raise ValueError("initial_covariance must end in [key_dim, key_dim]")
    if not _broadcastable(covariance0.shape[:-2], batch):
        raise ValueError("initial_covariance batch dimensions are not broadcastable to keys")
    covariance0 = covariance0.expand(*batch, d, d)
    _validate_covariance(covariance0, "initial_covariance", d)
    if initial_mean is None:
        mean0 = torch.zeros(*batch, d, m, device=device, dtype=dtype)
    else:
        mean0 = initial_mean.to(device=device, dtype=dtype)
    if mean0.shape[-2:] != (d, m):
        raise ValueError("initial_mean must end in [key_dim, value_dim]")
    if not _broadcastable(mean0.shape[:-2], batch):
        raise ValueError("initial_mean batch dimensions are not broadcastable to keys")
    mean0 = mean0.expand(*batch, d, m)
    _check_finite(mean0, "initial_mean")
    if not isinstance(denominator_floor, Real) or not torch.isfinite(
        torch.tensor(float(denominator_floor))
    ) or denominator_floor <= 0:
        raise ValueError("denominator_floor must be finite and positive")
    r2 = torch.as_tensor(observation_variances, device=device, dtype=dtype)
    r2 = torch.broadcast_to(r2, (*batch, t))
    # The information-form update divides by the observation variance itself.
    # Keep the positive variance unchanged so this path is algebraically the
    # same as the dense Kalman recurrence; its denominator floor applies to
    # the innovation form, not to a valid positive observation variance.
    if r2.numel() and (not bool(torch.isfinite(r2).all()) or
                       bool((r2 < float(denominator_floor)).any())):
        raise ValueError(
            "associative_static_bayesian requires finite observation_variances "
            "greater than or equal to denominator_floor"
        )
    if t == 0:
        return torch.empty(*batch, 0, m, device=device, dtype=dtype), mean0, covariance0
    precision0 = torch.linalg.inv(covariance0)
    natural0 = precision0 @ mean0
    weighted_keys = keys / r2[..., :, None]
    precision_increments = keys[..., :, :, None] * weighted_keys[..., :, None, :]
    natural_increments = weighted_keys[..., :, :, None] * values[..., :, None, :]
    precision = precision0.unsqueeze(-3) + precision_increments.cumsum(dim=-3)
    natural = natural0.unsqueeze(-3) + natural_increments.cumsum(dim=-3)
    # Outer-product accumulation is symmetric mathematically, but making the
    # FP32 matrix explicitly symmetric keeps the Cholesky scan stable on CUDA
    # when long prefixes accumulate roundoff.
    precision = (precision + precision.transpose(-1, -2)) * 0.5
    # The information matrix is symmetric positive definite. Cholesky keeps
    # the scan on the stable FP32 boundary and avoids forming an explicit
    # inverse before solving the natural parameter. It also maps cleanly to
    # batched GPU potrf/trsm kernels.
    chol = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve(natural, chol)
    covariance = torch.cholesky_inverse(chol)
    reads = (mean.transpose(-1, -2) @ queries.unsqueeze(-1)).squeeze(-1)
    return reads, mean[..., -1, :, :], covariance[..., -1, :, :]


_CUDA_STATIC_SCAN_CACHE: dict[tuple[object, ...], object] = {}


def associative_static_bayesian_cuda(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    *,
    observation_variances: Tensor | Real = 1.0,
    initial_mean: Optional[Tensor] = None,
    initial_covariance: Optional[Tensor] = None,
    denominator_floor: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """CUDA prefix-scan prototype for static dense Bayesian memory.

    This is an opt-in compiled CUDA path over the exact information-form scan,
    not a diagonal approximation.  It uses parallel ``cumsum`` followed by
    batched prefix solves and therefore requires identity dynamics and zero
    process noise.  General Kalman dynamics are non-associative and must use
    :class:`ReplaySSM`'s replay recurrence instead.
    """
    if not isinstance(keys, Tensor) or not keys.is_cuda:
        raise ValueError("associative_static_bayesian_cuda requires CUDA tensors")
    if not values.is_cuda or not queries.is_cuda:
        raise ValueError("keys, values, and queries must share a CUDA device")
    if keys.device != values.device or keys.device != queries.device:
        raise ValueError("keys, values, and queries must share a CUDA device")
    key = (
        keys.device.index, keys.dtype, tuple(keys.shape), tuple(values.shape),
        tuple(queries.shape),
        None if initial_mean is None else tuple(initial_mean.shape),
        None if initial_covariance is None else tuple(initial_covariance.shape),
    )
    # Inductor's linalg.solve lowering has produced incorrect broadcasts on
    # some CUDA/PyTorch combinations for unbatched initial covariance. Keep
    # the direct CUDA cumsum/solve graph as the correctness default; users can
    # opt into shape-specialized compilation after validating their image.
    if os.environ.get("REPLAYSSM_COMPILE_SCAN", "0") != "1":
        return associative_static_bayesian(
            keys, values, queries,
            observation_variances=observation_variances,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            denominator_floor=denominator_floor,
        )
    compiled = _CUDA_STATIC_SCAN_CACHE.get(key)
    if compiled is None:
        # The function contains shape/type checks and is intentionally allowed
        # graph breaks; the tensor prefix and solve operations remain in the
        # compiled CUDA graph.  ``max-autotune`` is useful for fixed pretrain
        # shapes, while the eager fallback preserves portability.
        compiled = torch.compile(
            associative_static_bayesian,
            dynamic=False,
            fullgraph=False,
            mode="max-autotune",
        )
        _CUDA_STATIC_SCAN_CACHE[key] = compiled
    # This mode is explicitly opt-in.  Let compile or kernel errors surface so
    # a failed validation cannot be mistaken for a benchmark of the scan.
    return compiled(
        keys, values, queries,
        observation_variances=observation_variances,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        denominator_floor=denominator_floor,
    )
