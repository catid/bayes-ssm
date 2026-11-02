"""Exact Bayesian Layer recurrence from arXiv:2605.31163.

The implementation deliberately keeps the dense covariance in FP32-friendly torch
operations.  It is an equation-level oracle: gradients are supplied by PyTorch
autograd, and ``forward_chunked`` carries exactly the same belief between chunks.
For production, replace the inner loop with the paper's Triton fused kernel while
keeping this module as the numerical reference.
"""
from __future__ import annotations

import math
import os
from numbers import Real
from typing import Optional, Tuple
import warnings

try:  # Keep imports compatible with older supported PyTorch releases.
    from torch._dynamo.exc import (
        BackendCompilerFailed as _BackendCompilerFailed,
        InternalTorchDynamoError as _InternalTorchDynamoError,
        Unsupported as _DynamoUnsupported,
    )
except ImportError:  # pragma: no cover - exercised only on older torch.
    _BackendCompilerFailed = _InternalTorchDynamoError = _DynamoUnsupported = ()
try:
    from torch._inductor.exc import InductorError as _InductorError
except ImportError:  # pragma: no cover
    _InductorError = ()

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Float8 tensors are valid PyTorch floating tensors, but they are not a safe
# representation for a recurrent mean/covariance.  Keep this list dynamic so
# the module imports on PyTorch builds that expose only a subset of FP8 dtypes.
_FLOAT8_DTYPES = tuple(
    dtype
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if (dtype := getattr(torch, name, None)) is not None
)


def _inv_softplus(value: float) -> float:
    """Stable inverse softplus for positive scalar initialization."""
    if value <= 0 or not math.isfinite(value):
        raise ValueError("softplus initialization must be finite and positive")
    return math.log(math.expm1(value))


def _broadcastable(shape: torch.Size, target: torch.Size) -> bool:
    """Whether ``shape`` can broadcast to ``target`` under PyTorch rules."""
    if len(shape) > len(target):
        return False
    return all(a == b or a == 1 for a, b in zip(reversed(shape), reversed(target)))


def _as_tensor(x: Tensor | Real, *, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    """Convert a scalar/parameter to the state device and dtype."""
    if isinstance(x, Tensor):
        if not x.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        return x.to(device=device, dtype=dtype)
    try:
        result = torch.as_tensor(x, device=device, dtype=dtype)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real scalar, sequence, or tensor; got {type(x).__name__}") from None
    if not result.is_floating_point():
        raise TypeError(f"{name} must contain real values")
    return result


def _as_time_tensor(
    x: Tensor | Real,
    t: int,
    batch_shape: torch.Size,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert a scalar or ``[..., T]`` parameter and validate its batch shape."""
    x = _as_tensor(x, device=device, dtype=dtype, name=name)
    if x.ndim == 0:
        return x.expand(*batch_shape, t)
    if x.shape[-1] != t or not _broadcastable(x.shape[:-1], batch_shape):
        raise ValueError(
            f"{name} must be scalar or have shape [..., T={t}] broadcastable to "
            f"batch shape {tuple(batch_shape)}; got {tuple(x.shape)}"
        )
    return x


def _is_cuda_graph_capturing(x: Tensor) -> bool:
    """Return whether host-side validation must be skipped for ``x``."""
    if not x.is_cuda:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _check_finite(x: Tensor, name: str) -> None:
    # Data-dependent Python checks are intentionally skipped under full-graph
    # compilation; the fused graph will propagate any non-finite values.
    if torch.compiler.is_compiling():
        return
    # Host-visible validation is illegal during CUDA graph capture. Inputs
    # have already been validated during warmup; the captured FP32 recurrence
    # propagates any non-finite value to the graph output.
    if _is_cuda_graph_capturing(x):
        return
    if x.numel() and not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_variance(x: Tensor, name: str, *, strictly_positive: bool) -> None:
    _check_finite(x, name)
    if torch.compiler.is_compiling() or _is_cuda_graph_capturing(x):
        return
    if x.numel() == 0:
        return
    minimum = float(x.detach().min())
    if strictly_positive and minimum <= 0:
        raise ValueError(f"{name} must be strictly positive")
    if not strictly_positive and minimum < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_covariance(x: Tensor, name: str, d: int) -> None:
    if x.shape[-2:] != (d, d):
        raise ValueError(f"{name} must end in {(d, d)}, got {tuple(x.shape)}")
    _check_finite(x, name)
    if torch.compiler.is_compiling() or _is_cuda_graph_capturing(x):
        return
    if not x.numel():
        return
    sym = (x + x.transpose(-1, -2)) * 0.5
    scale = max(1.0, float(sym.detach().abs().max()))
    if not torch.allclose(x, x.transpose(-1, -2), rtol=1e-5, atol=1e-6 * scale):
        raise ValueError(f"{name} must be symmetric")
    minimum = float(torch.linalg.eigvalsh(sym.detach()).min())
    if minimum < -1e-5 * scale:
        raise ValueError(f"{name} must be positive semidefinite")


def _matvec(a: Tensor, x: Tensor) -> Tensor:
    return torch.matmul(a, x.unsqueeze(-1)).squeeze(-1)


def _step(
    mean: Tensor,
    covariance: Tensor,
    key: Tensor,
    value: Tensor,
    query: Tensor,
    transition: Tensor,
    process_covariance: Tensor,
    observation_variance: Tensor,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """One predict/update/read step, matching Eqs. (8--13) of the paper."""
    # Prediction through S_t = A_t S_{t-1} + noise.
    predicted_cov = transition @ covariance @ transition.transpose(-1, -2) + process_covariance
    predicted_mean = transition @ mean

    # Rank-one Kalman conditioning on v_t = S_t^T k_t + noise.
    uncertainty_key = _matvec(predicted_cov, key)
    innovation_variance = observation_variance + torch.sum(key * uncertainty_key, dim=-1)
    innovation_variance = innovation_variance.clamp_min(denominator_floor)
    beta = innovation_variance.reciprocal()
    innovation = value - _matvec(predicted_mean.transpose(-1, -2), key)
    updated_mean = predicted_mean + beta[..., None, None] * uncertainty_key[..., :, None] * innovation[..., None, :]
    updated_cov = predicted_cov - beta[..., None, None] * uncertainty_key[..., :, None] * uncertainty_key[..., None, :]
    # Symmetrization only removes roundoff; mathematically the update is already symmetric.
    updated_cov = (updated_cov + updated_cov.transpose(-1, -2)) * 0.5

    readout = _matvec(updated_mean.transpose(-1, -2), query)
    return updated_mean, updated_cov, readout


def _core_loop(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    transitions: Tensor,
    process_covariances: Tensor,
    observation_variances: Tensor,
    mean: Tensor,
    covariance: Tensor,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Dense sequential core used by the optional compiled execution path."""
    outputs = []
    for i in range(keys.shape[-2]):
        mean, covariance, readout = _step(
            mean, covariance, keys[..., i, :], values[..., i, :], queries[..., i, :],
            transitions[..., i, :, :], process_covariances[..., i, :, :],
            observation_variances[..., i], denominator_floor,
        )
        outputs.append(readout)
    return torch.stack(outputs, dim=-2), mean, covariance


def _core_loop_diagonal_transition(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    transition_diagonals: Tensor,
    process_variances: Tensor,
    observation_variances: Tensor,
    mean: Tensor,
    covariance: Tensor,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Exact dense-covariance core for diagonal transition matrices.

    This is only a dynamics specialization: ``covariance`` remains a full
    dense matrix and every rank-one update is retained.  Scaling rows and
    columns elementwise avoids the two D³ matrix products caused by
    materializing ``diag_embed(A)``.  It therefore preserves the Bayesian
    model while removing the largest cost in the six-layer GDN-style block.
    """
    outputs = []
    for i in range(keys.shape[-2]):
        diagonal = transition_diagonals[..., i, :]
        predicted_mean = diagonal[..., :, None] * mean
        predicted_covariance = diagonal[..., :, None] * covariance * diagonal[..., None, :]
        predicted_covariance = torch.diagonal_scatter(
            predicted_covariance,
            torch.diagonal(predicted_covariance, dim1=-2, dim2=-1)
            + process_variances[..., i, None],
            dim1=-2,
            dim2=-1,
        )
        key = keys[..., i, :]
        uncertainty_key = torch.matmul(predicted_covariance, key.unsqueeze(-1)).squeeze(-1)
        innovation_variance = observation_variances[..., i] + torch.sum(
            key * uncertainty_key, dim=-1
        )
        beta = innovation_variance.clamp_min(denominator_floor).reciprocal()
        innovation = values[..., i, :] - torch.matmul(
            predicted_mean.transpose(-1, -2), key.unsqueeze(-1)
        ).squeeze(-1)
        mean = predicted_mean + beta[..., None, None] * uncertainty_key[..., :, None] * innovation[..., None, :]
        covariance = predicted_covariance - beta[..., None, None] * uncertainty_key[..., :, None] * uncertainty_key[..., None, :]
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        outputs.append(torch.matmul(mean.transpose(-1, -2), queries[..., i, :].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=-2), mean, covariance


def _core_loop_rotation(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    rotation_radii: Tensor,
    rotation_angles: Tensor,
    process_variances: Tensor,
    observation_variances: Tensor,
    mean: Tensor,
    covariance: Tensor,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Exact dense-covariance core for block-diagonal 2-D rotations.

    ``mean`` and ``covariance`` retain the paper's full state dimensions.  The
    only specialization is applying each 2x2 damped rotation by reshaping the
    rows and columns of the dense covariance, avoiding materialization of a
    D-by-D transition matrix and its two cubic matmuls.
    """
    outputs = []
    d = keys.shape[-1]
    pairs = d // 2
    for i in range(keys.shape[-2]):
        radius = rotation_radii[..., i, :]
        angle = rotation_angles[..., i, :]
        radius = radius.clamp_min(torch.finfo(covariance.dtype).tiny)
        radius = torch.where(radius <= 1.0, radius, torch.ones_like(radius))
        c = radius * torch.cos(angle)
        s = radius * torch.sin(angle)
        mean_pairs = mean.reshape(*mean.shape[:-2], pairs, 2, mean.shape[-1])
        mean0, mean1 = mean_pairs[..., 0, :], mean_pairs[..., 1, :]
        predicted_mean = torch.stack(
            (c[..., None] * mean0 - s[..., None] * mean1,
             s[..., None] * mean0 + c[..., None] * mean1), dim=-2
        ).reshape_as(mean)
        # Apply R on covariance rows, then R^T on covariance columns.  This
        # preserves all off-diagonal covariance entries exactly.
        rows = covariance.reshape(*covariance.shape[:-2], pairs, 2, d)
        row0, row1 = rows[..., 0, :], rows[..., 1, :]
        left = torch.stack(
            (c[..., None] * row0 - s[..., None] * row1,
             s[..., None] * row0 + c[..., None] * row1), dim=-2
        ).reshape(*covariance.shape[:-2], d, d)
        cols = left.reshape(*left.shape[:-2], d, pairs, 2)
        col0, col1 = cols[..., 0], cols[..., 1]
        predicted_covariance = torch.stack(
            (c.unsqueeze(-2) * col0 - s.unsqueeze(-2) * col1,
             s.unsqueeze(-2) * col0 + c.unsqueeze(-2) * col1), dim=-1
        ).reshape_as(covariance)
        process_diag = torch.zeros_like(
            torch.diagonal(predicted_covariance, dim1=-2, dim2=-1)
        )
        process_diag[..., 0::2] = process_variances[..., i, :]
        predicted_covariance = torch.diagonal_scatter(
            predicted_covariance,
            torch.diagonal(predicted_covariance, dim1=-2, dim2=-1) + process_diag,
            dim1=-2, dim2=-1,
        )
        key = keys[..., i, :]
        uncertainty_key = torch.matmul(predicted_covariance, key.unsqueeze(-1)).squeeze(-1)
        innovation_variance = observation_variances[..., i] + torch.sum(key * uncertainty_key, dim=-1)
        beta = innovation_variance.clamp_min(denominator_floor).reciprocal()
        innovation = values[..., i, :] - torch.matmul(
            predicted_mean.transpose(-1, -2), key.unsqueeze(-1)
        ).squeeze(-1)
        mean = predicted_mean + beta[..., None, None] * uncertainty_key[..., :, None] * innovation[..., None, :]
        covariance = predicted_covariance - beta[..., None, None] * uncertainty_key[..., :, None] * uncertainty_key[..., None, :]
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        outputs.append(torch.matmul(mean.transpose(-1, -2), queries[..., i, :].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=-2), mean, covariance


_ROTATION_CORES: dict[tuple[object, ...], object] = {}


def _run_compiled_rotation(*args):
    """Compile/cache the compact exact rotational recurrence."""
    mode = _compiled_core_mode()
    key = ("rotation", "mode", mode) + _compiled_core_key(args)
    core = _ROTATION_CORES.get(key)
    if core is None:
        core = torch.compile(_core_loop_rotation, fullgraph=False, dynamic=False, mode=mode)
        _ROTATION_CORES[key] = core
    try:
        return core(*args)
    except _COMPILE_FALLBACK_EXCEPTIONS:
        return _core_loop_rotation(*args)


def _core_loop_fixed_diagonal(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    transition_diagonals: Tensor,
    process_variances: Tensor,
    observation_variances: Tensor,
    mean: Tensor,
    covariance: Tensor,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Exact fixed-dynamics core for the production ``0.97/0.1/1`` case.

    The six-layer recipe uses a constant diagonal transition, isotropic process
    noise, and unit observation noise.  Keeping those constants in the kernel
    removes the per-token dense predicted-covariance temporary and avoids
    reading three broadcast parameter tensors.  ``transition_diagonals`` and
    the noise arguments remain in the signature so the compiled callable has
    the same state/autograd boundary as the general specialization; callers
    select this function only after opting into ``fixed_dynamics``.
    """
    outputs = []
    eye = torch.eye(keys.shape[-1], device=keys.device, dtype=keys.dtype)
    rho = 0.97
    rho2 = rho * rho
    process = 0.1
    observation = 1.0
    for i in range(keys.shape[-2]):
        key = keys[..., i, :]
        predicted_mean = rho * mean
        # Compute P'k without materializing P' = rho²P + qI.  The covariance
        # update below uses the same expression, preserving dense covariance.
        uncertainty_key = rho2 * torch.sum(
            covariance * key[..., None, :], dim=-1
        ) + process * key
        innovation_variance = observation + torch.sum(
            key * uncertainty_key, dim=-1
        )
        beta = innovation_variance.clamp_min(denominator_floor).reciprocal()
        innovation = values[..., i, :] - torch.sum(
            predicted_mean * key[..., :, None], dim=-2
        )
        mean = predicted_mean + beta[..., None, None] * uncertainty_key[..., :, None] * innovation[..., None, :]
        # Every term is symmetric for scalar rho, isotropic process noise, and
        # the rank-one Kalman update.  Skipping a transpose average removes a
        # full D×D pass while preserving the FP32 covariance state.
        covariance = (
            rho2 * covariance
            + process * eye
            - beta[..., None, None] * uncertainty_key[..., :, None] * uncertainty_key[..., None, :]
        )
        outputs.append(torch.sum(mean * queries[..., i, :, None], dim=-2))
    return torch.stack(outputs, dim=-2), mean, covariance


_COMPILED_CORES: dict[tuple[object, ...], object] = {}
_MAX_COMPILED_CORES = 32
_COMPILED_FALLBACK_WARNED: set[tuple[object, ...]] = set()
_COMPILE_FALLBACK_EXCEPTIONS = tuple(
    exception for exception in (
        _BackendCompilerFailed, _InternalTorchDynamoError, _DynamoUnsupported,
        _InductorError, NotImplementedError,
    ) if isinstance(exception, type)
)


def _compiled_core_mode() -> str:
    """Compiler mode for recurrent cores; default stays conservative for training.

    ``reduce-overhead`` enables CUDA graphs and can help B300 inference or a
    single fixed-shape training graph.  The state boundary clones below make
    the mode safe for the recurrent state, but callers with many independent
    heads should benchmark it against the default before selecting it.
    """
    mode = os.environ.get("MEMORY_BY_DESIGN_COMPILE_MODE", "default")
    if mode not in {"default", "reduce-overhead", "max-autotune"}:
        warnings.warn(
            f"unknown MEMORY_BY_DESIGN_COMPILE_MODE={mode!r}; using default",
            RuntimeWarning,
            stacklevel=2,
        )
        return "default"
    return mode


def _compiled_outputs_need_clone() -> bool:
    """Return whether Inductor may recycle compiled output storage.

    The recurrent state is consumed by the next chunk and by autograd, so it
    must be cloned when CUDA graphs own the output buffers.  The normal
    Inductor backend in the training configuration has CUDA graphs disabled;
    cloning every state tensor there adds a device-to-device copy at every
    chunk boundary.  Keep a conservative ``True`` fallback for older/private
    configurations where the setting cannot be inspected.
    """
    # ``torch.compile(mode=...)`` installs CUDA-graph settings as per-call
    # config patches; the global ``torch._inductor.config.triton.cudagraphs``
    # flag therefore remains false even in ``reduce-overhead`` mode.  Decide
    # from the mode that owns the compiled callable instead of inspecting the
    # unpatched global configuration.
    return _compiled_core_mode() != "default"


def _checkpointed_backward_enabled() -> bool:
    """Enable the Mamba-style replay boundary for optimized training.

    Replay is opt-in until a fused reverse kernel is available.  It reduces
    reserved activation memory, but the current implementation replays the
    compiled chunk through autograd and is slower on the measured GB300
    training workload.  Set ``MEMORY_BY_DESIGN_CHECKPOINTED_BACKWARD=1`` for
    memory-limited jobs.
    """
    return os.environ.get("MEMORY_BY_DESIGN_CHECKPOINTED_BACKWARD", "0") == "1"


def _compiled_core_key(args: tuple[object, ...]) -> tuple[object, ...]:
    """Build a stable specialization key for the recurrent core.

    MQAR and language-model evaluation intentionally mix sequence lengths and
    batch tails.  A single ``torch.compile`` callable with ``dynamic=False``
    then accumulates shape-specialized graphs until Dynamo's cache limit and
    silently falls back to eager execution.  Keeping a small callable cache
    gives each common shape its own graph and avoids that global limit while
    retaining static-shape kernel optimization.
    """
    # Dynamo specializes the recurrence on grad mode. Training and evaluation
    # therefore need separate callables; otherwise the first eval batch
    # recompiles the training graph (and can hit the recompile limit).
    key: list[object] = ["grad_enabled", bool(torch.is_grad_enabled())]
    for value in args:
        if isinstance(value, Tensor):
            key.extend((value.device.type, value.device.index, value.dtype,
                        tuple(value.shape), tuple(value.stride()),
                        bool(value.requires_grad)))
        else:
            key.append(value)
    return tuple(key)


def _run_compiled_core(*args):
    # The compiler mode changes graph ownership and kernel selection. Include
    # it in the cache key so an in-process ablation cannot accidentally reuse
    # a graph compiled under a previous environment setting.
    mode = _compiled_core_mode()
    key = ("mode", mode) + _compiled_core_key(args)
    compiled_core = _COMPILED_CORES.get(key)
    if compiled_core is None:
        # Keep CUDA graphs disabled here.  A training step can invoke this core
        # repeatedly for independent Bayesian heads; graph-owned intermediates
        # may otherwise be overwritten before autograd consumes them.  The
        # default Inductor mode still fuses the recurrence without unsafe graph
        # storage reuse.  Users benchmarking inference can wrap this module in
        # their own ``torch.compile(..., mode='reduce-overhead')`` call.
        compiled_core = torch.compile(
            _core_loop, fullgraph=False, dynamic=False, mode=mode
        )
        _COMPILED_CORES[key] = compiled_core
        if len(_COMPILED_CORES) > _MAX_COMPILED_CORES:
            # Dict insertion order gives a deterministic FIFO eviction.  A
            # compiled callable already referenced by an autograd graph stays
            # alive until that graph releases it.
            _COMPILED_CORES.pop(next(iter(_COMPILED_CORES)))
    try:
        needs_clone = _compiled_outputs_need_clone()
        mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark_step_begin is not None and args[0].is_cuda and needs_clone:
            mark_step_begin()
        # Inductor's CUDA-graph backend may recycle output storage between
        # launches. Clone at the boundary in that mode because the next chunk
        # consumes the state returned by this one and must not retain
        # graph-owned buffers.  The regular training backend has CUDA graphs
        # disabled, so avoid an unnecessary device-to-device copy there.
        output = compiled_core(*args)
        if needs_clone:
            return tuple(value.clone() for value in output)
        return output
    except _COMPILE_FALLBACK_EXCEPTIONS as exc:
        if _is_cuda_graph_capturing(args[0]) and _compiled_core_mode() != "default":
            raise RuntimeError(
                "Bayesian compiled core fell back while a CUDA graph was being captured"
            ) from exc
        # Some CUDA/Inductor versions cannot compile every dtype or operator
        # used by a caller.  Keep the numerical reference fallback, but make
        # it visible: silently switching to the Python recurrence makes an
        # ``optimized=True`` benchmark report the wrong implementation and
        # can hide a genuine OOM or kernel failure.
        if key not in _COMPILED_FALLBACK_WARNED:
            _COMPILED_FALLBACK_WARNED.add(key)
            warnings.warn(
                "BayesianLayer optimized core fell back to eager execution "
                f"for shape key {key!r}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return _core_loop(*args)


def _run_compiled_diagonal_transition(*args):
    """Compile/cache the exact dense-covariance diagonal-A specialization."""
    mode = _compiled_core_mode()
    key = ("diagonal-transition", "mode", mode) + _compiled_core_key(args)
    compiled_core = _COMPILED_CORES.get(key)
    if compiled_core is None:
        compiled_core = torch.compile(
            _core_loop_diagonal_transition, fullgraph=False, dynamic=False, mode=mode
        )
        _COMPILED_CORES[key] = compiled_core
        if len(_COMPILED_CORES) > _MAX_COMPILED_CORES:
            _COMPILED_CORES.pop(next(iter(_COMPILED_CORES)))
    try:
        needs_clone = _compiled_outputs_need_clone()
        mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark_step_begin is not None and args[0].is_cuda and needs_clone:
            mark_step_begin()
        output = compiled_core(*args)
        if needs_clone:
            return tuple(value.clone() for value in output)
        return output
    except _COMPILE_FALLBACK_EXCEPTIONS as exc:
        if _is_cuda_graph_capturing(args[0]) and _compiled_core_mode() != "default":
            raise RuntimeError(
                "Bayesian diagonal core fell back while a CUDA graph was being captured"
            ) from exc
        if key not in _COMPILED_FALLBACK_WARNED:
            _COMPILED_FALLBACK_WARNED.add(key)
            warnings.warn(
                "BayesianLayer exact diagonal-transition core fell back to eager execution "
                f"for shape key {key!r}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return _core_loop_diagonal_transition(*args)


def _run_compiled_fixed_diagonal(*args):
    """Compile/cache the constant-dynamics diagonal specialization."""
    mode = _compiled_core_mode()
    key = ("fixed-diagonal", "mode", mode) + _compiled_core_key(args)
    compiled_core = _COMPILED_CORES.get(key)
    if compiled_core is None:
        compiled_core = torch.compile(
            _core_loop_fixed_diagonal, fullgraph=False, dynamic=False, mode=mode
        )
        _COMPILED_CORES[key] = compiled_core
        if len(_COMPILED_CORES) > _MAX_COMPILED_CORES:
            _COMPILED_CORES.pop(next(iter(_COMPILED_CORES)))
    try:
        needs_clone = _compiled_outputs_need_clone()
        mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark_step_begin is not None and args[0].is_cuda and needs_clone:
            mark_step_begin()
        output = compiled_core(*args)
        if needs_clone:
            return tuple(value.clone() for value in output)
        return output
    except _COMPILE_FALLBACK_EXCEPTIONS as exc:
        if _is_cuda_graph_capturing(args[0]) and _compiled_core_mode() != "default":
            raise RuntimeError(
                "Bayesian fixed diagonal core fell back while a CUDA graph was being captured"
            ) from exc
        if key not in _COMPILED_FALLBACK_WARNED:
            _COMPILED_FALLBACK_WARNED.add(key)
            warnings.warn(
                "Bayesian fixed diagonal core fell back to eager execution "
                f"for shape key {key!r}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return _core_loop_fixed_diagonal(*args)


class _CheckpointedCoreFunction(torch.autograd.Function):
    """Mamba-style custom autograd boundary for one dense Bayesian chunk.

    Mamba-3 does not let autograd retain every operation in its recurrent
    kernel.  It saves the chunk inputs and a compact state boundary, then
    replays the chunk during the reverse pass.  The Bayesian equations remain
    the existing compiled reference; this wrapper changes only the tape
    policy, so it is exact up to the same FP32 roundoff as the normal path.
    """

    @staticmethod
    def forward(ctx, *args):
        # The final argument is a Python float and is not part of
        # save_for_backward.  Calling the compiled core under no_grad is the
        # important part: no token-level autograd graph is retained.
        *tensors, denominator_floor = args
        with torch.no_grad():
            output = _run_compiled_core(*tensors, denominator_floor)
        ctx.save_for_backward(*tensors)
        ctx.denominator_floor = float(denominator_floor)
        ctx.needs = tuple(ctx.needs_input_grad[:-1])
        return output

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_covariance):
        saved = ctx.saved_tensors
        needs = ctx.needs
        differentiable = []
        inputs = []
        for value, need in zip(saved, needs, strict=True):
            item = value.detach().requires_grad_(need)
            inputs.append(item)
            if need:
                differentiable.append(item)
        if not differentiable:
            return (*([None] * len(saved)), None)
        with torch.enable_grad():
            output, final_mean, final_covariance = _run_compiled_core(
                *inputs, ctx.denominator_floor,
            )
            gradients_used = torch.autograd.grad(
                (output, final_mean, final_covariance), differentiable,
                grad_outputs=(
                    torch.zeros_like(output) if grad_output is None else grad_output,
                    torch.zeros_like(final_mean) if grad_mean is None else grad_mean,
                    torch.zeros_like(final_covariance)
                    if grad_covariance is None else grad_covariance,
                ), allow_unused=True,
            )
        gradient_iter = iter(gradients_used)
        gradients = tuple(next(gradient_iter) if need else None for need in needs)
        return (*gradients, None)


def _run_chunked(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    transitions: Tensor,
    process_covariances: Tensor,
    observation_variances: Tensor,
    mean: Tensor,
    covariance: Tensor,
    *,
    chunk_length: int,
    denominator_floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Run the exact chunk algorithm from paper §C.2.

    ``prior_covariance`` carries :math:`\Pi_t`, the covariance with the current
    chunk's observations removed.  ``directions`` stores the transported
    rank-one update directions :math:`w_{t,s}` and therefore avoids constructing
    a dense posterior covariance at every token.  This remains a readable
    PyTorch reference (the paper replaces these loops with Triton tiles).
    """
    t = keys.shape[-2]
    outputs = []
    for start in range(0, t, chunk_length):
        stop = min(t, start + chunk_length)
        prior_mean, prior_covariance = mean, covariance
        directions: list[Tensor] = []
        betas: list[Tensor] = []
        chunk_outputs = []
        for i in range(start, stop):
            transition = transitions[..., i, :, :]
            prior_covariance = transition @ prior_covariance @ transition.transpose(-1, -2)
            prior_covariance = prior_covariance + process_covariances[..., i, :, :]
            prior_covariance = (prior_covariance + prior_covariance.transpose(-1, -2)) * 0.5
            prior_mean = transition @ prior_mean

            # Transport all earlier posterior update directions to this step.
            directions = [_matvec(transition, direction) for direction in directions]
            key = keys[..., i, :]
            value = values[..., i, :]
            query = queries[..., i, :]
            uncertainty_key = _matvec(prior_covariance, key)
            for direction, beta in zip(directions, betas):
                uncertainty_key = uncertainty_key - beta[..., None] * direction * torch.sum(direction * key, dim=-1)[..., None]
            innovation_variance = observation_variances[..., i] + torch.sum(key * uncertainty_key, dim=-1)
            innovation_variance = innovation_variance.clamp_min(denominator_floor)
            beta = innovation_variance.reciprocal()
            innovation = value - _matvec(prior_mean.transpose(-1, -2), key)
            prior_mean = prior_mean + beta[..., None, None] * uncertainty_key[..., :, None] * innovation[..., None, :]
            readout = _matvec(prior_mean.transpose(-1, -2), query)
            chunk_outputs.append(readout)
            directions.append(uncertainty_key)
            betas.append(beta)

        # At chunk exit, the transported history is the posterior covariance's
        # rank-one downdate from the unconditioned covariance Pi_L.
        covariance = prior_covariance
        for direction, beta in zip(directions, betas):
            covariance = covariance - beta[..., None, None] * direction[..., :, None] * direction[..., None, :]
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        mean = prior_mean
        outputs.extend(chunk_outputs)
    return torch.stack(outputs, dim=-2), mean, covariance


def bayesian_layer(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    *,
    transitions: Optional[Tensor] = None,
    transition_diagonals: Optional[Tensor] = None,
    rotation_radii: Optional[Tensor] = None,
    rotation_angles: Optional[Tensor] = None,
    process_variances: Optional[Tensor | Real] = None,
    process_covariances: Optional[Tensor] = None,
    observation_variances: Optional[Tensor | Real] = None,
    initial_mean: Optional[Tensor] = None,
    initial_covariance: Optional[Tensor] = None,
    denominator_floor: float = 1e-6,
    chunk_length: Optional[int] = None,
    optimized: bool = False,
    replay_ssm: bool = False,
    fixed_dynamics: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run an exact Bayesian Layer over a sequence.

    Args:
        keys: ``[..., T, D]`` key vectors.
        values: ``[..., T, m]`` observed values.
        queries: ``[..., T, D]`` read queries.
        transitions: ``[..., T, D, D]`` dynamics matrices. Defaults to identity.
        transition_diagonals: optional ``[..., T, D]`` diagonal dynamics. This is
            mutually exclusive with ``transitions`` and enables a compiled
            diagonal specialization without materializing ``D×D`` matrices.
        rotation_radii/rotation_angles: optional ``[..., T, D/2]`` damped 2-D
            rotational dynamics. Adjacent coordinates use ``rho R(theta)``;
            process noise is injected into the first coordinate of each pair.
        process_variances: ``[..., T]`` isotropic process variances (``ell_t^2``).
        process_covariances: optional full ``[..., T, D, D]`` process covariance;
            mutually exclusive with ``process_variances``.
        observation_variances: scalar or ``[..., T]`` ``r_t^2``; defaults to 1.
        initial_mean: ``[..., D, m]``; defaults to zero.
        initial_covariance: ``[..., D, D]``; defaults to identity.
        chunk_length: carry state across chunks of this size. This is exact and
            useful for validating a fused/chunked implementation.
        optimized: use a lazily compiled dense core for each chunk. This preserves
            the same recurrence (within floating-point roundoff) but does not
            replace the paper's transported-history kernel.
        replay_ssm: use the Mamba-style replay boundary for the exact dense
            recurrence when ``optimized=True``. Each chunk saves only its
            inputs and entry mean/covariance, runs forward without a token
            autograd tape, and replays that chunk during backward. This keeps
            the full covariance equations. The environment variable
            ``MEMORY_BY_DESIGN_CHECKPOINTED_BACKWARD=1`` enables the same
            boundary globally.
        fixed_dynamics: assert the compile-time ``A=0.97I, Q=0.1I, R=1``
            dynamics used by the six-layer recipe. This selects the exact
            scalar-dynamics specialization for Inductor and the fused CUDA
            path; callers using other dynamics must leave it false.

    Inputs with float16, bfloat16, or any available float8 dtype are promoted
    to float32 for the recurrent state and readouts.  This is intentional: the
    paper keeps the covariance and chunk-boundary state in FP32 for numerical
    stability, while allowing low-precision projection outputs at the boundary.

    Returns ``(readouts, final_mean, final_covariance)``.
    """
    if not all(isinstance(x, Tensor) for x in (keys, values, queries)):
        raise TypeError("keys, values and queries must be torch tensors")
    if keys.ndim < 2 or values.ndim < 2 or queries.ndim < 2:
        raise ValueError("keys, values and queries must have a time and feature dimension")
    if not all(x.is_floating_point() for x in (keys, values, queries)):
        raise TypeError("keys, values and queries must have floating-point dtype")
    if not (keys.device == values.device == queries.device):
        raise ValueError("keys, values and queries must be on the same device")
    if keys.shape[:-1] != queries.shape[:-1] or keys.shape[:-2] != values.shape[:-2]:
        raise ValueError("keys/queries batch and time dimensions must match values")
    t, d = keys.shape[-2:]
    m = values.shape[-1]
    if d <= 0 or m <= 0:
        raise ValueError("key and value dimensions must be positive")
    if values.shape[-2] != t:
        raise ValueError("values and keys must have the same sequence length")
    batch_shape = keys.shape[:-2]
    device, input_dtype = keys.device, keys.dtype
    # The paper keeps recurrent means/covariances and chunk boundaries in FP32.
    # Promote low precision inputs for the state update; casts remain differentiable.
    # CroissAttn's stability boundary applies to FP8 as well as BF16/FP16:
    # low-precision projections may feed the recurrence, but its state,
    # covariance, denominators, and reductions must remain FP32.  Without the
    # explicit FP8 cases below, a direct FP8 caller would accidentally retain
    # an FP8 covariance and either overflow or fail in matmul.
    low_precision_dtypes = (torch.float16, torch.bfloat16, *_FLOAT8_DTYPES)
    dtype = torch.float32 if input_dtype in low_precision_dtypes else input_dtype
    keys = keys.to(dtype)
    values = values.to(dtype)
    queries = queries.to(dtype)
    _check_finite(keys, "keys")
    _check_finite(values, "values")
    _check_finite(queries, "queries")
    if (rotation_radii is None) != (rotation_angles is None):
        raise ValueError("rotation_radii and rotation_angles must be supplied together")
    if transitions is not None and transition_diagonals is not None:
        raise ValueError("pass transitions or transition_diagonals, not both")
    if (rotation_radii is not None or rotation_angles is not None) and (
        transitions is not None or transition_diagonals is not None
    ):
        raise ValueError("rotation dynamics are mutually exclusive with transitions and transition_diagonals")
    if not isinstance(replay_ssm, bool):
        raise TypeError("replay_ssm must be a bool")
    if not isinstance(fixed_dynamics, bool):
        raise TypeError("fixed_dynamics must be a bool")
    # ``transition_diagonals`` is a compact exact-dynamics representation. The
    # specialized core below still carries a full dense covariance; it only
    # avoids materializing diagonal A and its redundant matrix products.
    diagonal_transition = None
    rotation_transition = None
    if rotation_radii is not None:
        if d % 2:
            raise ValueError("rotational dynamics require an even key dimension")
        for name, value in (("rotation_radii", rotation_radii), ("rotation_angles", rotation_angles)):
            if not isinstance(value, Tensor) or value.ndim < 2:
                raise ValueError(f"{name} must have shape [..., T={t}, D/2={d // 2}]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.shape[-2:] != (t, d // 2):
                raise ValueError(f"{name} must end in {(t, d // 2)}, got {tuple(value.shape)}")
            if not _broadcastable(value.shape[:-2], batch_shape):
                raise ValueError(f"{name} batch dimensions are not broadcastable to input batch")
            value = value.to(device=device, dtype=dtype)
            _check_finite(value, name)
            if name == "rotation_radii":
                rotation_radii = value
            else:
                rotation_angles = value
        rotation_transition = True
        transitions = None
    elif transition_diagonals is not None:
        if not isinstance(transition_diagonals, Tensor) or transition_diagonals.ndim < 2:
            raise ValueError(f"transition_diagonals must have shape [..., T={t}, D={d}]")
        if not transition_diagonals.is_floating_point():
            raise TypeError("transition_diagonals must have a floating-point dtype")
        transition_diagonals = transition_diagonals.to(device=device, dtype=dtype)
        if transition_diagonals.shape[-2:] != (t, d):
            raise ValueError(f"transition_diagonals must end in {(t, d)}, got {tuple(transition_diagonals.shape)}")
        if not _broadcastable(transition_diagonals.shape[:-2], batch_shape):
            raise ValueError("transition_diagonals batch dimensions are not broadcastable to the input batch")
        _check_finite(transition_diagonals, "transition_diagonals")
        diagonal_transition = transition_diagonals
        transitions = None
    elif transitions is None:
        eye = torch.eye(d, device=device, dtype=dtype)
        transitions = eye.expand(*batch_shape, t, d, d)
    else:
        if not isinstance(transitions, Tensor) or transitions.ndim < 3:
            raise ValueError(f"transitions must have shape [..., T={t}, D={d}, D={d}]")
        if not transitions.is_floating_point():
            raise TypeError("transitions must have a floating-point dtype")
        transitions = transitions.to(device=device, dtype=dtype)
        if transitions.shape[-3:] != (t, d, d):
            raise ValueError(f"transitions must end in {(t, d, d)}, got {tuple(transitions.shape)}")
        if not _broadcastable(transitions.shape[:-3], batch_shape):
            raise ValueError("transitions batch dimensions are not broadcastable to the input batch")
        _check_finite(transitions, "transitions")
    if rotation_transition:
        if process_covariances is not None:
            raise ValueError("rotational dynamics require per-pair process_variances, not process_covariances")
        if process_variances is None:
            process_variances = torch.zeros(*batch_shape, t, d // 2, device=device, dtype=dtype)
        else:
            process_variances = _as_tensor(
                process_variances, device=device, dtype=dtype, name="process_variances"
            )
            if process_variances.ndim == 0:
                process_variances = process_variances.expand(*batch_shape, t, d // 2)
            elif process_variances.shape[-2:] == (t, d // 2) and _broadcastable(
                process_variances.shape[:-2], batch_shape
            ):
                process_variances = process_variances.expand(*batch_shape, t, d // 2)
            else:
                raise ValueError(
                    f"rotational process_variances must be scalar or end in {(t, d // 2)}"
                )
            _validate_variance(process_variances, "process_variances", strictly_positive=False)
        process_covariances = None
    elif process_covariances is not None and process_variances is not None:
        raise ValueError("pass process_covariances or process_variances, not both")
    if rotation_transition:
        # Rotational process variances are stored per 2-D state pair and are
        # consumed directly by the structured recurrence below.
        pass
    elif process_covariances is not None:
        if not isinstance(process_covariances, Tensor) or process_covariances.ndim < 3:
            raise ValueError(f"process_covariances must have shape [..., T={t}, D={d}, D={d}]")
        if not process_covariances.is_floating_point():
            raise TypeError("process_covariances must have a floating-point dtype")
        process_covariances = process_covariances.to(device=device, dtype=dtype)
        if process_covariances.shape[-3:] != (t, d, d):
            raise ValueError(f"process_covariances must end in {(t, d, d)}, got {tuple(process_covariances.shape)}")
        if not _broadcastable(process_covariances.shape[:-3], batch_shape):
            raise ValueError("process_covariances batch dimensions are not broadcastable to the input batch")
        _validate_covariance(process_covariances, "process_covariances", d)
    elif not rotation_transition:
        if process_variances is None:
            process_variances = 0.0
        process_variances = _as_time_tensor(
            process_variances, t, batch_shape, "process_variances", device=device, dtype=dtype
        )
        _validate_variance(process_variances, "process_variances", strictly_positive=False)
        if optimized and diagonal_transition is not None:
            process_covariances = None
        else:
            eye = torch.eye(d, device=device, dtype=dtype)
            process_covariances = process_variances[..., None, None] * eye
    if observation_variances is None:
        observation_variances = 1.0
    observation_variances = _as_time_tensor(
        observation_variances, t, batch_shape, "observation_variances", device=device, dtype=dtype
    )
    _validate_variance(observation_variances, "observation_variances", strictly_positive=True)

    if initial_mean is None:
        mean = torch.zeros(*batch_shape, d, m, device=device, dtype=dtype)
    else:
        if not isinstance(initial_mean, Tensor) or initial_mean.ndim < 2:
            raise ValueError(f"initial_mean must have shape [..., D={d}, m={m}]")
        if not initial_mean.is_floating_point():
            raise TypeError("initial_mean must have a floating-point dtype")
        mean = initial_mean.to(device=device, dtype=dtype)
        if mean.shape[-2:] != (d, m) or not _broadcastable(mean.shape[:-2], batch_shape):
            raise ValueError("initial_mean has incompatible dimensions or batch shape")
        _check_finite(mean, "initial_mean")
        mean = mean.expand(*batch_shape, d, m)
    if initial_covariance is None:
        covariance = torch.eye(d, device=device, dtype=dtype).expand(*batch_shape, d, d).clone()
    else:
        if not isinstance(initial_covariance, Tensor) or initial_covariance.ndim < 2:
            raise ValueError(f"initial_covariance must have shape [..., D={d}, D={d}]")
        if not initial_covariance.is_floating_point():
            raise TypeError("initial_covariance must have a floating-point dtype")
        covariance = initial_covariance.to(device=device, dtype=dtype)
        if not _broadcastable(covariance.shape[:-2], batch_shape):
            raise ValueError("initial_covariance batch dimensions are not broadcastable to the input batch")
        _validate_covariance(covariance, "initial_covariance", d)
        covariance = covariance.expand(*batch_shape, d, d)


    if not isinstance(denominator_floor, Real) or not math.isfinite(float(denominator_floor)):
        raise ValueError("denominator_floor must be finite")
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    if chunk_length is not None and (not isinstance(chunk_length, int) or isinstance(chunk_length, bool) or chunk_length <= 0):
        raise ValueError("chunk_length must be a positive integer or None")
    if t == 0:
        return torch.empty(*batch_shape, 0, m, device=device, dtype=dtype), mean, covariance
    # The replay autograd boundary saves its dense transition/process inputs.
    # Materialize the exact representation in that opt-in mode so gradients
    # through diagonal dynamics and process noise are not dropped.
    if diagonal_transition is not None and (
        not optimized
        or (torch.is_grad_enabled() and (replay_ssm or _checkpointed_backward_enabled()))
    ):
        transitions = torch.diag_embed(diagonal_transition)
        if process_covariances is None:
            eye = torch.eye(d, device=device, dtype=dtype)
            process_covariances = process_variances[..., None, None] * eye
        diagonal_transition = None
    if rotation_transition and not optimized:
        # The eager oracle uses the same generic dense recurrence.  The
        # optimized path above avoids these temporaries with the structured
        # row/column rotation kernel.
        radius = rotation_radii.clamp_min(torch.finfo(dtype).tiny)
        radius = torch.where(radius <= 1.0, radius, torch.ones_like(radius))
        transitions = make_rotation_dynamics(torch.log(radius), rotation_angles)
        process_covariances = make_rotation_process_covariances(process_variances)
        rotation_transition = None
    if chunk_length is None:
        # A missing chunk length means one full sequence chunk. Use explicit
        # ``chunk_length=1`` when a token-by-token oracle is desired.
        chunk_length = t
    if optimized:
        if rotation_transition:
            # Compact rotational specialization: retain dense covariance but
            # apply each 2x2 block by row/column reshapes instead of forming
            # dense transition and process-covariance tensors.
            outputs = []
            for start in range(0, t, chunk_length):
                stop = min(t, start + chunk_length)
                mean = mean.contiguous()
                covariance = covariance.contiguous()
                core_args = (
                    keys[..., start:stop, :], values[..., start:stop, :],
                    queries[..., start:stop, :], rotation_radii[..., start:stop, :],
                    rotation_angles[..., start:stop, :],
                    process_variances[..., start:stop, :],
                    observation_variances[..., start:stop], mean, covariance,
                    float(denominator_floor),
                )
                output, mean, covariance = _run_compiled_rotation(*core_args)
                outputs.append(output)
            return torch.cat(outputs, dim=-2), mean, covariance
        # The six-layer production configuration uses 64-dimensional heads
        # with diagonal dynamics and scalar isotropic process noise.  Dispatch
        # that exact case to the optional persistent CUDA kernel before the
        # general Inductor path.  The extension keeps only chunk-boundary
        # states and recomputes each chunk in reverse, so it is safe for
        # training and CUDA graph capture while retaining the full covariance.
        # Other dimensions, dense transitions, learnable noise, and CPU runs
        # continue through the reference implementation below.
        if (
            device.type == "cuda"
            and len(batch_shape) == 1
            and ((d == 64 and m == 64) or (d == 8 and m == 8)
                 or (d == 4 and m == 8) or (d == 2 and m == 8)
                 or (d == 1 and m == 8))
            and diagonal_transition is not None
            and process_covariances is None
        ):
            try:
                from .fused_cuda import fused_diagonal_bayesian

                fused = fused_diagonal_bayesian(
                    keys, values, queries, diagonal_transition,
                    process_variances, observation_variances, mean, covariance,
                    chunk_length=chunk_length,
                    denominator_floor=float(denominator_floor),
                    fixed_dynamics=fixed_dynamics,
                )
            except RuntimeError as exc:
                # A compiler/toolkit mismatch should leave the validated
                # Inductor path available rather than making the public API
                # unusable.  The optional loader caches the failure, so later
                # calls do not repeatedly attempt a broken build.
                if os.environ.get("BAYES_SSM_VERBOSE_BUILD", "0") == "1":
                    warnings.warn(
                        f"fused Bayesian CUDA core unavailable; using Inductor: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                fused = None
            if fused is not None:
                return fused
        # Keep the recurrent state autograd metadata stable across chunks.  The
        # first chunk starts from constant initial state, while later chunks
        # receive state tensors whose values depend on the input.  Inductor
        # otherwise recompiles the same core for ``requires_grad`` changes on
        # the state (especially in mixed MQAR training), erasing the benefit of
        # compilation.  A zero-valued input dependency preserves exact values
        # while giving every chunk the same autograd metadata.
        # Include every differentiable input, rather than only Q/K/V: callers
        # may learn dynamics or noise while feeding detached activations.
        # Otherwise the first chunk starts with constant state metadata and the
        # next chunk receives state that requires gradients through A/Q/R,
        # causing another avoidable Inductor specialization.
        checkpointed = bool(
            torch.is_grad_enabled()
            and (_checkpointed_backward_enabled() or replay_ssm)
        )
        if torch.is_grad_enabled() and not checkpointed:
            differentiable_inputs = (
                keys, values, queries, transitions, process_covariances,
                observation_variances, mean, covariance,
            )
            # Slice before summing so an empty batch remains valid while the
            # anchor still touches only one element for large state tensors.
            anchors = [item.reshape(-1)[:1].sum() for item in differentiable_inputs
                       if isinstance(item, Tensor) and item.requires_grad]
            if anchors:
                state_anchor = torch.stack(anchors).sum() * 0
                mean = mean + state_anchor
                covariance = covariance + state_anchor
        outputs = []
        for start in range(0, t, chunk_length):
            stop = min(t, start + chunk_length)
            # Expanded initial states have zero batch strides, while the
            # compiled recurrence returns ordinary contiguous states. Normalize
            # those two layouts before dispatch so Dynamo does not recompile
            # the same graph on the first chunk of every sequence.
            mean = mean.contiguous()
            covariance = covariance.contiguous()
            if diagonal_transition is not None and process_covariances is None and not checkpointed:
                core_args = (
                    keys[..., start:stop, :], values[..., start:stop, :],
                    queries[..., start:stop, :], diagonal_transition[..., start:stop, :],
                    process_variances[..., start:stop],
                    observation_variances[..., start:stop], mean, covariance,
                    float(denominator_floor),
                )
                if (
                    fixed_dynamics
                    and not diagonal_transition.requires_grad
                    # The custom reverse kernel also computes exact process
                    # and observation-noise gradients. Keep the dynamic path
                    # opt-in until validated on the target CUDA toolchain.
                    and (
                        (
                            not process_variances.requires_grad
                            and not observation_variances.requires_grad
                        )
                        or os.environ.get("BAYES_SSM_ALLOW_DYNAMIC_FUSED", "0") == "1"
                    )
                ):
                    output, mean, covariance = _run_compiled_fixed_diagonal(*core_args)
                else:
                    output, mean, covariance = _run_compiled_diagonal_transition(*core_args)
            else:
                if diagonal_transition is not None and process_covariances is None:
                    eye = torch.eye(d, device=device, dtype=dtype)
                    dense_transitions = torch.diag_embed(diagonal_transition[..., start:stop, :])
                    dense_process = process_variances[..., start:stop, None, None] * eye
                else:
                    dense_transitions = transitions[..., start:stop, :, :]
                    dense_process = process_covariances[..., start:stop, :, :]
                chunk_args = (
                    keys[..., start:stop, :], values[..., start:stop, :],
                    queries[..., start:stop, :], dense_transitions, dense_process,
                    observation_variances[..., start:stop], mean, covariance,
                    float(denominator_floor),
                )
                if checkpointed:
                    output, mean, covariance = _CheckpointedCoreFunction.apply(*chunk_args)
                else:
                    output, mean, covariance = _run_compiled_core(*chunk_args)
            outputs.append(output)
        return torch.cat(outputs, dim=-2), mean, covariance
    if rotation_transition:
        return _core_loop_rotation(
            keys, values, queries, rotation_radii, rotation_angles,
            process_variances, observation_variances, mean, covariance,
            float(denominator_floor),
        )
    if diagonal_transition is not None:
        transitions = torch.diag_embed(diagonal_transition)
        if process_covariances is None:
            eye = torch.eye(d, device=device, dtype=dtype)
            process_covariances = process_variances[..., None, None] * eye
    return _run_chunked(keys, values, queries, transitions, process_covariances,
                        observation_variances, mean, covariance,
                        chunk_length=chunk_length, denominator_floor=denominator_floor)


class BayesianLayer(nn.Module):
    """Convenience module with positive noise floors and initial covariance.

    Noise values are registered as buffers when ``learn_noise=False`` and as
    trainable parameters when ``learn_noise=True``.  The initial covariance is
    always a buffer: it is part of the layer configuration, rather than a
    trainable scalar.
    """

    def __init__(self, key_dim: int, value_dim: int, *, initial_covariance: float = 3.0,
                process_variance: float = 0.05, observation_variance: float = 0.05,
                 learn_noise: bool = False, denominator_floor: float = 1e-6):
        super().__init__()
        if not isinstance(key_dim, int) or isinstance(key_dim, bool) or not isinstance(value_dim, int) or isinstance(value_dim, bool):
            raise TypeError("key_dim and value_dim must be integers")
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key_dim and value_dim must be positive")
        for value, name, strictly_positive in (
            (initial_covariance, "initial_covariance", False),
            (process_variance, "process_variance", False),
            (observation_variance, "observation_variance", True),
        ):
            if not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real scalar")
            if strictly_positive and value <= 0:
                raise ValueError(f"{name} must be strictly positive")
            if not strictly_positive and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if learn_noise and process_variance <= 0:
            raise ValueError("process_variance must be positive when learn_noise=True")
        if not isinstance(denominator_floor, Real) or not math.isfinite(float(denominator_floor)):
            raise ValueError("denominator_floor must be finite")
        if denominator_floor <= 0:
            raise ValueError("denominator_floor must be positive")
        self.key_dim, self.value_dim = key_dim, value_dim
        self.denominator_floor = denominator_floor
        self.register_buffer("initial_covariance", torch.tensor(float(initial_covariance)))
        self._learn_noise = bool(learn_noise)
        # Keep raw names stable for checkpoint compatibility.  Fixed noise is
        # still a buffer; learned noise uses Parameters with the same names.
        # Store unconstrained logits for the paper's softplus noise
        # parameterization.  Keeping the parameter names stable preserves
        # checkpoint keys while avoiding the exp parameterization's very
        # different gradients for large variances.
        process_raw = torch.tensor(_inv_softplus(float(max(process_variance, 1e-8))))
        observation_raw = torch.tensor(_inv_softplus(float(observation_variance)))
        if learn_noise:
            self._process_raw = nn.Parameter(process_raw)
            self._observation_raw = nn.Parameter(observation_raw)
        else:
            self.register_buffer("_process_raw", process_raw)
            self.register_buffer("_observation_raw", observation_raw)
        self.register_buffer("_process_is_zero", torch.tensor(process_variance == 0))

    @property
    def process_variance(self) -> Tensor:
        return torch.where(self._process_is_zero, self._process_raw * 0,
                           F.softplus(self._process_raw).clamp_min(1e-8))

    @property
    def observation_variance(self) -> Tensor:
        return F.softplus(self._observation_raw).clamp_min(1e-8)

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor, *,
                transitions: Optional[Tensor] = None, chunk_length: Optional[int] = None,
                initial_mean: Optional[Tensor] = None, initial_covariance: Optional[Tensor] = None,
                optimized: bool = False):
        if not all(isinstance(x, Tensor) for x in (keys, values, queries)):
            raise TypeError("keys, values and queries must be torch tensors")
        if keys.ndim < 2 or values.ndim < 2 or queries.ndim < 2:
            raise ValueError("keys, values and queries must have a time and feature dimension")
        if keys.ndim >= 1 and keys.shape[-1] != self.key_dim:
            raise ValueError(f"keys last dimension must be key_dim={self.key_dim}")
        if values.ndim >= 1 and values.shape[-1] != self.value_dim:
            raise ValueError(f"values last dimension must be value_dim={self.value_dim}")
        if queries.ndim >= 1 and queries.shape[-1] != self.key_dim:
            raise ValueError(f"queries last dimension must be key_dim={self.key_dim}")
        if initial_covariance is None:
            d = keys.shape[-1]
            initial_covariance = torch.eye(d, device=keys.device, dtype=torch.float32) * self.initial_covariance.to(
                device=keys.device, dtype=torch.float32
            )
        return bayesian_layer(
            keys, values, queries, transitions=transitions,
            process_variances=self.process_variance,
            observation_variances=self.observation_variance,
            initial_mean=initial_mean, initial_covariance=initial_covariance,
            denominator_floor=self.denominator_floor, chunk_length=chunk_length,
            optimized=optimized,
        )


def make_diagonal_dynamics(log_decay: Tensor) -> Tensor:
    """Build diagonal A_t from log radii, shape ``[..., T, D]`` -> ``[..., T,D,D]``."""
    raw_radius = log_decay.exp()
    # Preserve the paper's forward constraint rho <= 1.  A straight-through
    # clamp would make the gradient twice as large throughout the admissible
    # region (the forward clamp already has unit derivative there), so use the
    # ordinary bounded map.
    # ``where`` keeps a one-sided unit derivative at the boundary rho=1,
    # while values above one receive zero gradient and are projected back to
    # the stable boundary.
    radius = torch.where(raw_radius <= 1.0, raw_radius, torch.ones_like(raw_radius))
    return torch.diag_embed(radius)


def make_rotation_dynamics(log_radius: Tensor, angle: Tensor) -> Tensor:
    """Build block-diagonal damped 2-D rotations.

    ``log_radius`` and ``angle`` have shape ``[..., T, J]``; output has final
    dimensions ``[..., T, 2J, 2J]``. Radii are clamped to one for stability.
    """
    if not isinstance(log_radius, Tensor) or not isinstance(angle, Tensor):
        raise TypeError("log_radius and angle must be tensors")
    if log_radius.shape != angle.shape:
        raise ValueError("log_radius and angle must have the same shape")
    if not log_radius.is_floating_point() or not angle.is_floating_point():
        raise TypeError("log_radius and angle must have floating-point dtype")
    if log_radius.device != angle.device:
        raise ValueError("log_radius and angle must be on the same device")
    raw_radius = log_radius.exp()
    radius = torch.where(raw_radius <= 1.0, raw_radius, torch.ones_like(raw_radius))
    c, s = radius * angle.cos(), radius * angle.sin()
    j = angle.shape[-1]
    out = torch.zeros(*angle.shape[:-1], 2 * j, 2 * j, device=angle.device, dtype=angle.dtype)
    # Advanced indexing writes all 2x2 blocks in one operation.  The previous
    # Python loop performed one device-side slice assignment per state pair;
    # with the paper's 512-dimensional rotational state that was 256 launches
    # every forward call before the actual recurrence started.
    idx = torch.arange(j, device=angle.device)
    even = 2 * idx
    odd = even + 1
    out[..., even, even] = c
    out[..., odd, odd] = c
    out[..., even, odd] = -s
    out[..., odd, even] = s
    return out


def make_rotation_process_covariances(process_variance: Tensor) -> Tensor:
    """Paper rotation noise: ``ell² diag(1, 0)`` in each 2-D block.

    ``process_variance`` has shape ``[..., T, J]`` and the result has shape
    ``[..., T, 2J, 2J]``.  This is separate from the isotropic
    ``process_variances`` convenience argument because the paper's rotational
    construction injects noise only into the first coordinate of each pair.
    """
    if not isinstance(process_variance, Tensor):
        process_variance = torch.as_tensor(process_variance)
    if process_variance.ndim < 1:
        raise ValueError("process_variance must have a final block dimension")
    if not process_variance.is_floating_point():
        raise TypeError("process_variance must have a floating-point dtype")
    if not bool(torch.isfinite(process_variance).all()) or bool((process_variance < 0).any()):
        raise ValueError("process_variance must be finite and nonnegative")
    j = process_variance.shape[-1]
    zeros = torch.zeros_like(process_variance)
    diagonal = torch.stack((process_variance, zeros), dim=-1).reshape(*process_variance.shape[:-1], 2 * j)
    return torch.diag_embed(diagonal)
