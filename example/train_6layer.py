#!/usr/bin/env python3
"""Train the six-layer Bayesian/attention language model.

The model used in the repository's comparison has six blocks.  Blocks 1, 2,
4, and 5 use the exact dense Bayesian recurrence from :mod:`bayes_ssm`, while
blocks 3 and 6 use causal grouped-query (GQA) attention.  Each block has a
SwiGLU feed-forward network and the output projection is tied to the token
embedding.

The defaults are the approximately 100M-parameter configuration used in the
GB300 measurements (width 512, 64 query heads, 4 KV heads, and MLP width
8448).  A small CPU run is useful as a smoke test::

    python example/train_6layer.py --device cpu --width 32 --heads 4 \
        --kv-heads 2 --mlp-width 64 --vocab-size 256 --seq 32 --batch 2 \
        --steps 1 --warmup 0

For a CUDA run, ``--optimize-bayes`` enables the compiled exact dense core.
The recurrent mean, covariance, innovation denominator, and reductions stay
in FP32 even when projections are BF16 or TransformerEngine FP8.  FP8 is
explicit: requesting ``--precision te-fp8`` fails if a compatible TE build is
not installed, instead of silently reporting a BF16 run as FP8.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

# Make ``python example/train_6layer.py`` work directly from a source checkout
# without requiring an editable install first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayes_ssm.bayesian_layer import bayesian_layer


class LinearFactory:
    """Create projection and normalization modules for one precision policy."""

    def __init__(self, precision: str, device: torch.device, *, fused_rmsnorm: bool = False) -> None:
        if precision not in {"fp32", "bf16", "te-fp8"}:
            raise ValueError(f"unknown precision {precision!r}")
        self.requested = precision
        self.device = device
        self.fused_rmsnorm = bool(fused_rmsnorm)
        self.te = None
        self._rms_te = None
        self._recipe_types = None
        if precision == "te-fp8":
            try:
                import transformer_engine.pytorch as te
                from transformer_engine.common.recipe import DelayedScaling, Format
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "--precision te-fp8 requires a CUDA-compatible "
                    "TransformerEngine installation"
                ) from exc
            if device.type != "cuda":
                raise RuntimeError("TransformerEngine FP8 requires a CUDA device")
            self.te = te
            self._recipe_types = (DelayedScaling, Format)
        self.active = "te-fp8" if self.te is not None else precision
        self.dtype = torch.bfloat16 if self.active == "te-fp8" else (
            torch.bfloat16 if precision == "bf16" else torch.float32
        )
        if self.fused_rmsnorm:
            if device.type != "cuda":
                raise RuntimeError("--fused-rmsnorm requires a CUDA device")
            try:
                import transformer_engine.pytorch as te
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "--fused-rmsnorm requires a CUDA-compatible TransformerEngine installation"
                ) from exc
            self._rms_te = te

    def linear(self, in_features: int, out_features: int, *, bias: bool = False) -> nn.Module:
        if self.te is not None:
            return self.te.Linear(
                in_features, out_features, bias=bias, params_dtype=torch.bfloat16
            )
        return nn.Linear(in_features, out_features, bias=bias, dtype=self.dtype)

    def rms_norm(self, hidden_size: int) -> nn.Module:
        if self._rms_te is not None:
            return self._rms_te.RMSNorm(hidden_size, dtype=self.dtype)
        return nn.RMSNorm(hidden_size, dtype=self.dtype)

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if self.te is None:
            yield
            return
        delayed_scaling, fmt = self._recipe_types
        recipe = delayed_scaling(fp8_format=fmt.HYBRID, amax_history_len=1, interval=1)
        autocast = getattr(self.te, "autocast", None)
        if autocast is not None:
            with autocast(enabled=True, recipe=recipe):
                yield
        else:  # TE <= 2.0 compatibility
            with self.te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                yield


class GQAAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, factory: LinearFactory):
        super().__init__()
        if width % heads or heads % kv_heads:
            raise ValueError("width must be divisible by heads; kv_heads must divide heads")
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.norm = factory.rms_norm(width)
        self.q_proj = factory.linear(width, width)
        self.k_proj = factory.linear(width, kv_heads * self.head_dim)
        self.v_proj = factory.linear(width, kv_heads * self.head_dim)
        self.out_proj = factory.linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        b, t, _ = z.shape
        q = self.q_proj(z).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        except (TypeError, RuntimeError):  # PyTorch < 2.5 fallback
            repeat = self.heads // self.kv_heads
            y = F.scaled_dot_product_attention(
                q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
                is_causal=True,
            )
        y = y.transpose(1, 2).reshape(b, t, self.width)
        return self.out_proj(y)


class BayesianMixer(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        factory: LinearFactory,
        *,
        optimized: bool,
        chunk_length: int,
        bayes_features: str = "paper",
        bayes_dynamics: str = "learned",
        bayes_key_dim: int | None = None,
        fuse_paper_qkv: bool = True,
        fuse_paper_projection: bool = True,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.head_dim = width, heads, width // heads
        if bayes_key_dim is not None and (
            not isinstance(bayes_key_dim, int) or isinstance(bayes_key_dim, bool)
        ):
            raise ValueError("bayes_key_dim must be an integer or None")
        self.key_dim = self.head_dim if bayes_key_dim is None else bayes_key_dim
        if self.key_dim <= 0 or self.key_dim > self.head_dim:
            raise ValueError("bayes_key_dim must be in [1, width // heads]")
        self.state_width = self.heads * self.key_dim
        self.optimized = optimized
        self.chunk_length = chunk_length
        if bayes_features not in {"fixed", "paper"}:
            raise ValueError("bayes_features must be 'fixed' or 'paper'")
        if bayes_dynamics not in {"learned", "identity", "rotation"}:
            raise ValueError("bayes_dynamics must be 'learned', 'identity', or 'rotation'")
        if bayes_dynamics == "rotation" and bayes_features != "paper":
            raise ValueError("rotation dynamics require bayes_features='paper'")
        if bayes_dynamics == "rotation" and (self.key_dim % 2):
            raise ValueError("rotation dynamics require an even per-head dimension")
        self.bayes_features = bayes_features
        self.bayes_dynamics = bayes_dynamics
        # The rotation parameterization has different Q/K and V widths
        # (keys/queries occupy the first coordinate of each pair), so retain
        # separate projections for that mode.
        self.fuse_paper_qkv = bool(fuse_paper_qkv and bayes_dynamics != "rotation")
        # The auxiliary projection can only be concatenated when Q/K/V are
        # already concatenated; keep this effective flag false for the
        # legacy three-projection reference path.  TransformerEngine FP8
        # linears use one amax scale for the whole output tensor.  Decay and
        # noise have very different ranges from Q/K/V, so keep them in their
        # separate projections under FP8 to preserve the paper's stability
        # boundary.  BF16/FP32 use the fused projection.
        te_active = getattr(factory, "te", None) is not None
        self.fuse_paper_projection = bool(
            fuse_paper_projection and self.fuse_paper_qkv and not te_active
        )
        self.norm = factory.rms_norm(width)
        if bayes_features == "paper":
            # Memory by Design's language/MQAR path uses three independent
            # depthwise causal length-4 convolutions after Q/K/V projection.
            # We warm-start each as the identity map so enabling this path
            # does not destroy the stable fixed-path initialization, then let
            # training learn the short context filter.
            if self.fuse_paper_qkv:
                # Concatenating independent Q/K/V projections and using one
                # depthwise grouped convolution preserves the paper equations
                # while replacing six small launches with two larger ones.
                # The Q/K/V, noise, and (learned) transition projections all
                # consume the same normalized input.  A single row-concatenated
                # projection removes two GEMM launches from every Bayesian
                # block while retaining the exact paper equations.  Keep the
                # legacy three-module path available for state-dict parity and
                # CPU/reference checks.
                projection_width = 2 * self.state_width + width
                if self.fuse_paper_projection and bayes_dynamics == "learned":
                    projection_width += self.state_width + 2 * heads
                self.qkv_proj = factory.linear(
                    width, projection_width,
                    # Legacy Q/K/V projections are bias-free.  Keep the
                    # auxiliary rows bias-only via ``aux_bias`` below so the
                    # fused module has exactly the same parameter count.
                    bias=False,
                )
                self.aux_bias = None
                if self.fuse_paper_projection and bayes_dynamics == "learned":
                    self.aux_bias = nn.Parameter(
                        torch.zeros(self.state_width + 2 * heads, dtype=factory.dtype,
                                    device=self.qkv_proj.weight.device)
                    )
                self.qkv_conv = nn.Conv1d(
                    2 * self.state_width + width, 2 * self.state_width + width, kernel_size=4,
                    groups=2 * self.state_width + width, bias=False, dtype=factory.dtype,
                )
                nn.init.zeros_(self.qkv_conv.weight)
                with torch.no_grad():
                    self.qkv_conv.weight[:, 0, -1] = 1
            else:
                qk_width = self.state_width // 2 if bayes_dynamics == "rotation" else self.state_width
                self.q_conv = nn.Conv1d(
                    qk_width, qk_width, kernel_size=4, groups=qk_width, bias=False,
                    dtype=factory.dtype,
                )
                self.k_conv = nn.Conv1d(
                    qk_width, qk_width, kernel_size=4, groups=qk_width, bias=False,
                    dtype=factory.dtype,
                )
                self.v_conv = nn.Conv1d(
                    width, width, kernel_size=4, groups=width, bias=False,
                    dtype=factory.dtype,
                )
                for conv in (self.q_conv, self.k_conv, self.v_conv):
                    nn.init.zeros_(conv.weight)
                    with torch.no_grad():
                        conv.weight[:, 0, -1] = 1
            # The paper parameterizes positive input-dependent process and
            # observation variances with softplus. Per-head scalars preserve
            # the exact dense covariance while avoiding a large extra head-D
            # projection. Zero weights plus calibrated biases start at the
            # fixed path's Q=.1 and R=1 values.
            if bayes_dynamics == "rotation":
                # One radius and one angle per 2-D state pair.  The radius
                # uses the same bounded exp(-softplus(.)) parameterization as
                # diagonal dynamics; zero angle initializes to a diagonal
                # positive transition while retaining learnable rotations.
                self.rotation_proj = factory.linear(width, self.state_width, bias=True)
                nn.init.zeros_(self.rotation_proj.weight)
                nn.init.zeros_(self.rotation_proj.bias)
                half_lives = torch.logspace(
                    math.log10(32.0), math.log10(4096.0), self.state_width // 2,
                    device=self.norm.weight.device, dtype=torch.float32,
                )
                raw = torch.log(torch.expm1(math.log(2.0) / half_lives))
                rotation_bias = self.rotation_proj.bias.data.view(
                    self.heads, 2, self.key_dim // 2
                )
                rotation_bias[:, 0, :].copy_(raw.to(rotation_bias.dtype).view(
                    self.heads, self.key_dim // 2
                ))
                self.decay_proj = None
                self.noise_proj = factory.linear(width, 2 * heads, bias=True)
                nn.init.zeros_(self.noise_proj.weight)
                nn.init.zeros_(self.noise_proj.bias)
                self.noise_proj.bias.data[0::2].fill_(math.log(math.expm1(0.1)))
                self.noise_proj.bias.data[1::2].fill_(math.log(math.expm1(1.0)))
            elif bayes_dynamics == "learned":
                # A learned diagonal transition is input-dependent. Initialize
                # per-coordinate half-lives over the paper's 32..4096-token
                # range, then let the input projection modulate them.
                half_lives = torch.logspace(
                    math.log10(32.0), math.log10(4096.0), self.state_width,
                    # Use the mixer device for both the fused and legacy
                    # projection layouts.  The unfused reference path does
                    # not create ``qkv_proj`` and must still be constructible
                    # when ``--no-paper-fused-qkv`` is used.
                    device=self.norm.weight.device, dtype=torch.float32,
                )
                # rho=exp(-softplus(raw)); inverse-softplus(-log(rho))
                # initializes the transition exactly at each half-life.
                raw = torch.log(torch.expm1(math.log(2.0) / half_lives))
                if self.fuse_paper_projection and bayes_dynamics == "learned":
                    # qkv_proj is the fused [Q,K,V,decay,noise] projection.
                    # Its bias starts at zero for Q/K/V, then at the exact
                    # inverse-softplus initial values used by the legacy
                    # modules.  The order of the trailing noise rows is the
                    # interleaved [process_h, observation_h] layout.
                    aux_bias = self.aux_bias
                    assert aux_bias is not None
                    # Standalone noise/decay projections are zero initialized;
                    # preserve that stable paper initialization in the fused
                    # rows while retaining their learned gradients.
                    self.qkv_proj.weight.data[2 * self.state_width + width :].zero_()
                    aux_bias.data[:self.state_width].copy_(raw.to(aux_bias.dtype))
                    aux_bias.data[self.state_width : self.state_width + 2 * heads : 2].fill_(math.log(math.expm1(0.1)))
                    aux_bias.data[self.state_width + 1 : self.state_width + 2 * heads : 2].fill_(math.log(math.expm1(1.0)))
                    self.decay_proj = None
                    self.noise_proj = None
                else:
                    self.decay_proj = factory.linear(width, self.state_width, bias=True)
                    nn.init.zeros_(self.decay_proj.weight)
                    self.decay_proj.bias.data.copy_(raw.to(self.decay_proj.bias.dtype))
                    self.noise_proj = factory.linear(width, 2 * heads, bias=True)
                    nn.init.zeros_(self.noise_proj.weight)
                    nn.init.zeros_(self.noise_proj.bias)
                    self.noise_proj.bias.data[0::2].fill_(math.log(math.expm1(0.1)))
                    self.noise_proj.bias.data[1::2].fill_(math.log(math.expm1(1.0)))
            else:
                self.rotation_proj = None
                self.decay_proj = None
                self.noise_proj = factory.linear(width, 2 * heads, bias=True)
                nn.init.zeros_(self.noise_proj.weight)
                nn.init.zeros_(self.noise_proj.bias)
                self.noise_proj.bias.data[0::2].fill_(math.log(math.expm1(0.1)))
                self.noise_proj.bias.data[1::2].fill_(math.log(math.expm1(1.0)))
        else:
            self.rotation_proj = None
        if bayes_features != "paper" or not self.fuse_paper_qkv:
            qk_width = self.state_width // 2 if bayes_dynamics == "rotation" else self.state_width
            self.q_proj = factory.linear(width, qk_width)
            self.k_proj = factory.linear(width, qk_width)
            self.v_proj = factory.linear(width, width)
        self.out_proj = factory.linear(width, width)
        self.read_norm = factory.rms_norm(self.head_dim)
        # CroissAttn's gated readout keeps the recurrent contribution bounded
        # before it is added to the residual stream.  The gate is projected
        # from the same normalized input as Q/K/V and is applied per head.
        self.g_proj = factory.linear(width, width)
        # A fixed, stable diagonal transition is the default production path;
        # covariance itself remains dense in bayesian_layer.
        self.register_buffer("decay", torch.full((1, 1, self.key_dim), 0.97), persistent=False)
        # Keep scalar noise on the module device.  Besides avoiding a tiny
        # host-to-device conversion every token block, this makes the complete
        # fixed-shape training step capturable by torch.cuda.CUDAGraph.
        self.register_buffer("process_variance", torch.tensor(0.1), persistent=False)
        self.register_buffer("observation_variance", torch.tensor(1.0), persistent=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        """Load checkpoints written by either projection layout.

        Fusing rows changes only the storage layout, not the learned
        parameters.  Translate the old ``decay_proj``/``noise_proj`` keys to
        the fused weight and ``aux_bias`` before delegating to ``nn.Module``;
        the reverse translation keeps an unfused reference module usable for
        checkpoint inspection and numerical tests.
        """
        qkv_key = prefix + "qkv_proj.weight"
        qkv_conv_key = prefix + "qkv_conv.weight"
        aux_key = prefix + "aux_bias"
        decay_w_key, decay_b_key = prefix + "decay_proj.weight", prefix + "decay_proj.bias"
        noise_w_key, noise_b_key = prefix + "noise_proj.weight", prefix + "noise_proj.bias"
        q_keys = tuple(prefix + f"{name}.weight" for name in ("q_proj", "k_proj", "v_proj"))
        conv_keys = tuple(prefix + f"{name}.weight" for name in ("q_conv", "k_conv", "v_conv"))
        width = self.width
        state_width = self.state_width
        has_fused_qkv = hasattr(self, "qkv_proj")
        has_legacy_qkv = hasattr(self, "q_proj")
        if has_fused_qkv:
            # Legacy -> fused Q/K/V projection and depthwise convolution.
            if qkv_key not in state_dict and all(key in state_dict for key in q_keys):
                state_dict[qkv_key] = torch.cat(tuple(state_dict.pop(key) for key in q_keys), dim=0)
            if qkv_conv_key not in state_dict and all(key in state_dict for key in conv_keys):
                state_dict[qkv_conv_key] = torch.cat(tuple(state_dict.pop(key) for key in conv_keys), dim=0)
            qkv = state_dict.get(qkv_key)
            expected = self.qkv_proj.weight.shape
            aux_keys = (decay_w_key, decay_b_key, noise_w_key, noise_b_key)
            if qkv is not None and tuple(qkv.shape) != tuple(expected) and all(key in state_dict for key in aux_keys):
                # Legacy -> fused auxiliary projections.  The row order is
                # [Q, K, V, decay, process/observation noise].
                state_dict[qkv_key] = torch.cat(
                    (qkv, state_dict.pop(decay_w_key), state_dict.pop(noise_w_key)), dim=0
                )
                state_dict[aux_key] = torch.cat(
                    (state_dict.pop(decay_b_key), state_dict.pop(noise_b_key)), dim=0
                )
        elif has_legacy_qkv:
            # Fused -> legacy layout for reference modules and checkpoint
            # inspection.  Split the same rows used in the forward path.
            qkv = state_dict.pop(qkv_key, None)
            qkv_conv = state_dict.pop(qkv_conv_key, None)
            if qkv is not None:
                qkv_core = qkv[: 2 * state_width + width]
                q_rows, k_rows, v_rows = qkv_core.split((state_width, state_width, width), dim=0)
                for name, rows in zip(("q_proj", "k_proj", "v_proj"), (q_rows, k_rows, v_rows), strict=True):
                    state_dict[prefix + name + ".weight"] = rows
            if qkv_conv is not None:
                for name, rows in zip(("q_conv", "k_conv", "v_conv"), qkv_conv.split((state_width, state_width, width), dim=0), strict=True):
                    state_dict[prefix + name + ".weight"] = rows
            aux = state_dict.pop(aux_key, None)
            if aux is not None:
                state_dict[decay_w_key] = qkv[2 * state_width + width : 3 * state_width + width]
                state_dict[noise_w_key] = qkv[3 * state_width + width :]
                state_dict[decay_b_key] = aux[:state_width]
                state_dict[noise_b_key] = aux[state_width:]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        z = self.norm(x)
        if self.bayes_features == "paper":
            # The paper projects first and applies an independent depthwise
            # causal short convolution to each projected stream.
            if self.fuse_paper_qkv:
                projected = self.qkv_proj(z)
                if self.fuse_paper_projection and self.bayes_dynamics == "learned":
                    qkv_core_width = 2 * self.state_width + self.width
                    qkv_projected = projected[..., :qkv_core_width]
                    aux_projected = projected[..., qkv_core_width:] + self.aux_bias
                else:
                    qkv_projected = projected
                convolved = self.qkv_conv(
                    F.pad(qkv_projected.transpose(1, 2), (3, 0))
                ).transpose(1, 2)
                q_input, k_input, v_input = convolved.split(
                    (self.state_width, self.state_width, self.width), dim=-1
                )
            else:
                projected = (self.q_proj(z), self.k_proj(z), self.v_proj(z))
                q_input, k_input, v_input = tuple(
                    conv(F.pad(value.transpose(1, 2), (3, 0))).transpose(1, 2)
                    for value, conv in zip(
                        projected, (self.q_conv, self.k_conv, self.v_conv), strict=True
                    )
                )
        else:
            q_input = k_input = v_input = z
        # Normalize each head before entering the FP32 state boundary.  The
        # normalization must happen after splitting the projection into heads:
        # normalizing the flattened width would shrink every head's key norm
        # by roughly sqrt(num_heads), weakening the Kalman update by that
        # factor while leaving observation variance unchanged.
        q_projected = q_input if self.bayes_features == "paper" else self.q_proj(q_input)
        k_projected = k_input if self.bayes_features == "paper" else self.k_proj(k_input)
        v_projected = v_input if self.bayes_features == "paper" else self.v_proj(v_input)
        q_feature_dim = self.key_dim // 2 if self.bayes_dynamics == "rotation" else self.key_dim
        q = F.silu(q_projected.float()).view(b, t, self.heads, q_feature_dim)
        k = F.silu(k_projected.float()).view(b, t, self.heads, q_feature_dim)
        v = F.silu(v_projected.float()).view(b, t, self.heads, self.head_dim)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if self.bayes_dynamics == "rotation":
            # Paper embedding: each scalar key/query occupies the first
            # coordinate of a 2-D rotational state pair.
            q = torch.stack((q, torch.zeros_like(q)), dim=-1).reshape(
                b, t, self.heads, self.key_dim
            ).permute(0, 2, 1, 3)
            k = torch.stack((k, torch.zeros_like(k)), dim=-1).reshape(
                b, t, self.heads, self.key_dim
            ).permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        shape = (b * self.heads, t, self.key_dim)
        q, k = (value.reshape(shape) for value in (q, k))
        v = v.reshape(b * self.heads, t, self.head_dim)
        if self.bayes_features == "paper":
            if self.fuse_paper_projection and self.bayes_dynamics == "learned":
                aux = aux_projected
                noise = aux[..., self.state_width :].float().view(b, t, self.heads, 2)
            else:
                noise = self.noise_proj(z).float().view(b, t, self.heads, 2)
            process = F.softplus(noise[..., 0]) + 1e-4
            observation = F.softplus(noise[..., 1]) + 1e-4
            process = process.permute(0, 2, 1).reshape(b * self.heads, t)
            observation = observation.permute(0, 2, 1).reshape(b * self.heads, t)
            if self.bayes_dynamics == "rotation":
                process = process[..., None].expand(
                    b * self.heads, t, self.key_dim // 2
                )
                rotation = self.rotation_proj(z).float().view(b, t, self.heads, 2, self.key_dim // 2)
                rotation_decay = torch.exp(-F.softplus(rotation[..., 0, :]))
                rotation_angle = rotation[..., 1, :]
                rotation_decay = rotation_decay.reshape(b * self.heads, t, self.key_dim // 2)
                rotation_angle = rotation_angle.reshape(b * self.heads, t, self.key_dim // 2)
                decay = None
            elif self.decay_proj is None and not (
                self.fuse_paper_projection and self.bayes_dynamics == "learned"
            ):
                decay = torch.ones(
                    b * self.heads, t, self.key_dim,
                    device=z.device, dtype=torch.float32,
                )
            else:
                # Paper parameterization: rho=exp(-softplus(raw)), which is
                # stable in (0, 1] and gives usable gradients for long memory.
                if self.fuse_paper_projection:
                    decay_logits = aux[..., : self.state_width].float()
                else:
                    decay_logits = self.decay_proj(z).float()
                # Keep the paper parameterization exactly as written.  The
                # algebraically equivalent sigmoid(-x) has different FP32
                # rounding and can amplify small errors through covariance
                # updates, so use exp(-softplus(x)) for parity with the
                # unfused/reference path.
                decay = torch.exp(-F.softplus(decay_logits))
                decay = decay.view(b, t, self.heads, self.key_dim)
                decay = decay.permute(0, 2, 1, 3).reshape(
                    b * self.heads, t, self.key_dim
                )
        else:
            decay = self.decay.expand(b * self.heads, t, self.key_dim)
            process = self.process_variance
            observation = self.observation_variance
        layer_kwargs = dict(
            process_variances=process,
            observation_variances=observation,
            chunk_length=self.chunk_length,
            optimized=self.optimized,
            fixed_dynamics=self.bayes_features == "fixed",
        )
        if self.bayes_dynamics == "rotation":
            layer_kwargs.update(rotation_radii=rotation_decay, rotation_angles=rotation_angle)
        else:
            layer_kwargs.update(transition_diagonals=decay)
        read, _, _ = bayesian_layer(k, v, q, **layer_kwargs)
        read = read.view(b, self.heads, t, self.head_dim).permute(0, 2, 1, 3)
        # RMSNorm's parameter dtype follows the projection path. Cast at the
        # state boundary so BF16 runs do not invoke a mixed-dtype fallback;
        # covariance and readout reductions above were FP32.
        read = self.read_norm(read.to(x.dtype))
        gate = self.g_proj(z).view(b, t, self.heads, self.head_dim)
        read = (read * F.silu(gate)).reshape(b, t, self.width)
        return self.out_proj(read.to(x.dtype))


class SwiGLU(nn.Module):
    def __init__(self, width: int, mlp_width: int, factory: LinearFactory):
        super().__init__()
        self.norm = factory.rms_norm(width)
        self.gate = factory.linear(width, mlp_width)
        self.up = factory.linear(width, mlp_width)
        self.down = factory.linear(mlp_width, width)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        return self.down(F.silu(self.gate(z)) * self.up(z))


class HybridBlock(nn.Module):
    def __init__(
        self,
        kind: str,
        width: int,
        heads: int,
        kv_heads: int,
        mlp_width: int,
        factory: LinearFactory,
        *,
        optimized: bool,
        chunk_length: int,
        bayes_features: str = "paper",
        bayes_dynamics: str = "learned",
        bayes_key_dim: int | None = None,
        fuse_paper_qkv: bool = True,
        fuse_paper_projection: bool = True,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "bayesian":
            self.mixer = BayesianMixer(
                width, heads, factory, optimized=optimized, chunk_length=chunk_length,
                bayes_features=bayes_features, bayes_dynamics=bayes_dynamics,
                bayes_key_dim=bayes_key_dim,
                fuse_paper_qkv=fuse_paper_qkv,
                fuse_paper_projection=fuse_paper_projection,
            )
        elif kind == "fa":
            self.mixer = GQAAttention(width, heads, kv_heads, factory)
        else:
            raise ValueError(f"unknown block kind {kind!r}")
        self.ffn = SwiGLU(width, mlp_width, factory)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(x)
        return x + self.ffn(x)


class SixLayerLM(nn.Module):
    """Six blocks: Bayesian, Bayesian, FA-GQA, Bayesian, Bayesian, FA-GQA."""

    block_kinds = ("bayesian", "bayesian", "fa", "bayesian", "bayesian", "fa")

    def __init__(
        self,
        vocab_size: int = 32_000,
        width: int = 512,
        heads: int = 64,
        kv_heads: int = 4,
        mlp_width: int = 8448,
        *,
        factory: LinearFactory,
        optimized: bool = False,
        chunk_length: int = 64,
        bayes_features: str = "paper",
        bayes_dynamics: str = "learned",
        bayes_key_dim: int | None = None,
        fuse_paper_qkv: bool = True,
        fuse_paper_projection: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.embedding = nn.Embedding(vocab_size, width, dtype=factory.dtype)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            HybridBlock(
                kind, width, heads, kv_heads, mlp_width, factory,
                optimized=optimized, chunk_length=chunk_length,
                bayes_features=bayes_features, bayes_dynamics=bayes_dynamics,
                bayes_key_dim=bayes_key_dim,
                fuse_paper_qkv=fuse_paper_qkv,
                fuse_paper_projection=fuse_paper_projection,
            )
            for kind in self.block_kinds
        )
        self.final_norm = factory.rms_norm(width)

    def forward(self, tokens: Tensor, factory: LinearFactory) -> Tensor:
        x = self.embedding(tokens)
        with factory.autocast():
            for block in self.blocks:
                x = block(x)
        # The CroissAttn precision boundary keeps the tied head in FP32.
        return self.final_norm(x).float() @ self.embedding.weight.float().t()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _read_tokens(path: Path, *, token_dtype: str = "auto") -> Tensor:
    """Read NumPy token arrays or headerless uint16/int64 token streams.

    The paper repository's ``.npy``-named shards are headerless uint16 files,
    while ordinary NumPy arrays carry the ``.npy`` magic header.  Detect that
    distinction instead of interpreting a uint16 shard as int64 and silently
    reducing its apparent length by 4x.
    """
    try:
        with path.open("rb") as stream:
            has_numpy_header = stream.read(6) == b"\x93NUMPY"
    except OSError:
        has_numpy_header = False
    if has_numpy_header:
        try:
            import numpy as np
            values = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
            return torch.from_numpy(values.astype("int64", copy=True))
        except (ImportError, ValueError, OSError) as exc:
            raise ValueError(f"unable to read NumPy token array {path}") from exc
    if token_dtype == "auto":
        token_dtype = "uint16"
    raw_dtype = {"uint16": torch.uint16, "int64": torch.int64}.get(token_dtype)
    if raw_dtype is None:
        raise ValueError(f"unsupported token dtype {token_dtype!r}")
    element_size = torch.empty((), dtype=raw_dtype).element_size()
    size = path.stat().st_size // element_size
    return torch.from_file(str(path), dtype=raw_dtype, size=size).to(torch.int64)


def _make_data(args: argparse.Namespace, *, seed: int) -> tuple[Tensor, Tensor]:
    if args.tokens is not None:
        train = _read_tokens(args.tokens, token_dtype=args.token_dtype)
        valid = (_read_tokens(args.valid_tokens, token_dtype=args.token_dtype)
                 if args.valid_tokens else train)
        if train.numel() < args.seq + 1 or valid.numel() < args.seq + 1:
            raise ValueError("token files must contain at least seq+1 tokens")
        return train, valid
    generator = torch.Generator().manual_seed(seed)
    count = max(args.batch * (args.steps + args.warmup + 4) * (args.seq + 1), args.seq + 1)
    train = torch.randint(args.vocab_size, (count,), generator=generator)
    valid = torch.randint(args.vocab_size, (max(args.batch * (args.seq + 1), args.seq + 1),), generator=generator)
    return train, valid


def _sample_batch(tokens: Tensor, *, batch: int, seq: int, generator: torch.Generator) -> Tensor:
    starts = torch.randint(tokens.numel() - seq, (batch,), generator=generator)
    offsets = torch.arange(seq + 1)
    return tokens[starts[:, None] + offsets[None, :]]


def _distributed_context(args: argparse.Namespace) -> tuple[torch.device, int, int, int, bool]:
    """Initialize torchrun state and return (device, rank, world, local, enabled).

    A regular single-process invocation keeps the device selected by
    ``--device``.  Under torchrun, each process owns its local CUDA device and
    NCCL/Gloo is initialized exactly once here so model construction and all
    later collectives use the same process group.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid torchrun environment: rank={rank}, world_size={world_size}"
        )
    enabled = world_size > 1
    requested = torch.device(args.device)
    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1")
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun requested CUDA but CUDA is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise ValueError(
                    f"LOCAL_RANK={local_rank} exceeds CUDA device count "
                    f"({torch.cuda.device_count()})"
                )
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            backend = "nccl"
        elif requested.type == "cpu":
            device = torch.device("cpu")
            backend = "gloo"
        else:
            raise ValueError(f"DDP supports CUDA or CPU, got {requested}")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        device = requested
        if device.type == "cuda":
            # ``torch.device("cuda")`` has no explicit index and is accepted
            # by tensor factories, but ``set_device`` requires one.  Preserve
            # the caller's current device for single-process runs and make it
            # explicit for the rest of the training path.
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                torch.cuda.set_device(device)
    return device, rank, world_size, local_rank, enabled


def _barrier(enabled: bool) -> None:
    if enabled:
        dist.barrier()


def _reduce_scalar(value: float, *, device: torch.device, op: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def run(args: argparse.Namespace) -> dict[str, object]:
    device, rank, world_size, local_rank, ddp_enabled = _distributed_context(args)
    if args.bayes_backend == "inductor":
        os.environ["BAYES_SSM_DISABLE_FUSED_CUDA"] = "1"
    elif args.bayes_backend == "fused":
        os.environ["BAYES_SSM_DISABLE_FUSED_CUDA"] = "0"
    if args.cuda_graph and device.type != "cuda":
        raise ValueError("--cuda-graph requires a CUDA device")
    # Every rank must initialize the model identically before DDP's parameter
    # broadcast.  Preserve the regular single-process invocation's ambient
    # RNG behavior; its data generator is already explicitly seeded below.
    if ddp_enabled:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    factory = LinearFactory(args.precision, device, fused_rmsnorm=args.fused_rmsnorm)
    optimized = bool(args.optimize_bayes and device.type == "cuda")
    model = SixLayerLM(
        args.vocab_size, args.width, args.heads, args.kv_heads, args.mlp_width,
        factory=factory, optimized=optimized, chunk_length=args.chunk_length,
        bayes_features=args.bayes_features, bayes_dynamics=args.bayes_dynamics,
        bayes_key_dim=args.bayes_key_dim,
        fuse_paper_qkv=args.fuse_paper_qkv,
        fuse_paper_projection=args.fuse_paper_projection,
    ).to(device)
    # Compile the dense attention and SwiGLU components independently.  This
    # preserves the custom Bayesian recurrence boundary while fusing the
    # projection/activation backward graphs.  It is optional because the first
    # run pays the one-time Dynamo/Inductor compilation cost.
    if args.compile_components != "none":
        for block in model.blocks:
            if args.compile_components in ("gqa", "both") and isinstance(block.mixer, GQAAttention):
                block.mixer = torch.compile(
                    block.mixer, fullgraph=False, dynamic=False, mode="default"
                )
            if args.compile_components in ("swiglu", "both"):
                block.ffn = torch.compile(
                    block.ffn, fullgraph=False, dynamic=False, mode="default"
                )
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            find_unused_parameters=False,
            static_graph=bool(args.cuda_graph),
        )
        if args.cuda_graph:
            # DDP's C++ logger records a CUDA event in every pre-forward call;
            # event synchronization is illegal during graph capture.  Keep a
            # delegating proxy so reducer internals remain intact while the
            # capture-only runtime-stat call becomes a no-op.
            class _CaptureLoggerProxy:
                def __init__(self, logger):
                    self._logger = logger

                def set_runtime_stats_and_log(self):
                    return None

                def __getattr__(self, name):
                    return getattr(self._logger, name)

            model.logger = _CaptureLoggerProxy(model.logger)
    train_tokens, valid_tokens = _make_data(args, seed=args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=device.type == "cuda",
        capturable=args.cuda_graph,
    )
    # Distinct streams avoid every rank doing identical synthetic windows while
    # preserving reproducibility for a fixed rank/world-size configuration.
    generator = torch.Generator().manual_seed(args.seed + 1 + rank)
    # Synthetic benchmark runs do not need a host-side token sampler.  Keeping
    # the fixed-shape input pool on the GPU removes the CPU indexing and H2D
    # copy from the CUDA-graph replay loop.  Real token streams continue to use
    # the ordinary host sampler, since their corpus must remain authoritative.
    gpu_batches: Tensor | None = None
    if args.cuda_graph and args.gpu_batch_staging and args.tokens is None:
        gpu_generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
        gpu_batches = torch.randint(
            args.vocab_size,
            (args.steps, args.batch, args.seq + 1),
            generator=gpu_generator,
            device=device,
            dtype=torch.long,
        )
    model.train()

    def train_step(inputs: Tensor, targets: Tensor) -> Tensor:
        # CUDA graph capture requires stable gradient storage.  ``set_to_none``
        # is therefore false for both warmup and capture.
        optimizer.zero_grad(set_to_none=False)
        logits = model(inputs, factory)
        loss = F.cross_entropy(
            logits.reshape(-1, args.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        return loss

    static_inputs: Tensor | None = None
    static_targets: Tensor | None = None
    if args.cuda_graph:
        static_inputs = torch.empty(
            (args.batch, args.seq), device=device, dtype=torch.long
        )
        static_targets = torch.empty_like(static_inputs)

    for _ in range(args.warmup):
        batch = _sample_batch(
            train_tokens, batch=args.batch, seq=args.seq, generator=generator
        )
        if args.cuda_graph:
            assert static_inputs is not None and static_targets is not None
            static_inputs.copy_(batch[:, :-1])
            static_targets.copy_(batch[:, 1:])
            loss = train_step(static_inputs, static_targets)
        else:
            batch = batch.to(device)
            loss = train_step(batch[:, :-1], batch[:, 1:])
        del loss
    if ddp_enabled and args.cuda_graph and device.type == "cuda":
        # The first DDP forward may rebuild gradient buckets.  Do that while
        # the stream is not capturing so the capture sees fixed CUDA buckets;
        # otherwise the reducer attempts a CPU-to-CUDA copy inside capture.
        model.reducer._rebuild_buckets()
        torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    _barrier(ddp_enabled)
    started = time.perf_counter()
    losses: list[float] = []
    graph_loss_tensor: Tensor | None = None
    if args.cuda_graph:
        assert static_inputs is not None and static_targets is not None
        # The override keeps autograd nodes created during warmup on the
        # capture stream.  Without it, PyTorch rejects AccumulateGrad nodes
        # that still reference the default stream.
        set_capture_stream = getattr(
            torch.autograd.graph, "set_override_stale_capture_stream", None
        )
        if set_capture_stream is not None:
            set_capture_stream(True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                graph_loss = train_step(static_inputs, static_targets)
        except Exception as exc:
            raise RuntimeError(
                "CUDA graph capture failed; use the default launch path or "
                "a fixed-shape capturable optimizer"
            ) from exc
        for step_index in range(args.steps):
            if gpu_batches is None:
                batch = _sample_batch(
                    train_tokens, batch=args.batch, seq=args.seq, generator=generator
                )
                static_inputs.copy_(batch[:, :-1])
                static_targets.copy_(batch[:, 1:])
            else:
                batch = gpu_batches[step_index]
                static_inputs.copy_(batch[:, :-1])
                static_targets.copy_(batch[:, 1:])
            graph.replay()
        # Replays only enqueue work.  Synchronizing and converting the scalar
        # loss to a Python float inside this loop would force a device/host
        # round trip for every step and largely defeat CUDA graph launch
        # amortization.  The final device synchronization below makes the
        # latest captured loss value safe to read once for reporting.
        if args.steps:
            graph_loss_tensor = graph_loss
    else:
        for _ in range(args.steps):
            batch = _sample_batch(
                train_tokens, batch=args.batch, seq=args.seq, generator=generator
            ).to(device)
            loss = train_step(batch[:, :-1], batch[:, 1:])
            losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if graph_loss_tensor is not None:
        losses.append(float(graph_loss_tensor.detach()))
    local_elapsed = time.perf_counter() - started
    _barrier(ddp_enabled)
    elapsed = (
        _reduce_scalar(local_elapsed, device=device, op=dist.ReduceOp.MAX)
        if ddp_enabled else local_elapsed
    )
    model.eval()
    with torch.no_grad():
        batch = _sample_batch(valid_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
        validation_loss = F.cross_entropy(
            model(batch[:, :-1], factory).reshape(-1, args.vocab_size), batch[:, 1:].reshape(-1)
        )
    validation_value = float(validation_loss)
    if ddp_enabled:
        validation_value = _reduce_scalar(
            validation_value, device=device, op=dist.ReduceOp.SUM
        ) / world_size
    final_train_loss = losses[-1] if losses else None
    if ddp_enabled and final_train_loss is not None:
        final_train_loss = _reduce_scalar(
            final_train_loss, device=device, op=dist.ReduceOp.SUM
        ) / world_size
    effective_bayes_key_dim = args.bayes_key_dim or (args.width // args.heads)
    value_dim = args.width // args.heads
    fused_shape_supported = (effective_bayes_key_dim, value_dim) in {
        (4, 8), (8, 8), (64, 64)
    }
    result: dict[str, object] = {
        "layers": list(SixLayerLM.block_kinds),
        "parameters": parameter_count(model),
        "target_parameters": 100_000_000,
        "width": args.width,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "mlp_width": args.mlp_width,
        "bayes_key_dim": args.bayes_key_dim,
        "bayes_key_dim_effective": effective_bayes_key_dim,
        "bayes_features": args.bayes_features,
        "fuse_paper_qkv": bool(args.fuse_paper_qkv),
        "fuse_paper_projection": bool(args.fuse_paper_projection),
        "bayes_dynamics": (
            args.bayes_dynamics if args.bayes_features == "paper" else "fixed-constant"
        ),
        "vocab_size": args.vocab_size,
        "sequence_length": args.seq,
        "batch": args.batch,
        "steps": args.steps,
        "precision_requested": args.precision,
        "precision_active": factory.active,
        "fused_rmsnorm": bool(args.fused_rmsnorm),
        "compile_components": args.compile_components,
        "bayesian_core": (
            "exact dense covariance; fused CUDA chunk replay"
            if optimized and args.bayes_backend == "fused" and fused_shape_supported
            else (
                "exact dense covariance; Inductor diagonal-transition"
                if optimized and (args.bayes_backend == "inductor" or not fused_shape_supported) and args.bayes_features == "paper"
                else (
                    "exact dense covariance; Inductor fallback"
                    if optimized else "exact dense covariance; eager"
                )
            )
        ),
        "bayes_backend_requested": args.bayes_backend,
        "chunk_length": args.chunk_length,
        "cuda_graph": bool(args.cuda_graph),
        "gpu_batch_staging": bool(gpu_batches is not None),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "ddp": ddp_enabled,
        "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb if ddp_enabled else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "elapsed_seconds": elapsed,
        "local_elapsed_seconds": local_elapsed,
        "tokens_per_second": (
            args.steps * args.batch * args.seq * world_size / elapsed
            if elapsed else float("inf")
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "final_train_loss": final_train_loss,
        "validation_loss": validation_value,
        "validation_ppl": math.exp(validation_value) if validation_value < 20 else float("inf"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16", "te-fp8"), default="bf16")
    parser.add_argument(
        "--fused-rmsnorm", action="store_true",
        help="use TransformerEngine's fused RMSNorm (CUDA + TransformerEngine required)",
    )
    parser.add_argument(
        "--compile-components", choices=("none", "gqa", "swiglu", "both"), default="none",
        help="compile selected GQA/SwiGLU modules independently",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--mlp-width", type=int, default=8448)
    parser.add_argument("--bayes-key-dim", type=int, default=None,
                        help="per-head Bayesian key/state dimension (default: head width; e.g. 4 with value width 8)")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument(
        "--bayes-features", choices=("fixed", "paper"), default="paper",
        help="Bayesian mixer features (default: paper; fixed is the old fast benchmark path)",
    )
    parser.add_argument(
        "--bayes-dynamics", choices=("learned", "identity", "rotation"), default="learned",
        help="paper dynamics: learned diagonal decay, identity, or damped 2-D rotation",
    )
    parser.add_argument(
        "--paper-fused-qkv", dest="fuse_paper_qkv",
        action=argparse.BooleanOptionalAction, default=True,
        help="fuse paper-mode Q/K/V projections and depthwise convolutions",
    )
    parser.add_argument(
        "--paper-fused-projection", dest="fuse_paper_projection",
        action=argparse.BooleanOptionalAction, default=True,
        help="fuse paper-mode Q/K/V, noise, and learned-transition projections",
    )
    parser.add_argument("--optimize-bayes", action="store_true",
                        help="use the compiled exact Bayesian core on CUDA")
    parser.add_argument("--bayes-backend", choices=("auto", "fused", "inductor"), default="auto",
                        help="select the exact CUDA replay kernel or Inductor fallback")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="capture a fixed-shape forward/backward/AdamW step")
    parser.add_argument("--gpu-batch-staging", action="store_true",
                        help="pre-stage synthetic CUDA-graph batches on GPU (benchmark mode)")
    parser.add_argument("--ddp-bucket-cap-mb", type=float, default=25.0,
                        help="DDP gradient bucket size in MiB (torchrun only)")
    parser.add_argument("--tokens", type=Path,
                        help="optional NumPy or headerless token file; otherwise synthetic data is used")
    parser.add_argument("--valid-tokens", type=Path)
    parser.add_argument("--token-dtype", choices=("auto", "uint16", "int64"), default="auto",
                        help="dtype for headerless token files (auto defaults to uint16)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path,
                        help="optional JSON output path")
    args = parser.parse_args()
    if args.seq < 2 or args.batch <= 0 or args.steps < 0 or args.warmup < 0:
        parser.error("seq must be >=2; batch, steps, and warmup must be nonnegative/positive")
    result = run(args)
    # Only rank zero emits a result file/record.  Other ranks still execute
    # validation and reductions, then exit quietly after the process group is
    # torn down so torchrun cannot observe interleaved JSON output.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(text, end="")
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
