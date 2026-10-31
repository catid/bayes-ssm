#!/usr/bin/env python3
"""Train the six-layer Bayesian/attention language model.

The model used in the repository's comparison has six blocks.  Blocks 1, 2,
4, and 5 use the exact dense Bayesian recurrence from :mod:`bayes_ssm`, while
blocks 3 and 6 use causal grouped-query (GQA) attention.  Each block has a
SwiGLU feed-forward network and the output projection is tied to the token
embedding.

The defaults are the approximately 100M-parameter configuration used in the
GB300 measurements (width 640, 10 query heads, 2 KV heads, and MLP width
6144).  A small CPU run is useful as a smoke test::

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
import time
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from bayes_ssm.bayesian_layer import bayesian_layer


class LinearFactory:
    """Create ordinary linears or TransformerEngine FP8 linears."""

    def __init__(self, precision: str, device: torch.device) -> None:
        if precision not in {"fp32", "bf16", "te-fp8"}:
            raise ValueError(f"unknown precision {precision!r}")
        self.requested = precision
        self.device = device
        self.te = None
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

    def linear(self, in_features: int, out_features: int, *, bias: bool = False) -> nn.Module:
        if self.te is not None:
            return self.te.Linear(
                in_features, out_features, bias=bias, params_dtype=torch.bfloat16
            )
        return nn.Linear(in_features, out_features, bias=bias, dtype=self.dtype)

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
            raise ValueError("width must be divisible by heads; heads must divide kv_heads")
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.norm = nn.RMSNorm(width, dtype=factory.dtype)
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
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.optimized = optimized
        self.chunk_length = chunk_length
        self.norm = nn.RMSNorm(width, dtype=factory.dtype)
        self.q_proj = factory.linear(width, width)
        self.k_proj = factory.linear(width, width)
        self.v_proj = factory.linear(width, width)
        self.out_proj = factory.linear(width, width)
        self.read_norm = nn.RMSNorm(self.head_dim, dtype=factory.dtype)
        # A fixed, stable diagonal transition is the default production path;
        # covariance itself remains dense in bayesian_layer.
        self.register_buffer("decay", torch.full((1, 1, self.head_dim), 0.97), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        z = self.norm(x)
        # Normalize each head before entering the FP32 state boundary.  This
        # prevents a single projection outlier from dominating the covariance.
        q = F.normalize(F.silu(self.q_proj(z).float()), dim=-1)
        k = F.normalize(F.silu(self.k_proj(z).float()), dim=-1)
        v = F.silu(self.v_proj(z).float())
        q = q.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        shape = (b * self.heads, t, self.head_dim)
        q, k, v = (value.reshape(shape) for value in (q, k, v))
        decay = self.decay.expand(b * self.heads, t, self.head_dim)
        read, _, _ = bayesian_layer(
            k, v, q,
            transition_diagonals=decay,
            process_variances=0.1,
            observation_variances=1.0,
            chunk_length=self.chunk_length,
            optimized=self.optimized,
        )
        read = read.view(b, self.heads, t, self.head_dim).permute(0, 2, 1, 3)
        # RMSNorm's parameter dtype follows the projection path. Cast at the
        # state boundary so BF16 runs do not invoke a mixed-dtype fallback;
        # covariance and readout reductions above were FP32.
        read = self.read_norm(read.to(x.dtype))
        return self.out_proj(read.reshape(b, t, self.width).to(x.dtype))


class SwiGLU(nn.Module):
    def __init__(self, width: int, mlp_width: int, factory: LinearFactory):
        super().__init__()
        self.norm = nn.RMSNorm(width, dtype=factory.dtype)
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
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "bayesian":
            self.mixer = BayesianMixer(
                width, heads, factory, optimized=optimized, chunk_length=chunk_length
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
        width: int = 640,
        heads: int = 10,
        kv_heads: int = 2,
        mlp_width: int = 6144,
        *,
        factory: LinearFactory,
        optimized: bool = False,
        chunk_length: int = 64,
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
            )
            for kind in self.block_kinds
        )
        self.final_norm = nn.RMSNorm(width, dtype=factory.dtype)

    def forward(self, tokens: Tensor, factory: LinearFactory) -> Tensor:
        x = self.embedding(tokens)
        with factory.autocast():
            for block in self.blocks:
                x = block(x)
        # The CroissAttn precision boundary keeps the tied head in FP32.
        return self.final_norm(x).float() @ self.embedding.weight.float().t()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _read_tokens(path: Path) -> Tensor:
    try:
        import numpy as np
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        values = values.reshape(-1)
        return torch.from_numpy(values.astype("int64", copy=True))
    except (ImportError, ValueError, OSError):
        # Also accept a flat uint16/int64 binary token file.
        return torch.from_file(str(path), dtype=torch.int64)


def _make_data(args: argparse.Namespace, *, seed: int) -> tuple[Tensor, Tensor]:
    if args.tokens is not None:
        train = _read_tokens(args.tokens)
        valid = _read_tokens(args.valid_tokens) if args.valid_tokens else train
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


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    factory = LinearFactory(args.precision, device)
    optimized = bool(args.optimize_bayes and device.type == "cuda")
    model = SixLayerLM(
        args.vocab_size, args.width, args.heads, args.kv_heads, args.mlp_width,
        factory=factory, optimized=optimized, chunk_length=args.chunk_length,
    ).to(device)
    train_tokens, valid_tokens = _make_data(args, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay,
                                  fused=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    for _ in range(args.warmup):
        batch = _sample_batch(train_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
        loss = F.cross_entropy(model(batch[:, :-1], factory).reshape(-1, args.vocab_size), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(args.steps):
        batch = _sample_batch(train_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
        logits = model(batch[:, :-1], factory)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        batch = _sample_batch(valid_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
        validation_loss = F.cross_entropy(
            model(batch[:, :-1], factory).reshape(-1, args.vocab_size), batch[:, 1:].reshape(-1)
        )
    result: dict[str, object] = {
        "layers": list(SixLayerLM.block_kinds),
        "parameters": parameter_count(model),
        "target_parameters": 100_000_000,
        "width": args.width,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "mlp_width": args.mlp_width,
        "vocab_size": args.vocab_size,
        "sequence_length": args.seq,
        "batch": args.batch,
        "steps": args.steps,
        "precision_requested": args.precision,
        "precision_active": factory.active,
        "bayesian_core": "exact dense covariance; compiled" if optimized else "exact dense covariance; eager",
        "chunk_length": args.chunk_length,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "elapsed_seconds": elapsed,
        "tokens_per_second": args.steps * args.batch * args.seq / elapsed if elapsed else float("inf"),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "final_train_loss": losses[-1] if losses else None,
        "validation_loss": float(validation_loss),
        "validation_ppl": math.exp(float(validation_loss)) if float(validation_loss) < 20 else float("inf"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16", "te-fp8"), default="bf16")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--mlp-width", type=int, default=6144)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--optimize-bayes", action="store_true",
                        help="use the compiled exact Bayesian core on CUDA")
    parser.add_argument("--tokens", type=Path,
                        help="optional .npy or flat int64 token file; otherwise synthetic data is used")
    parser.add_argument("--valid-tokens", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path,
                        help="optional JSON output path")
    args = parser.parse_args()
    if args.seq < 2 or args.batch <= 0 or args.steps < 0 or args.warmup < 0:
        parser.error("seq must be >=2; batch, steps, and warmup must be nonnegative/positive")
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
