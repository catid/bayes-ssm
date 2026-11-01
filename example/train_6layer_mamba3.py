#!/usr/bin/env python3
"""Apples-to-apples six-layer Mamba-3 training benchmark.

This uses the same data, DDP, CUDA-graph, optimizer, batch, sequence length,
embedding, tied FP32 vocabulary head, and reporting path as ``train_6layer.py``.
By default, Mamba-3 SISO mixers replace the four Bayesian positions while
the GQA attention blocks remain in positions 3 and 6.  ``--mixer all-mamba``
keeps the original six-Mamba control used by the earlier measurement.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_6layer import (
    GQAAttention,
    LinearFactory,
    _barrier,
    _distributed_context,
    _make_data,
    _reduce_scalar,
    _sample_batch,
)


def _load_mamba3():
    # ``Path("")`` is ``Path(".")`` and therefore truthy.  Check the raw
    # environment value before constructing it, otherwise a clean checkout
    # silently skips the fallback paths and imports the wrong package.
    configured = os.environ.get("MAMBA3_ROOT")
    candidates = tuple(
        p for p in (
            Path(configured) if configured else None,
            Path("/workspace/fp8/mamba3-gb300"),
            Path(__file__).resolve().parents[2] / "mamba3-gb300",
        ) if p is not None
    )
    root = next((p for p in candidates if (p / "mamba_ssm").is_dir()), None)
    if root is None:
        raise RuntimeError(
            "Mamba-3 source tree not found; set MAMBA3_ROOT to a checkout "
            "containing mamba_ssm/"
        )
    package = types.ModuleType("mamba_ssm")
    package.__path__ = [str(root / "mamba_ssm")]
    package.__package__ = "mamba_ssm"
    sys.modules["mamba_ssm"] = package
    from mamba_ssm.modules.mamba3 import Mamba3
    return Mamba3


class MambaBlock(nn.Module):
    def __init__(
        self,
        width: int,
        mlp_width: int,
        factory: LinearFactory,
        Mamba3,
        *,
        mixer_kind: str = "mamba3",
        heads: int = 64,
        kv_heads: int = 4,
    ):
        super().__init__()
        self.mixer_kind = mixer_kind
        if mixer_kind == "mamba3":
            self.norm = factory.rms_norm(width)
            self.mixer = Mamba3(
                d_model=width,
                d_state=128,
                expand=2,
                headdim=64,
                is_mimo=False,
                mimo_rank=1,
                chunk_size=64,
                is_outproj_norm=False,
                device="cuda",
                dtype=factory.dtype,
            )
        elif mixer_kind == "gqa":
            # Keep the two attention positions identical to the Bayesian
            # six-layer reference (blocks 3 and 6).  This is the actual
            # hybrid comparison; ``--mixer all-mamba`` remains available for
            # the original all-Mamba control.
            self.mixer = GQAAttention(width, heads=heads, kv_heads=kv_heads, factory=factory)
        else:
            raise ValueError(f"unknown mixer kind {mixer_kind!r}")
        self.ffn_norm = factory.rms_norm(width)
        self.ffn = MambaFFN(width, mlp_width, factory)

    def forward(self, x: Tensor) -> Tensor:
        # Both reference mixers own their pre-normalization.  Mamba-3 uses
        # the block norm above; GQA supplies the same norm internally as the
        # Bayesian reference, so do not normalize it twice here.
        mixer_input = self.norm(x) if self.mixer_kind == "mamba3" else x
        x = x + self.mixer(mixer_input)
        z = self.ffn_norm(x)
        return x + self.ffn(z)


class MambaFFN(nn.Module):
    def __init__(self, width: int, mlp_width: int, factory: LinearFactory):
        super().__init__()
        self.gate = factory.linear(width, mlp_width)
        self.up = factory.linear(width, mlp_width)
        self.down = factory.linear(mlp_width, width)

    def forward(self, z: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(z)) * self.up(z))


class MambaSixLayerLM(nn.Module):
    def __init__(self, vocab_size: int, width: int, mlp_width: int,
                 factory: LinearFactory, *, compile_ffn: bool,
                 mixer: str = "hybrid", heads: int = 64, kv_heads: int = 4):
        super().__init__()
        Mamba3 = _load_mamba3()
        if mixer not in {"hybrid", "all-mamba"}:
            raise ValueError("mixer must be 'hybrid' or 'all-mamba'")
        block_mixers = (
            ("mamba3", "mamba3", "gqa", "mamba3", "mamba3", "gqa")
            if mixer == "hybrid" else ("mamba3",) * 6
        )
        self.vocab_size = vocab_size
        self.width = width
        self.mixer_layout = mixer
        self.embedding = nn.Embedding(vocab_size, width, dtype=factory.dtype)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            MambaBlock(
                width, mlp_width, factory, Mamba3, mixer_kind=kind,
                heads=heads, kv_heads=kv_heads,
            )
            for kind in block_mixers
        )
        if compile_ffn:
            for block in self.blocks:
                block.ffn = torch.compile(block.ffn, fullgraph=False, dynamic=False, mode="default")
                if isinstance(block.mixer, GQAAttention):
                    # Match train_6layer.py's ``--compile-components both``
                    # setting for the two attention positions.
                    block.mixer = torch.compile(
                        block.mixer, fullgraph=False, dynamic=False, mode="default"
                    )
        self.final_norm = factory.rms_norm(width)

    def forward(self, tokens: Tensor, factory: LinearFactory) -> Tensor:
        x = self.embedding(tokens)
        with factory.autocast():
            for block in self.blocks:
                x = block(x)
        return self.final_norm(x).float() @ self.embedding.weight.float().t()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run(args: argparse.Namespace) -> dict[str, object]:
    device, rank, world_size, local_rank, ddp_enabled = _distributed_context(args)
    if args.cuda_graph and device.type != "cuda":
        raise ValueError("--cuda-graph requires CUDA")
    if ddp_enabled:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    factory = LinearFactory(args.precision, device, fused_rmsnorm=args.fused_rmsnorm)
    model = MambaSixLayerLM(
        args.vocab_size, args.width, args.mlp_width, factory,
        compile_ffn=args.compile_ffn, mixer=args.mixer,
        heads=args.heads, kv_heads=args.kv_heads,
    ).to(device)
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
            class _CaptureLoggerProxy:
                def __init__(self, logger): self._logger = logger
                def set_runtime_stats_and_log(self): return None
                def __getattr__(self, name): return getattr(self._logger, name)
            model.logger = _CaptureLoggerProxy(model.logger)
    train_tokens, valid_tokens = _make_data(args, seed=args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=device.type == "cuda",
        capturable=args.cuda_graph,
    )
    generator = torch.Generator().manual_seed(args.seed + 1 + rank)
    gpu_batches: Tensor | None = None
    if args.cuda_graph and args.gpu_batch_staging and args.tokens is None:
        gpu_generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
        gpu_batches = torch.randint(
            args.vocab_size, (args.steps, args.batch, args.seq + 1),
            generator=gpu_generator, device=device, dtype=torch.long,
        )
    model.train()

    def train_step(inputs: Tensor, targets: Tensor) -> Tensor:
        optimizer.zero_grad(set_to_none=False)
        logits = model(inputs, factory)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        return loss

    static_inputs = torch.empty((args.batch, args.seq), device=device, dtype=torch.long) if args.cuda_graph else None
    static_targets = torch.empty_like(static_inputs) if static_inputs is not None else None
    for _ in range(args.warmup):
        batch = _sample_batch(train_tokens, batch=args.batch, seq=args.seq, generator=generator)
        if args.cuda_graph:
            static_inputs.copy_(batch[:, :-1]); static_targets.copy_(batch[:, 1:])
            train_step(static_inputs, static_targets)
        else:
            batch = batch.to(device); train_step(batch[:, :-1], batch[:, 1:])
    if ddp_enabled and args.cuda_graph and device.type == "cuda":
        model.reducer._rebuild_buckets()
        torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    _barrier(ddp_enabled)
    started = time.perf_counter(); losses: list[float] = []
    if args.cuda_graph:
        set_capture_stream = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
        if set_capture_stream is not None: set_capture_stream(True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_loss = train_step(static_inputs, static_targets)
        for i in range(args.steps):
            if gpu_batches is None:
                batch = _sample_batch(train_tokens, batch=args.batch, seq=args.seq, generator=generator)
                static_inputs.copy_(batch[:, :-1]); static_targets.copy_(batch[:, 1:])
            else:
                batch = gpu_batches[i]
                static_inputs.copy_(batch[:, :-1]); static_targets.copy_(batch[:, 1:])
            graph.replay()
        graph_loss_tensor = graph_loss
    else:
        for _ in range(args.steps):
            batch = _sample_batch(train_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
            losses.append(float(train_step(batch[:, :-1], batch[:, 1:]).detach()))
        graph_loss_tensor = None
    if device.type == "cuda": torch.cuda.synchronize(device)
    if graph_loss_tensor is not None: losses.append(float(graph_loss_tensor.detach()))
    local_elapsed = time.perf_counter() - started
    elapsed = _reduce_scalar(local_elapsed, device=device, op=dist.ReduceOp.MAX) if ddp_enabled else local_elapsed
    model.eval()
    with torch.no_grad():
        batch = _sample_batch(valid_tokens, batch=args.batch, seq=args.seq, generator=generator).to(device)
        validation_loss = F.cross_entropy(model(batch[:, :-1], factory).reshape(-1, args.vocab_size), batch[:, 1:].reshape(-1))
    validation_value = float(validation_loss)
    if ddp_enabled: validation_value = _reduce_scalar(validation_value, device=device, op=dist.ReduceOp.SUM) / world_size
    final_train_loss = losses[-1] if losses else None
    if ddp_enabled and final_train_loss is not None:
        final_train_loss = _reduce_scalar(final_train_loss, device=device, op=dist.ReduceOp.SUM) / world_size
    result = {
        "schema": (
            "bayes-ssm-six-layer-mamba3-hybrid-v2"
            if args.mixer == "hybrid"
            else "bayes-ssm-six-layer-mamba3-all-mamba-v1"
        ),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "model": {
            "layers": (
                ["mamba3-siso", "mamba3-siso", "gqa", "mamba3-siso", "mamba3-siso", "gqa"]
                if args.mixer == "hybrid" else ["mamba3-siso"] * 6
            ),
            "mixer_layout": args.mixer,
            "parameters": parameter_count(model),
            "width": args.width,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "mlp_width": args.mlp_width,
            "vocab_size": args.vocab_size,
            "sequence_length": args.seq,
            "batch_per_rank": args.batch,
        },
        "world_size": world_size, "rank": rank, "precision": "BF16 projections + TransformerEngine fused RMSNorm with FP32 tied vocabulary head" if args.fused_rmsnorm else "BF16",
        "optimizer": "capturable fused AdamW", "cuda_graph": bool(args.cuda_graph), "gpu_batch_staging": bool(gpu_batches is not None), "compile_ffn": bool(args.compile_ffn), "compile_gqa": bool(args.compile_ffn and args.mixer == "hybrid"), "chunk_size": 64,
        "timed_steps": args.steps, "warmup_steps": args.warmup, "dataset": "synthetic random token stream (seed 123)",
        "tokens_per_second_global": args.steps * args.batch * args.seq * world_size / elapsed if elapsed else float("inf"),
        "elapsed_seconds": elapsed, "peak_allocated_mib_per_rank": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib_per_rank": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "final_train_loss": final_train_loss, "validation_loss": validation_value, "validation_perplexity": math.exp(validation_value) if validation_value < 20 else float("inf"),
        "measurement": "full forward, backward, and optimizer step",
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0"); p.add_argument("--precision", choices=("fp32", "bf16", "te-fp8"), default="bf16")
    p.add_argument("--fused-rmsnorm", action="store_true"); p.add_argument("--compile-ffn", action="store_true")
    p.add_argument("--vocab-size", type=int, default=32000); p.add_argument("--width", type=int, default=512)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--mlp-width", type=int, default=8256,
                    help="FFN width (8256 nearly matches the 100.6M Bayesian hybrid; use 8000 for the all-Mamba control)")
    p.add_argument("--mixer", choices=("hybrid", "all-mamba"), default="hybrid",
                    help="Mamba3 in blocks 1/2/4/5 plus GQA in 3/6, or the original all-Mamba control")
    p.add_argument("--seq", type=int, default=512); p.add_argument("--batch", type=int, default=1); p.add_argument("--steps", type=int, default=20); p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.1); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--chunk-length", type=int, default=64)
    p.add_argument("--cuda-graph", action="store_true"); p.add_argument("--gpu-batch-staging", action="store_true"); p.add_argument("--ddp-bucket-cap-mb", type=float, default=12.0)
    p.add_argument("--tokens", type=Path); p.add_argument("--valid-tokens", type=Path); p.add_argument("--seed", type=int, default=123); p.add_argument("--output", type=Path)
    args = p.parse_args(); result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2), flush=True)
        if args.output is not None: args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__": main()
