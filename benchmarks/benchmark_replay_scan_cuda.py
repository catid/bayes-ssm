#!/usr/bin/env python3
"""Benchmark the exact static Bayesian scan against dense ReplaySSM.

The associative scan is valid only for identity dynamics and zero process noise.
It uses information form (prefix cumsum of precision/natural parameters), so it
is a dense covariance algorithm; no diagonal approximation is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayes_ssm.bayesian_layer import bayesian_layer
from bayes_ssm.replay_ssm import (
    ReplaySSM,
    associative_static_bayesian,
    associative_static_bayesian_cuda,
)


def timed(fn, warmup: int, steps: int) -> tuple[float, tuple[torch.Tensor, ...]]:
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / steps, out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is required; use --device cpu for a correctness-only run")
    device = torch.device(args.device)
    torch.manual_seed(0)
    keys = torch.randn(args.batch, args.sequence, args.key_dim, device=device)
    values = torch.randn(args.batch, args.sequence, args.value_dim, device=device)
    queries = torch.randn_like(keys)

    exact = lambda: bayesian_layer(
        keys, values, queries, process_variances=0.0,
        observation_variances=1.0, chunk_length=64, optimized=False,
    )
    scan = lambda: (
        associative_static_bayesian_cuda(keys, values, queries)
        if device.type == "cuda" else associative_static_bayesian(keys, values, queries)
    )
    exact_s, exact_out = timed(exact, args.warmup, args.steps)
    scan_s, scan_out = timed(scan, args.warmup, args.steps)
    max_abs = max((a - b).abs().max().item() for a, b in zip(exact_out, scan_out))
    ref_flat = torch.cat([x.float().reshape(-1) for x in exact_out])
    scan_flat = torch.cat([x.float().reshape(-1) for x in scan_out])
    relative_l2 = (scan_flat - ref_flat).norm().item() / ref_flat.norm().clamp_min(1e-12).item()
    cosine = torch.nn.functional.cosine_similarity(ref_flat, scan_flat, dim=0).item()

    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch": args.batch, "sequence": args.sequence,
        "key_dim": args.key_dim, "value_dim": args.value_dim,
        "dense_replay_reference_ms": exact_s * 1e3,
        "associative_scan_ms": scan_s * 1e3,
        "speedup": exact_s / scan_s,
        "max_abs_difference": max_abs,
        "relative_l2_difference": relative_l2,
        "cosine_similarity": cosine,
        "process_variance": 0.0,
        "dynamics": "identity",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
