# bayes-ssm

Production dense Bayesian sequence layers for large-model training. This repository contains the Bayesian Layer equations, the ReplaySSM checkpoint/replay boundary, and the validated associative CUDA scan for the static identity-transition case.

## Research

The implementation follows **[Memory by Design: Bayesian Layers for Sequence Modeling](https://arxiv.org/abs/2605.31163)** (Dowling, Jeon, Savin, and Park). The hand-cleaned paper conversion is [docs/MemoryByDesign_2605.31163.md](docs/MemoryByDesign_2605.31163.md).

The package keeps recurrent state, covariance, reductions, and scan solves in FP32. It accepts BF16/FP8 projection outputs at the state boundary. `ReplaySSM` provides exact dense covariance with Mamba-style chunk replay for backward.

## Associative CUDA scan

`associative_static_bayesian_cuda` uses CUDA prefix accumulation in information form followed by batched Cholesky solves. It is mathematically valid for `A=I` and `Q=0`; the API rejects observation variances below the denominator floor instead of silently changing the recurrence.

Final GB300 measurements are in [benchmarks/associative_scan_gb300.json](benchmarks/associative_scan_gb300.json). The scan benchmark is [benchmarks/benchmark_replay_scan_cuda.py](benchmarks/benchmark_replay_scan_cuda.py).

| Shape | Dense reference | Associative CUDA scan | Speedup | Max abs. error | Cosine |
|---|---:|---:|---:|---:|---:|
| B1 × T16 × D4 × M4 | 21.71 ms | 1.04 ms | 20.9× | 7.2e-7 | 1.0000001 |
| B4 × T64 × D16 × M16 | 218.28 ms | 1.00 ms | 218.8× | 6.7e-6 | 1.0 |

These are CUDA forward timings, not end-to-end optimizer-step measurements. The current prototype uses optimized PyTorch CUDA operations rather than a handwritten `.cu` extension.

## Six-layer training example

The [`example/`](example/) directory contains a runnable causal language-model recipe with the production layer arrangement used in the training comparison: Bayesian mixers in blocks 1, 2, 4, and 5, causal GQA attention in blocks 3 and 6, SwiGLU feed-forwards, tied embeddings, and the CroissAttn FP32 recurrent-state boundary. The default dimensions are approximately 100M parameters (640-wide, 10 query heads, 2 KV heads, 6,144-wide MLP, 32k vocabulary). It supports synthetic data for a smoke run and `.npy` or flat token files for a real corpus.

```bash
python example/train_6layer.py --device cuda:0 --precision te-fp8 \
  --optimize-bayes --cuda-graph --tokens train.npy --valid-tokens valid.npy \
  --seq 512 --batch 128 --steps 100 --warmup 5 --output run.json
```

The recorded GB300 training result below includes forward, backward, and fused AdamW updates. It uses fixed-shape CUDA graph capture for a 100-step synthetic training run; the validation loss is a health metric rather than a language-quality measurement. The complete Bayesian record is [`example/results/gb300_6layer_final.json`](example/results/gb300_6layer_final.json), with the matched Mamba-3 baseline in [`example/results/gb300_6layer_mamba3_baseline.json`](example/results/gb300_6layer_mamba3_baseline.json).

| Model | Parameters | Precision | Batch × sequence | Throughput | Peak VRAM | Validation CE / PPL* | Step scope |
|---|---:|---|---:|---:|---:|---:|---|
| Bayesian/GQA six-layer | 101.43M | TE FP8 projections + FP32 state | 128 × 512 per GPU | **175,875 tok/s** | 182,364 MiB | 10.501 / 36,353 | forward + backward + fused AdamW |
| Matched Mamba-3 SISO six-layer | 107.38M | BF16 | 128 × 512 per GPU | **495,529 tok/s** | 64,921 MiB | — | forward + backward + fused AdamW |

The Bayesian mixer uses the exact fixed-diagonal-decay dense-covariance recurrence. Its compiled diagonal-transition specialization removes diagonal-matrix materialization while retaining the full covariance matrix; CUDA graph capture removes repeated launch and allocator overhead for the fixed shape. The associative scan numbers above are a separate forward-only identity-transition, zero-process-noise kernel benchmark.

\* The validation stream in this recorded run is synthetic and random; CE/PPL are included as health metrics, not accuracy claims.

## Install

```bash
pip install -e .
```

## Run the benchmark

```bash
python benchmarks/benchmark_replay_scan_cuda.py \
  --batch 4 --sequence 64 --key-dim 16 --value-dim 16 \
  --device cuda:0
```
