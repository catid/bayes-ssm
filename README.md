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

The [`example/`](example/) directory contains a runnable causal language-model recipe with the production layer arrangement used in the training comparison: Bayesian mixers in blocks 1, 2, 4, and 5, causal GQA attention in blocks 3 and 6, SwiGLU feed-forwards, tied embeddings, and the CroissAttn FP32 recurrent-state boundary. The fixed benchmark dimensions are approximately 100M parameters (512-wide, 64 query heads, 4 KV heads, 8,448-wide MLP, 32k vocabulary). The default paper-feature path adds learned noise/transition and short-convolution projections, bringing this shape to about 101.94M parameters. It supports synthetic data for a smoke run and NumPy or headerless uint16/int64 token files for a real corpus.

```bash
python example/train_6layer.py --device cuda:0 --precision bf16 \
  --fused-rmsnorm --compile-components both --optimize-bayes \
  --bayes-backend inductor --bayes-features paper --bayes-dynamics learned \
  --cuda-graph --gpu-batch-staging --width 512 --heads 64 --kv-heads 4 \
  --mlp-width 8448 --seq 512 --batch 256 --steps 60 --warmup 20 \
  --output bayes_paper.json
```

The parameter-matched Mamba-3/GQA run uses the same six-layer layout, optimizer,
and graph/DDP settings. Its MLP width is 8,403 so that both models have the
same 101.94M parameter count (within 0.002%):

```bash
python example/train_6layer_mamba3.py --device cuda:0 --precision bf16 \
  --fused-rmsnorm --compile-ffn --mixer hybrid --width 512 --heads 64 \
  --kv-heads 4 --mlp-width 8403 --seq 512 --batch 256 --steps 60 --warmup 20 \
  --cuda-graph --gpu-batch-staging --output mamba3_paper_match.json
```

The final paper-feature comparison below was measured on two GB300 GPUs with
one NCCL DDP rank per GPU. Each timed step includes the complete forward pass,
backward pass, gradient clipping, and fused AdamW update. Both models use BF16
projections, TE fused RMSNorm, a 32k tied vocabulary head, GQA in blocks 3 and
6, and the same 256 × 512 tokens per GPU. Bayesian blocks 1, 2, 4, and 5 use
the exact dense-covariance recurrence with the paper's causal Q/K/V short
convolutions, input-dependent process/observation noise, and input-dependent
diagonal transitions. The first run pays one-time compilation cost; it is not
included in the timed 60 steps. The raw records are [Bayesian](example/results/gb300_6layer_bayes_paper.json)
and [Mamba-3](example/results/gb300_6layer_mamba3_paper_match.json).

| Model | Parameters | Mixer layout | Batch × sequence / GPU | Throughput (global / per GPU) | Peak VRAM allocated / reserved / rank | Validation loss / PPL |
|---|---:|---|---:|---:|---:|---:|
| Bayesian paper path | **101.942M** | Bayes, Bayes, GQA, Bayes, Bayes, GQA | 256 × 512 | **437,376 / 218,688 tok/s** | **149,228 / 191,220 MiB** | 10.3806 / 32,228 |
| Mamba-3 matched hybrid | **101.944M** | Mamba-3, Mamba-3, GQA, Mamba-3, Mamba-3, GQA | 256 × 512 | **998,046 / 499,023 tok/s** | **119,268 / 165,628 MiB** | 10.3488 / 31,219 |

The recurrent state and covariance remain FP32. The paper path is about 2.28×
slower and uses about 25% more allocated VRAM than this Mamba-3 baseline at
this shape; that cost is the exact dynamic dense-covariance computation, not a
fixed-dynamics approximation.

Use `--bayes-backend fused` when the lower-memory custom CUDA replay path is preferred; `inductor` with the fixed scalar dynamics is the fastest measured backend for the recorded shape.

The validation stream is synthetic random data, so its loss is a health check rather than a language-quality claim.

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
