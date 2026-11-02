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
  --fused-rmsnorm --compile-components both --optimize-bayes --bayes-backend inductor --cuda-graph --tokens train.npy --valid-tokens valid.npy \
  --bayes-features fixed --seq 512 --batch 256 --steps 100 --warmup 5 --output run.json
```

The matched Mamba-3/GQA run uses the same fixed shape and optimizer settings:

```bash
python example/train_6layer_mamba3.py --device cuda:0 --precision bf16 \
  --fused-rmsnorm --compile-ffn --mixer hybrid --width 512 --heads 64 \
  --kv-heads 4 --mlp-width 8258 --seq 512 --batch 256 --steps 60 --warmup 20 \
  --cuda-graph --gpu-batch-staging --output mamba3_hybrid.json
```

The recorded GB300 result below is the fixed fast-path benchmark. It includes forward, backward, and fused AdamW updates, fixed-shape CUDA graph capture, independently compiled GQA/SwiGLU components, the exact dense-covariance recurrence with scalar dynamics, and two-rank NCCL DDP. Use `--bayes-features paper --bayes-dynamics learned` for the paper-feature path: causal depthwise Q/K/V convolution, input-dependent process/observation noise, and input-dependent diagonal transitions. That path remains exact dense covariance but uses the general diagonal-transition Inductor kernel and has additional parameters. Synthetic benchmark input staging is enabled only for the synthetic run; corpus-backed runs use the normal token loader. The first run pays the one-time component compilation cost. The final Bayesian record is [example/results/gb300_6layer_final.json](example/results/gb300_6layer_final.json). The corrected hybrid Mamba-3 record is [example/results/gb300_6layer_mamba3_hybrid.json](example/results/gb300_6layer_mamba3_hybrid.json); the earlier six-Mamba measurement is retained as an explicitly labeled control in [example/results/gb300_6layer_mamba3_apples.json](example/results/gb300_6layer_mamba3_apples.json). Both Mamba runs use [example/train_6layer_mamba3.py](example/train_6layer_mamba3.py). Every row below is a complete forward, backward, gradient-clipping, and fused-AdamW step on two GB300 ranks, with batch and sequence shown per GPU.

| Model | Parameters | Mixer layout | Precision and state | Batch × sequence / GPU | Throughput (global / per GPU) | Peak VRAM allocated / reserved / rank | Validation loss / PPL |
|---|---:|---|---|---:|---:|---:|---:|
| Bayesian/GQA six-layer | **100.604M** | Bayes, Bayes, GQA, Bayes, Bayes, GQA | BF16 projections + TE fused RMSNorm + FP32 Bayesian state and tied vocabulary head | 256 × 512 | **985,761 / 492,880 tok/s** | **122,581 / 163,268 MiB** | 10.3793 / 32,187 |
| Mamba-3/GQA hybrid | **100.608M** | Mamba-3, Mamba-3, GQA, Mamba-3, Mamba-3, GQA | BF16 projections + TE fused RMSNorm + FP32 tied vocabulary head | 256 × 512 | **1,059,506 / 529,753 tok/s** | **118,589 / 164,714 MiB** | 10.3473 / 31,172 |
| Mamba-3 all-Mamba control | **100.614M** | Six Mamba-3 SISO mixers | BF16 projections + TE fused RMSNorm + FP32 tied vocabulary head | 256 × 512 | **1,356,200 / 678,100 tok/s** | **122,724 / 170,970 MiB** | 10.3578 / 31,484 |

The recurrent state and covariance remain FP32. The selectable fused CUDA backend launches on the active PyTorch stream, so CUDA graph replay executes its Bayesian forward and reverse kernels on every step; it retains exact dense-covariance math while storing only chunk boundaries and recomputation intermediates.

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
