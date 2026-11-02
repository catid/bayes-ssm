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
  --paper-fused-qkv --paper-fused-projection --output bayes_paper.json
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
included in the timed 60 steps. The raw records are [Bayesian](example/results/gb300_6layer_bayes_paper_fusedproj_v2.json)
and [Mamba-3](example/results/gb300_6layer_mamba3_paper_match_v2.json). The Mamba-3
validation pass uses 64 sequences per GPU after timing to keep the post-run
evaluation within available memory; this does not change the timed batch.

| Model | Parameters | Mixer layout | Batch × sequence / GPU | Throughput (global / per GPU) | Peak VRAM allocated / reserved / rank | Validation loss / PPL |
|---|---:|---|---:|---:|---:|---:|
| Bayesian paper path | **101.942M** | Bayes, Bayes, GQA, Bayes, Bayes, GQA | 256 × 512 | **434,006 / 217,003 tok/s** | **149,225 / 191,800 MiB** | 10.3787 / 32,168 |
| Mamba-3 matched hybrid | **101.944M** | Mamba-3, Mamba-3, GQA, Mamba-3, Mamba-3, GQA | 256 × 512 | **1,218,503 / 609,251 tok/s** | **119,268 / 133,256 MiB** | 10.3623 / 31,645 |

The recurrent state and covariance remain FP32. The paper path is about 2.81×
slower and uses about 25% more allocated VRAM than this Mamba-3 baseline at
this shape; that cost is the exact dynamic dense-covariance computation, not a
fixed-dynamics approximation. The Mamba validation used a 64-example per-GPU
validation batch to avoid post-run memory pressure; validation is not part of
the timed training measurement.

### Mixed-width exact CUDA path

The full paper equations do not require the key/state width to equal the value
width. `--bayes-key-dim 4` keeps 8-wide values and dense 4×4 covariance per
head, while retaining causal Q/K/V convolutions, input-dependent decay,
process noise, observation noise, FP32 state, and exact replay backward. The
fused CUDA loader has dedicated D=4/M=8 and D=8/M=8 kernels. Use it with:

```bash
python example/train_6layer.py --device cuda:0 --precision bf16 \
  --fused-rmsnorm --compile-components both --optimize-bayes \
  --bayes-backend fused --bayes-key-dim 4 --cuda-graph --gpu-batch-staging \
  --width 512 --heads 64 --kv-heads 4 --mlp-width 8619 \
  --seq 512 --batch 128 --steps 40 --warmup 5
```

This is a single-GB300 measurement of the complete forward, backward, gradient
clip, and AdamW step. The Mamba-3 row uses the same batch and graph settings,
with the parameter-matched 8,403-wide MLP:

| Model | Parameters | Bayesian state | Batch × sequence | Throughput | Peak allocated VRAM | Validation loss |
|---|---:|---:|---:|---:|---:|---:|
| Bayesian fused | 101.936M | D=4, M=8 | 128 × 512 | **434,185 tok/s** | 60,197 MiB | 10.4264 |
| Bayesian fused | 101.942M | D=8, M=8 | 128 × 512 | 394,081 tok/s | 62,427 MiB | — |
| Mamba-3 matched hybrid | 101.944M | — | 128 × 512 | **660,341 tok/s** | 60,093 MiB | 10.4612 |

The mixed-width kernel is 10.2% faster and uses 3.6% less allocated memory than
the D=8 Bayesian path, and its recurrence-only benchmark is 29.6% faster. It
does not yet beat Mamba-3 end to end: at this matched single-GPU shape it is
1.52× slower. Dense covariance has quadratic state cost, so preserving the
paper's full covariance and input-dependent dynamics imposes a real lower
bound; claiming a Mamba-3 speedup would require either a faster kernel for the
remaining projections/FFNs or a smaller state width. The raw records are
[Bayesian D4/M8](example/results/gb300_6layer_bayes_paper_d4m8_fused_single.json),
[Bayesian D8/M8](example/results/gb300_6layer_bayes_paper_d8_fused.json), and
[Mamba-3](example/results/gb300_6layer_mamba3_paper_match_single_b128.json).

An earlier D=2/M=8 TE FP8 measurement, before the final projection-side
optimizations below, reached
628,401 tok/s at 101.928M parameters and 61,509 MiB allocated on one GB300.
This retains a dense 2×2 covariance and all input-dependent paper features;
the same Mamba-3 hardware run reaches 730,456 tok/s at 101.916M parameters.
It is therefore 14% slower at this point, rather than a claimed Mamba win.
The D=2 record is [here](example/results/gb300_6layer_bayes_d2m8_tefp8_fused_gate_single.json).

### Current TE FP8 end-to-end result

The fastest measured configuration keeps every paper mechanism enabled
(learned input-dependent decay, process and observation noise, causal length-4
Q/K/V convolutions, dense covariance, and FP32 recurrent state). It uses the
explicit TE FP8 optimizations `--fuse-bayes-gate`,
`--fuse-te-aux-projection`, `--bf16-state-projection`, and
`--fuse-bayes-norm-linear`; the last two flags only
changes SiLU and per-head normalization before the inputs are cast to the
unchanged FP32 recurrence. A whole-sequence replay chunk removes unnecessary
chunk launch overhead at this fixed 512-token shape.

These are single-GB300, batch-128, sequence-512 CUDA-graph measurements in the
current PyTorch 2.14.1/TransformerEngine 2.18 runtime. Each timed step includes
forward, backward, gradient clipping, and fused AdamW. The Mamba row was rerun
with the same graph, optimizer, data, and batch settings.
The Bayesian D=1 row has a scalar dense covariance per head; it retains all
paper mechanisms but has lower state capacity than the default D=8 row.

| Model | Parameters | State | Throughput | Peak allocated VRAM | Validation loss |
|---|---:|---|---:|---:|---:|
| Mamba-3 TE FP8 matched hybrid | 101.916M | Mamba-3 SISO + GQA | **385,083 tok/s** | 63,415 MiB | 10.3737 |
| Bayesian TE FP8, all optimizations | 101.828M | D=1, M=8; scalar covariance/head | 366,615 tok/s | 61,225 MiB | 10.3918 |
| Bayesian TE FP8, all optimizations | 101.928M | D=2, M=8; dense 2×2 covariance/head | 355,014 tok/s | 61,610 MiB | 10.3837 |
| Bayesian TE FP8, optimized D=4 | 99.918M | D=4, M=8; dense 4×4 covariance/head | 323,185 tok/s | 62,492 MiB | 10.4752 |
| Bayesian TE FP8, optimized D=8 | 101.500M | D=8, M=8; dense 8×8 covariance/head | 298,740 tok/s | 63,480 MiB | 10.4705 |

The D=1 run is 4.8% slower than Mamba-3 in this current runtime, while retaining
the full paper feature set. Its parameter count is 0.087% lower because TE
requires the MLP output dimension to be a multiple of 16; `--mlp-width 8736`
is the nearest valid width to the 101.916M Mamba configuration. D=2, D=4, and
D=8 preserve progressively more Bayesian state capacity at a throughput cost.
The previous 758,504 versus 756,067 tok/s pair was measured with an older
Torch/TE stack and is retained only in the historical result JSON. Run the
fastest current configuration with:

```bash
python example/train_6layer.py --device cuda:0 --precision te-fp8 \
  --fused-rmsnorm --compile-components all --optimize-bayes \
  --bayes-backend fused --bayes-key-dim 1 --fuse-bayes-gate \
  --fuse-te-aux-projection --bf16-state-projection --fuse-bayes-norm-linear \
  --cuda-graph \
  --gpu-batch-staging --chunk-length 512 --width 512 --heads 64 \
  --kv-heads 4 --mlp-width 8736 --seq 512 --batch 128 \
  --steps 120 --warmup 5
```

The current raw records are [Bayesian D=1](example/results/gb300_6layer_bayes_d1_tefp8_current.json),
[Bayesian D=2](example/results/gb300_6layer_bayes_d2_tefp8_current.json), and
[Mamba-3](example/results/gb300_6layer_mamba3_tefp8_current.json).

Use `--bayes-backend fused` when the lower-memory custom CUDA replay path is preferred. The recorded paper-feature result uses the Inductor backend with learned input-dependent dynamics; backend rankings can change with sequence length and batch shape.

The D1, D2, D4, and D8 kernels have received exact SM103 optimizations. D1
replay backward now uses warp-shuffle reductions; D2/D4 remove the temporary
covariance tile, reuse `decay * key`, store only the symmetric replay triangle,
and use warp barriers for their single-warp launches. On GB300, the D1 direct
dynamic forward-plus-backward benchmark improved from 6.067 ms to 4.039 ms
(1.50× throughput) at `N=8192, T=512, D=1, M=8`; its full parity check was
≤`4.8e-7` forward and ≤`3.9e-6` gradients. The D2 benchmark improved from
25.371 ms to 22.532 ms at `N=32768, T=512, D=2, M=8` (1.13× throughput),
with ≤`4.8e-7` forward error and ≤`3.9e-6` gradient error.

The D8 kernel has also received exact SM103 optimizations: the forward
covariance prediction and mean innovation reuse `d * k`, covariance adjoints
compute one symmetric triangle, and the backward decay gradient hoists the
per-row reciprocal of the input-dependent decay.  On a GB300 with
`N=8192, T=512, D=M=8`, full dynamic forward plus backward through all Q/K/V,
decay, process-noise, and observation-noise inputs measured **16.05 ms →
10.67 ms (1.50× throughput)** over 30 CUDA-event runs.  The maximum differences against
the dense PyTorch reference were `7.2e-7` in forward outputs and `1.9e-6` in
gradients.  Dense covariance, FP32 state, and all paper input dependencies
remain enabled.  The experimental on-chip replay fusion is kept behind
`BAYES_SSM_FUSED_REPLAY=1`; it is disabled by default because its larger
shared-memory history lowers occupancy on this shape.

The paper's damped rotational variant is available with
`--bayes-dynamics rotation`. It embeds each scalar Q/K component as
`(component, 0)`, learns an input-dependent radius and angle for every 2-D
state pair, and injects process noise only into the first coordinate. The
covariance remains dense; the structured kernel applies the 2-D blocks without
materializing a dense transition matrix. This mode is selectable but is not the
diagonal-dynamics row in the benchmark table above.

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
