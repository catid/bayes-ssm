# Six-layer language-model example

`train_6layer.py` is a small, self-contained causal-LM training recipe for
the hybrid model used in the repository's GB300 comparison:

| Block | Mixer |
| ---: | --- |
| 1, 2, 4, 5 | Exact dense Bayesian recurrence |
| 3, 6 | Causal GQA attention |

Every block has a SwiGLU feed-forward network. The token embedding and output
head are tied. The fixed fast-path width 512 / 64 query heads / 4 KV heads /
8448 MLP width / 32k vocabulary instantiates 100,604,448 trainable parameters
(about 100M); the default paper-feature path is about 102.3M because it adds
learned noise, transition, and short-convolution projections. Bayesian
covariance and all recurrent reductions are kept in FP32;
projection and MLP linears can use BF16 or TransformerEngine FP8.

On CUDA with TransformerEngine installed, `--fused-rmsnorm` selects
TransformerEngine's fused RMSNorm while leaving projections in the requested
precision. The flag is optional and is rejected clearly on CPU or without TE.
`--compile-components both` independently compiles the GQA and SwiGLU modules;
the first run pays the compilation cost, while fixed-shape replays use the
compiled backward graphs.

Run a CPU smoke test with synthetic tokens:

```bash
PYTHONPATH=. python example/train_6layer.py \
  --device cpu --precision fp32 --width 32 --heads 4 --kv-heads 2 \
  --mlp-width 64 --vocab-size 256 --seq 32 --batch 2 \
  --steps 1 --warmup 0
```

Use `--tokens path.npy` (and optionally `--valid-tokens`) for a flat token
corpus. On CUDA, `--optimize-bayes` selects the compiled exact recurrence;
the first shape incurs compilation. `--precision te-fp8` requires a
CUDA-compatible TransformerEngine installation and fails explicitly when TE
is unavailable. Every run prints JSON and can write it with `--output run.json`.

The top-level README records the final GB300 training comparison and links to
the JSON artifacts. The apples-to-apples Mamba-3 example uses Mamba-3 in
blocks 1, 2, 4, and 5 and the same GQA attention in blocks 3 and 6;
`--mixer all-mamba` is available only as a separate six-Mamba control. The
Bayesian example defaults to the paper-feature path; pass
`--bayes-features fixed` to reproduce the published fixed-path throughput row.
For paper feature parity, use `--bayes-features paper --bayes-dynamics learned`;
this enables input-dependent process/observation noise and diagonal transitions
plus the causal Q/K/V short convolutions. The graph path requires fixed batch
and sequence shapes;
use the default launch path when shapes change between steps. The short
default run is intended to validate the model and training path; use a real
token file and longer training schedule for language-model experiments.
