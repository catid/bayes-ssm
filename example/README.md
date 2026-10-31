# Six-layer language-model example

`train_6layer.py` is a small, self-contained causal-LM training recipe for
the hybrid model used in the repository's GB300 comparison:

| Block | Mixer |
| ---: | --- |
| 1, 2, 4, 5 | Exact dense Bayesian recurrence |
| 3, 6 | Causal GQA attention |

Every block has a SwiGLU feed-forward network. The token embedding and output
head are tied. The default width 640 / 10 query heads / 2 KV heads / 6144 MLP
width / 32k vocabulary instantiates 101,425,536 trainable parameters (about
100M). Bayesian covariance and all recurrent reductions are kept in FP32;
projection and MLP linears can use BF16 or TransformerEngine FP8.

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

The top-level README records the final GB300 smoke-training result and links
to its JSON artifact. The short default run is intended to validate the model
and training path; use a real token file and longer training schedule for
language-model experiments.
