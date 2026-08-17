# strassen-tpu

One-level classical Strassen GEMM for a single TPU v5e, written in JAX
Pallas. On large aligned BF16 matmuls it beats native XLA — a baseline
already at ~93% of the chip's 197 TFLOP/s peak — by `1.07–1.10x`, with the
speedup attributed to the seven-product formula itself via cubic control
kernels on the identical substrate. See [RESULTS.md](RESULTS.md) for the
headline numbers, including the Mistral-7B evaluation and the task-level
deployment gate. A full writeup is forthcoming.

## Layout

| Path | Contents |
|---|---|
| `strassen_pallas.py` | The kernel, VMEM profiles, and the conservative exact-shape dispatcher (`tuned_matmul`, `strassen_matmul`, `native_matmul`). |
| `benchmarks/` | The experiments behind every claim in RESULTS.md: GEMM release gates, cubic attribution controls, VMEM screens, and the Mistral-7B block/layer/full-model/task-level gates. |
| `benchmarks/screens/` | Historical screens of rejected ideas (alternative rank-7 formulas, diagonal scalings, sparse layer schedules, datatype sweeps) — retained because their raw results are part of the evidence record. |
| `results/` | Raw JSONL output of every run, SHA-256-hashed; `results/README.md` maps questions to files. |
| `COLAB_RUNBOOK.md` | The exact one-session reproduction procedure (Colab TPU v5e, pinned JAX/libtpu). |

## Why so many benchmark files

One file per registered experiment, on purpose: each declares its gate
before running, produces one hashed JSONL, and is never edited to fit a
result. The layout separates the experiments that back current claims
(`benchmarks/`) from the closed investigations (`benchmarks/screens/`),
but both stay reproducible.

## Requirements

A Colab `TPU v5 lite` runtime with JAX/jaxlib 0.7.2 and libtpu 0.0.21 (a
narrowly guarded compatibility shim in `strassen_pallas.py` targets exactly
this pair). Checkpoint experiments additionally use `transformers`,
`huggingface_hub`, and `datasets`, and pin `mistralai/Mistral-7B-v0.1`
revision `27d67f1`.
