# Strassen GEMM on TPU v5e

This repository is a public research snapshot of one-level, tile-local
Strassen GEMM in JAX Pallas. The strongest result is a selective Qwen3 MLP
inference policy that beats ordinary XLA on summed resident layer compute for
8B, 14B, and 32B models on one TPU v5e.

## Main result

All timings are same-run measurements on Colab `TPU v5 lite`. Compilation and
checkpoint transfer are excluded.

| Model | Layers | Strassen vs XLA | Strassen vs cubic Pallas |
|---|---:|---:|---:|
| Qwen3-8B | 36 | `1.0411x` | `1.0578x` |
| Qwen3-14B | 40 | `1.0507x` | `1.0645x` |
| Qwen3-32B | 64 | `1.0368x` | `1.0809x` |

Strassen was faster than XLA in every measured layer: 36/36, 40/40, and
64/64. The trend is roughly flat rather than improving monotonically with
model size.

The primary control is ordinary XLA applied to the complete gate/up plus
SwiGLU function, with its normal fusion opportunities. The cubic Pallas arm
uses the same blocked custom-kernel substrate, but its epilogue boundary is
not identical to the product-aware Strassen arm. It is useful attribution
evidence, not a pure seven-versus-eight-product comparison.

## What worked

The successful policy is narrow:

1. Use Strassen only for large, favorable gate/up projections.
2. Keep full-size MXU products rather than shrinking the hardware tile.
3. Order the final K panel so the top gate/up pair becomes complete before the
   final products for the bottom pair.
4. Place the top SwiGLU store between those product groups, exposing VPU work
   to Mosaic while independent MXU work remains available.
5. Leave attention, down projections, thin shapes, and unsupported shapes on
   ordinary XLA.

The arithmetic saving alone was insufficient. The measurable model-level win
appeared only after the product schedule and consumer epilogue were designed
together.

## Natural-text quality

The quality follow-up uses the first 32,768 contiguous tokens of the
WikiText-2 test split, divided into 32 non-overlapping 1024-token windows. The
dataset is pinned to revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

| Model | Native perplexity | Loss delta | Mean KL | Top-1 agreement | Logit L2 drift |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | `10.850` | `0.000316` | `0.000872` | `98.769%` | `1.981%` |
| Qwen3-14B | `9.547` | `0.000070` | `0.000693` | `98.925%` | `1.619%` |
| Qwen3-32B | `8.456` | `0.000633` | `0.001154` | `98.463%` | `2.543%` |

The harness declares a gate of absolute loss delta `<= 0.01` nats, mean KL
`<= 0.02` nats, and top-1 agreement `>= 0.97`; every model passes. These
thresholds are declared in the harness and emitted before evaluation, but are
not claimed as an independently timestamped preregistration.

## Scope

This is not an end-to-end serving benchmark. The all-layer totals measure
resident layer compute and exclude checkpoint transfer, embedding lookup, and
final-logit work. A v5e cannot hold the 32B checkpoint, so layers are streamed
for quality evaluation.

The quality results are teacher-forced next-token measurements on one corpus.
Generation-mode error compounding and downstream task accuracy remain
unmeasured. The kernel is faster but not numerically equivalent to cubic GEMM.

Training is not a public claim of this snapshot. Short single-layer studies
found promising schedules, but they do not establish full-model convergence
or a training speedup.

## Repository map

- [`strassen_pallas.py`](strassen_pallas.py) — kernel and conservative native
  fallback.
- [`experiments/qwen3/`](experiments/qwen3/) — the promoted experiment, one
  runner, and its matched controls.
- [`evidence/qwen3/`](evidence/qwen3/) — immutable Qwen3 JSONL artifacts and
  their SHA-256 index.
- [`docs/RESULTS.md`](docs/RESULTS.md) — exact findings and interpretation.
- [`docs/COLAB_RUNBOOK.md`](docs/COLAB_RUNBOOK.md) — one-session reproduction
  procedure.
- [`tools/`](tools/) — TPU compatibility and evidence-integrity checks.
- [`archive/initial-snapshot/`](archive/initial-snapshot/) — the earlier
  kernel, VMEM, Mistral, and numerical-stability snapshot retained for
  provenance, not as the main reproduction path.

Fresh experiments write to the ignored `runs/` directory. They never
overwrite the published evidence tree.

Verify the public artifacts locally with:

```bash
python tools/verify_evidence.py
```

## Requirements

The measured environment used JAX/jaxlib `0.7.2`, libtpu `0.0.21.1`, and a
Colab TPU v5e runtime. The included compatibility probe should pass before a
measurement campaign:

```bash
python tools/check_tpu.py
```

See [`docs/COLAB_RUNBOOK.md`](docs/COLAB_RUNBOOK.md) before running any
benchmark. Use one TPU session and run the permanent XLA, cubic Pallas, and
Strassen arms in the same session.
