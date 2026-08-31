# Results

## Protocol

All performance numbers come from synchronized, same-session comparisons on
one Colab TPU v5e. The artifacts retain the sample arrays and reported means.
Each permanent experiment contains at least these three arms:

- `regular_xla`: complete gate/up plus SwiGLU expressed normally in JAX.
- `gated_cubic`: blocked cubic Pallas control.
- `gated_strassen`: product-aware Strassen Pallas kernel.

Compilation is excluded. The tile search used the same enumeration rule, two
warmups, and five timed samples per candidate at every model width. All three
widths selected `(2048, 2048, 512)`.

## Isolated gate/up plus SwiGLU

| Model | XLA | Strassen | Speedup |
|---|---:|---:|---:|
| Qwen3-8B | `9.888 ms` | `8.343 ms` | `1.1851x` |
| Qwen3-14B | `16.670 ms` | `14.342 ms` | `1.1623x` |
| Qwen3-32B | `24.328 ms` | `20.921 ms` | `1.1628x` |

These microbenchmarks establish that the fused product-aware schedule can
capture the seven-product saving. They are not model latency.

## Complete real layer 0

All non-gate/up work is common ordinary XLA.

| Model | XLA | Cubic Pallas | Strassen | Strassen vs XLA |
|---|---:|---:|---:|---:|
| Qwen3-8B | `28.625 ms` | `29.143 ms` | `27.472 ms` | `1.0419x` |
| Qwen3-14B | `45.812 ms` | `46.410 ms` | `43.589 ms` | `1.0510x` |
| Qwen3-32B | `63.865 ms` | `66.523 ms` | `61.564 ms` | `1.0374x` |

The 32B value is the in-campaign replication. The original promoted run was
`1.0356x`.

## All layers

The harness streams real checkpoint weights one layer at a time, but times
only resident layer compute. Checkpoint transfer, embeddings, and final-logit
work are excluded.

| Model | Layers | XLA total | Strassen total | vs XLA | vs cubic | Faster layers |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 36 | `1030.201 ms` | `989.504 ms` | `1.0411x` | `1.0578x` | 36/36 |
| Qwen3-14B | 40 | `1833.816 ms` | `1745.330 ms` | `1.0507x` | `1.0645x` | 40/40 |
| Qwen3-32B | 64 | `4083.450 ms` | `3938.459 ms` | `1.0368x` | `1.0809x` | 64/64 |

The original 32B run measured `4091.586 ms` for XLA and `3948.805 ms` for
Strassen (`1.0362x`). The campaign replication independently reproduced the
speedup.

## Natural-text quality

The dataset revision is
`b08601e04326c79dfdd32d625aee71d232d685c3`. The first 32,768 contiguous
tokens of WikiText-2 test are split into 32 non-overlapping windows. The
windows are processed in four batches; they are not statistically independent
samples.

| Model | Native PPL | Strassen loss delta | Mean KL | Top-1 | Logit L2/maxnorm |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | `10.850` | `0.000316` | `0.000872` | `98.769%` | `1.981%/22.159%` |
| Qwen3-14B | `9.547` | `0.000070` | `0.000693` | `98.925%` | `1.619%/20.126%` |
| Qwen3-32B | `8.456` | `0.000633` | `0.001154` | `98.463%` | `2.543%/17.282%` |

All three pass the declared gate. This is teacher-forced next-token evidence,
not a generation or downstream-task evaluation.

## Interpretation

The result supports a narrow claim: a product-aware Strassen schedule can beat
ordinary XLA for favorable Qwen3 MLP inference projections even after complete
layer composition. It does not support transparent replacement of arbitrary
GEMMs, an end-to-end serving speedup, numerical equivalence, or full-model
training convergence.

The cubic Pallas control is useful but secondary. It shares the blocked Pallas
substrate and shows that rank-7 arithmetic is valuable behind that boundary.
Because its epilogue boundary is not exactly the same as product-aware
Strassen, the ordinary-XLA comparison remains the deployment-facing control.

## Evidence

The exact JSONL artifacts and SHA-256 values are indexed in
`results/README.md`. The broader post-snapshot search, including most failed
Qwen3 scheduling probes, stays on the research branch and is intentionally
absent from this public update.
