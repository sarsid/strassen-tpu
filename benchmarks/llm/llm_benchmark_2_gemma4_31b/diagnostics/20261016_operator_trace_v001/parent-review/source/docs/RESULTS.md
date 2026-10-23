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
[`../evidence/qwen3/README.md`](../evidence/qwen3/README.md) and
[`../evidence/gemma3/README.md`](../evidence/gemma3/README.md). The broader
post-snapshot search, including most failed Qwen3 scheduling probes, stays on
the research branch and is intentionally absent from this public update.

## Trillium (TPU v6e) and the fused q/k epilogue

Added 2026-09-05. Same protocol: same-run arms on one chip, compilation
excluded, raw samples retained in the artifacts.

### Pure GEMM, no epilogue

The bare BF16 GEMM at the gate/up geometry, with SwiGLU, product-aware
finalization and residuals all removed, so the rank-7 saving is separated
from the fusion work layered on top of it. On v6e all four arms select the
same tile `(2048, 1024, 5120)`, which makes the matched-tile cubic a fair
control.

| Arm | v6e mean | vs XLA |
|---|---:|---:|
| `strassen` | `4.673 ms` | `1.1117x` |
| `cubic_matched` | `5.091 ms` | — |
| `cubic_best` | `5.108 ms` | `0.9339x`* |
| `regular_xla` | `5.194 ms` | `1.0000x` |

Strassen against cubic at the identical tile is `1.0896x`, below the
`8/7 = 1.1429x` ceiling. On v5e the matched-tile figure reads `1.3101x`,
above the ceiling and therefore not an algorithm effect: there the cubic arm
is forced `8%` off its own optimum. Where the two chips disagree, the v6e
number is the trustworthy one. (*v5e figure.)

### The scoped-vmem flag, priced on one chip

| Arm | 48 MiB | 128 MiB | cost |
|---|---:|---:|---:|
| `regular_xla` | `5.194 ms` | `5.409 ms` | `+4.1%` |
| `strassen` | `4.673 ms` | `4.668 ms` | `-0.1%` |

At the layer rather than the GEMM the same flag costs XLA `37%`, so the
penalty is in the surrounding work, not the matmul. Left unfixed it would
have inflated the pure-GEMM headline from `1.1117x` to `1.1586x`.

### Fused q_norm+RoPE, paired on and off

One block, identical policy and budget, differing only in whether q and k
carry their per-head RMSNorm and RoPE inside the kernel.

| Chip | off | on | ratio |
|---|---:|---:|---:|
| v5e | `60.469 ms` | `55.346 ms` | `1.0562x` -> `1.1566x` |
| v6e | `16.100 ms` | `14.552 ms` | `1.0914x` -> `1.2093x` |

The saving is `8.0%` and `8.8%` of the respective blocks, close to a constant
fraction rather than a constant absolute cost.

### Streamed, all 64 layers

| Arm | total | vs XLA |
|---|---:|---:|
| `gated_strassen` | `871.73 ms` | `1.2950x` |
| `regular_xla` | `1128.88 ms` | `1.0000x` |
| `gated_cubic` | `1144.01 ms` | `0.9868x` |

Task gate passes: top-1 agreement `0.99963`, mean KL `1.374e-4` nats,
absolute loss delta `8.93e-5`, all finite. The per-layer logit L2 of
`0.15-0.20` is the accumulated-drift artifact documented for streamed mode
and is present in the cubic control as well.

This streamed figure exceeds the single-block `1.2093x` under the same
policy. The harnesses differ in arm count, sample budget and when the q/k
layouts are materialised; the gap is unexplained and should not be read as
the fusion improving with depth.



## Gemma 3 27B: generalisation to a second family

Added 2026-10-03. Same protocol throughout: same-run arms on one chip,
compilation excluded, raw samples retained.

Gemma 3 27B was picked because it is harder for this kernel than Qwen3 in
four concrete ways — four norms per block rather than two, `(1 + w)` RMSNorm
rather than `w`, GeGLU rather than SwiGLU, and 5:1 sliding/global attention
carrying two RoPE bases (`10000` and `1000000` with a linear factor of 8). It
also ties `embed_tokens` as the head and scales embeddings by
`sqrt(hidden_size)`.

Before any timing, `tools/reference_fidelity.py` ran one real decoder layer
through `transformers` and through our formulation, CPU, FP32. Both Gemma and
Qwen reach l2 `0.0`, max_abs `0.0` — bit-exact. The tool was wrong first: with
`attention_mask=None` HF eager attention is bidirectional, which reported l2
`1.079` for a block that is exact.

### Layer-0 block, real weights

8 x 1024 tokens, tile `(1024, 768, 5376)`, kernel budget 104 MiB,
`--xla_tpu_scoped_vmem_limit_kib=49152`.

| Device | Policy | XLA | best Strassen | Ratio |
|---|---|---:|---:|---:|
| v5e | gate/up + GeGLU, fused q/k | `49.798 ms` | `43.015 ms` | `1.1577x` |
| v5e | gate/up + GeGLU only | `49.743 ms` | `46.139 ms` | `1.0781x` |
| v6e | gate/up + GeGLU, fused q/k | `14.884 ms` | `13.173 ms` | `1.1299x` |

The fused `q_norm`+RoPE epilogue is worth 8 points on Gemma, the same as on
Qwen3. Gemma puts its per-head RMSNorm in the same place, so the fusable unit
is the same shape, and the finding travels.

We predicted Gemma would finish *below* Qwen3, because its norms sit between
`o`/`down` and their residual adds and so block the fusion that wins those
sites on Qwen3. It matched instead: `1.1577x` against `1.1566x`. Recorded as
a wrong prediction, not a confirmation.

### A refuted epilogue, retained

If a norm blocks the `o`/`down` residual fusion, the apparent fix is to fuse
the norm too. `norm_residual_add` does that. It loses:

| Policy | XLA | best Strassen | Ratio |
|---|---:|---:|---:|
| o/down left to XLA | `49.740 ms` | `42.997 ms` | `1.1568x` |
| o/down routed, `norm_residual_add` | `49.758 ms` | `45.898 ms` | `1.0841x` |

**7.3 points worse.** The epilogue needs a row sum-of-squares across the full
output width, which forces `bn == n` and surrenders the tile freedom the
gate/up site depends on. The traffic saved was never going to pay for that:
the ceiling on the entire manoeuvre was about 1% of the block, and we did not
compute it before building the epilogue. Reproduce with `--norm-residual`.

The general rule this yields is sharper than "fuse the epilogue or don't route
the site": a fusable unit is only worth fusing if its reduction axis fits
inside the tile. `qk_norm_rope` reduces over `head_dim=128` and fits.
`norm_residual_add` reduces over the output width and does not.

### All layers streamed, 62 layers

Repetitive corpus, gate/up only, v5e: top-1 agreement `0.99890`, mean KL
`2.36e-5` nats, absolute loss delta `1.00e-4`. Gate passes.

Two gates are published **failing**, and both predate the BOS fix below:

| Gate | Threshold | Strassen | Verdict |
|---|---:|---:|---|
| WikiText-2 streamed top-1 | `0.97` | `0.96215` | fails |
| WikiText-2, task records only | `0.97` | `0.95910` | fails |
| HellaSwag choice agreement | `0.97` | `0.9875` | passes |
| LAMBADA greedy agreement | `0.97` | `0.9625` | fails |

Native top-1 on the WikiText-2 arm is `0.3952` — a near-flat output
distribution, the regime in which an agreement metric is maximally sensitive
to any perturbation. That flatness was itself the BOS defect. These runs are
retained as failures and are **pending re-run**; they are not evidence that
Gemma fails quality.

## Scoring under the published harness

Added 2026-10-03. Every quality number above is produced by our own scoring
code. That is sufficient to detect a defect in one arm and structurally
incapable of detecting a defect in both.

`experiments/lm_eval/` registers the streamed model with
lm-evaluation-harness as the model type `strassen`, so the reference harness
owns tokenisation, prompt construction, scoring and metric choice, and we
supply only a forward pass. HellaSwag, `--limit 200`, 256-token contexts:

| Arm | acc | acc_norm |
|---|---:|---:|
| `regular_xla` | `0.580` | `0.745` |
| `gated_strassen` | `0.580` | `0.745` |

Zero delta on both metrics. `acc_norm` is the character-length-normalised
metric the published tables report; `acc` is included because reporting the
wrong one of the two was an earlier error here.

Against a published `~0.85`, `0.745` leaves about `1.7` sigma at n=200 (one
sigma is `0.031`) plus two systematics we do not correct: this is the `-it`
checkpoint rather than the base model, and contexts are truncated to 256
tokens to keep a 54 GB streamed forward pass affordable.

### The defect it found

The first run under the harness scored `acc_norm 0.460`. The cause was a
missing BOS token. Gemma is trained with a leading `<bos>` and collapses
toward a flat output distribution without one; lm-eval's own HF model
prepends it, so a wrapper that omits it is not running the published
protocol. Adding it moved the score to `0.745` — **+28.5 points**, far larger
than any effect this project has measured from the kernel itself.

Nothing in this repository could have caught it. Both arms omitted the token
identically, so it cancels *exactly* in any agreement metric: WikiText-2
top-1, downstream choice agreement, per-layer divergence, block error norms.
All were structurally blind to it, and two of them were reporting failures
caused by the handicap rather than by the kernel.

The standing lesson: self-consistency gates catch kernel defects; only an
external anchor catches protocol defects. This project had thorough coverage
of the first kind and none of the second, and the second found more.
