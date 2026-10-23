# Gemma4-31B completed six-batch study

All six batches passed the approved formula-fidelity and unchanged same-TPU BF16 algorithm gates. Four arms, sequence length 1,024, BF16 output, FP32 accumulation, DEFAULT precision. Strassen S1 is fastest in summed resident decoder-layer time at every batch.

## Resident decoder-layer time (ms)

| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 saving vs Native | S1 saving vs Tuned Native |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 137.584 | 131.293 | 135.821 | 121.924 | 11.38% | 7.14% |
| 4 | 562.608 | 515.937 | 533.960 | 466.894 | 17.01% | 9.51% |
| 8 | 1207.597 | 1065.998 | 1097.026 | 935.688 | 22.52% | 12.22% |
| 16 | 2445.893 | 2091.712 | 2235.421 | 1858.854 | 24.00% | 11.13% |
| 32 | 4796.475 | 4151.352 | 4462.521 | 3769.584 | 21.41% | 9.20% |
| 64 | 9923.863 | 8349.988 | 8979.613 | 7540.176 | 24.02% | 9.70% |

These are sums of warmed layer means, not whole-model wall-clock measurements. They exclude embedding/head computation, weight preparation and transfers. Per-layer bootstrap intervals in raw summaries do not represent independent full-model repetitions.

## Serialized offloaded prefill (seconds)

| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 time saving vs Native |
|---:|---:|---:|---:|---:|---:|
| 1 | 241.456 | 242.005 | 242.477 | 242.330 | -0.362% |
| 4 | 242.690 | 242.869 | 242.914 | 242.184 | 0.209% |
| 8 | 240.251 | 240.110 | 239.402 | 239.586 | 0.276% |
| 16 | 242.562 | 242.062 | 242.414 | 241.168 | 0.575% |
| 32 | 247.072 | 246.424 | 246.386 | 246.267 | 0.326% |
| 64 | 253.453 | 252.262 | 252.805 | 251.593 | 0.734% |

Each mean has three measured passes per arm after one warmup, in rotated arm order. This scope includes cached disk reads, common weight preparation, recurring H2D transfers, embedding, all decoder layers and last-position logits. It excludes compilation, initial network downloads and autoregressive decode.

B64 saves 1.860 seconds (0.734%) versus Native and 0.669 seconds (0.265%) versus Tuned Native. Its three S1 samples are all below the three Native samples; the S1 and Tuned Native ranges overlap. This small-sample observation supports a modest gain in this particular offloaded configuration, not the 24% whole-inference speedup implied by confusing scopes. The other five batches have overlapping sample ranges.

## Natural-text quality

| Batch | Native perplexity | S1 perplexity | S1 relative PPL change | S1 top-1 agreement | S1 logit relative L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.007357 | 6.009909 | +0.042477% | 98.5093% | 4.8469% |
| 4 | 4.980536 | 4.980554 | +0.000355% | 98.3688% | 4.6552% |
| 8 | 5.714063 | 5.713757 | -0.005351% | 98.3229% | 4.4912% |
| 16 | 6.057986 | 6.054949 | -0.050127% | 98.3550% | 4.3656% |
| 32 | 5.793672 | 5.794087 | +0.007160% | 98.3886% | 4.3586% |
| 64 | 5.799099 | 5.799367 | +0.004634% | 98.3905% | 4.4380% |

Top-1 agreement is against Native, not ground-truth accuracy. All task gates passed; nonzero logit error remains reported. Natural inputs use BOS plus 1,023 contiguous corpus tokens per independent sequence, four segments. Different batch sizes consume different corpus spans, so absolute perplexity across rows is not a controlled batch-size effect.

## Provenance and limits

B1/B4/B8/B16/B32 come from the original v003 sweep. B64 reuses its exact saved tuning on the same retained TPU and software stack, with versioned bounded diagnostics and host parking of inactive diagnostic/quality buffers. Full B64 kernel dimensions and measured computation are unchanged; no microbatching or quality-gate relaxation. The original failures, first failed retry, failed local validation and successful repair remain preserved separately. No partial failed timing rounds are pooled.

The companion summary.json records every source path, source-summary SHA-256, archive SHA-256, raw prefill samples and full quality metrics. Complete stage logs, profiles, errors and results are under the linked result directories. This experiment does not measure serving throughput, decode latency or an all-weights-resident whole model.
