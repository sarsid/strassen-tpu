# Completed Qwen3-8B BF16 study on v6e

All 10 workloads and 50 comparisons are complete. All configured finite-output, NLL, KL and token-agreement gates passed.

S1 has some resident-layer wins, but no workload has a 95% matched-round interval entirely above one for full-forward speedup over Tuned Native. S2 has no such win either. Tuned Native is clearly faster than S1 and S2 on B4/S2048 and B8/S2048. These intervals are per workload and unadjusted for multiple comparisons.

## Full forward, median milliseconds

| Batch | Sequence | Default Native | Tuned Native | Tuned Cubic | S1 | S2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 1421.7 | 1406.3 | 1428.7 | 1400.2 | 1414.6 |
| 1 | 1024 | 1413.5 | 1428.8 | 1422.9 | 1411.9 | 1413.7 |
| 1 | 2048 | 1474.1 | 1468.2 | 1476.7 | 1473.1 | 1492.2 |
| 1 | 4096 | 1687.7 | 1674.3 | 1677.7 | 1664.3 | 1683.5 |
| 4 | 512 | 1457.4 | 1446.7 | 1450.3 | 1442.5 | 1445.6 |
| 4 | 1024 | 1700.1 | 1724.5 | 1714.8 | 1719.1 | 1729.6 |
| 4 | 2048 | 2010.4 | 1920.1 | 1960.3 | 1946.9 | 1968.4 |
| 8 | 512 | 1567.8 | 1558.0 | 1555.5 | 1543.3 | 1555.4 |
| 8 | 1024 | 1776.6 | 1751.0 | 1771.6 | 1755.0 | 1818.3 |
| 8 | 2048 | 2590.9 | 2311.5 | 2404.1 | 2379.1 | 2444.5 |

## Separate resident-layer pass, milliseconds

| Batch | Sequence | Default Native | Tuned Native | Tuned Cubic | S1 | S2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 42.9 | 40.5 | 45.2 | 43.0 | 47.8 |
| 1 | 2048 | 101.8 | 99.8 | 102.0 | 95.7 | 104.8 |
| 1 | 4096 | 318.1 | 296.7 | 313.5 | 303.2 | 315.6 |
| 4 | 512 | 79.6 | 71.7 | 78.6 | 73.6 | 82.5 |
| 4 | 1024 | 139.2 | 139.2 | 131.9 | 123.1 | 140.0 |
| 4 | 2048 | 364.0 | 289.4 | 314.3 | 298.2 | 330.6 |
| 8 | 512 | 159.6 | 149.0 | 152.1 | 142.7 | 158.0 |
| 8 | 1024 | 298.5 | 284.1 | 325.7 | 294.5 | 319.3 |
| 8 | 2048 | 854.0 | 559.8 | 645.2 | 605.4 | 669.7 |

BF16 output, DEFAULT arithmetic, v6e; streamed full forward includes recurring serialized H2D, embedding and all-token logits; compilation and prepared host layouts excluded. Resident pass is a separate sum of layer medians, not whole-model resident latency. B1/S512 predates the separate resident pass.

S1 has lower resident time than Tuned Native on 3 of the 9 comparable workloads; S2 on none. The strongest S1 reduction is B4/S1024: 123.1 versus 139.2 ms, or 11.6%. Its full-forward advantage there is only 0.3%, with an interval spanning a tie.

## Numerical differences from Default Native

| Algorithm | Perplexity change | Next-token top-1 agreement | Logit relative L2 | Maximum layer relative L2 |
|---|---:|---:|---:|---:|
| Tuned Native | -0.113% to +0.119% | 98.36%–100.00% | 0.00%–2.21% | 18.32% |
| Tuned Cubic | -0.443% to +0.086% | 98.29%–99.41% | 1.70%–2.86% | 10.05% |
| S1 | -0.551% to +0.300% | 98.31%–99.12% | 1.92%–3.22% | 10.13% |
| S2 | -0.473% to +0.334% | 97.99%–98.92% | 2.74%–3.92% | 13.64% |

Quality uses fixed held-out WikiText2 windows, not downstream task accuracy or independent full-dataset replications. Perplexity changes of either sign do not establish a quality improvement. Intermediate hidden errors and maximum logit errors are diagnostics; passing final predictive gates does not imply identical activations. The largest intermediate deviations merit follow-up.

All per-workload predictive metrics, error ranges, confidence intervals, entry identities and evidence-run paths are preserved in `summary.json`. Full profiles, tiles, per-layer errors and timing samples remain in the committed campaign ledger and case archives.

Per-workload paired bootstrap of the ratio of medians across 15 matched rounds; 10000 resamples, seed 20261003; percentile 95% intervals, no multiple-comparison adjustment.

Ledger SHA256: `297e475ad4fe03fd78d2ee6808feee566774603e80064fd03642d8f7b4a9f7b5`.
