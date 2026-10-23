# Interim Qwen3-32B BF16 study on v6e: first three workloads

Three of ten workloads and 15 of 50 comparisons are complete: batch 1, sequence lengths 512, 1024 and 2048. All configured finite-output, NLL, KL and token-agreement gates passed.

At B1/S2048, S1 lowers resident-layer time by 11.05% and S2 by 3.37% versus Tuned Native. Their full-forward reductions are only 0.95% and 0.57%. Neither has a 95% matched-round interval entirely above one on any completed workload. These intervals are per workload and unadjusted for multiple comparisons; the larger and batched workloads remain unfinished.

## Full forward, median milliseconds

| Batch | Sequence | Default Native | Tuned Native | Tuned Cubic | S1 | S2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 4882.1 | 4881.9 | 4906.4 | 4893.3 | 4925.8 |
| 1 | 1024 | 4998.4 | 4970.4 | 4973.6 | 4982.5 | 5021.1 |
| 1 | 2048 | 5252.9 | 5249.4 | 5189.5 | 5199.6 | 5219.3 |

## Separate resident-layer pass, milliseconds

| Batch | Sequence | Default Native | Tuned Native | Tuned Cubic | S1 | S2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 82.2 | 75.4 | 74.4 | 75.2 | 93.3 |
| 1 | 1024 | 162.9 | 146.7 | 152.2 | 147.3 | 163.3 |
| 1 | 2048 | 414.6 | 384.6 | 401.1 | 342.1 | 371.6 |

BF16 output, DEFAULT arithmetic, v6e; streamed full forward includes recurring serialized H2D, embedding and all-token logits; compilation and prepared host layouts excluded. Resident pass is a separate sum of layer medians, not whole-model resident latency.

At B1/S2048, S1 resident time is 342.1 ms versus 384.6 ms for Tuned Native; S2 takes 371.6 ms. S1 is effectively tied with Tuned Native at 512 and 1024 tokens, while S2 is slower. In the streamed B1/S2048 run, Tuned Native has median layer transfers of 4665.5 ms plus a 169.3 ms head transfer, compared with 5249.4 ms full latency. These component medians are not additive, but establish that transfers dominate. This limits how much faster matrix multiplication can improve the full run.

## Numerical differences from Default Native

| Algorithm | Perplexity change | Next-token top-1 agreement | Logit relative L2 | Maximum layer relative L2 |
|---|---:|---:|---:|---:|
| Tuned Native | -0.345% to +0.094% | 98.44%–98.88% | 2.34%–3.42% | 4.78% |
| Tuned Cubic | -0.102% to +0.180% | 98.58%–99.02% | 2.72%–4.01% | 5.43% |
| S1 | -0.245% to +0.128% | 98.43%–98.73% | 3.23%–4.52% | 6.29% |
| S2 | -0.597% to +0.216% | 98.09%–98.83% | 4.06%–5.43% | 7.36% |

Quality uses fixed held-out WikiText2 windows, not downstream task accuracy or independent full-dataset replications. Perplexity changes of either sign do not establish a quality improvement. Intermediate hidden errors and maximum logit errors are diagnostics; passing final predictive gates does not imply identical activations. S1 maximum layer relative L2 is 6.29%; S2 is 7.36%.

All per-workload predictive metrics, error ranges, confidence intervals, entry identities and evidence-run paths are preserved in `summary.json`. Full profiles, tiles, per-layer errors and timing samples remain in the committed campaign ledger and case archives.

Per-workload paired bootstrap of the ratio of medians across 15 matched rounds; 10000 resamples, seed 20261003; percentile 95% intervals, no multiple-comparison adjustment.

Ledger SHA256: `adf78e9da14d66100716568c15a62683ecc219f3c49d60a2f98088021da28781`.
