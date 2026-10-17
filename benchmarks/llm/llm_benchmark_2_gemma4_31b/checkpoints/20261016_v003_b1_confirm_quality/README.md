# B1/S1024 resident and quality checkpoint

Tuning, representative block qualification, all 60 resident decoder-layer
measurements and repetitive/natural quality completed. Every arm passed the
unchanged scientific gates. Offloaded prefill is still running; no final
offloaded latency is claimed here.

| Algorithm | Sum of warmed resident-layer means (ms) |
|---|---:|
| Native | 137.584 |
| Tuned Native | 131.293 |
| Tuned Cubic | 135.821 |
| Tuned Strassen S1 | 121.924 |

Strassen saves 11.38% versus Native
(1.1284x) and
7.14% versus Tuned Native
(1.0768x). These sums exclude embedding,
head, compilation, preparation and transfers; they are not full-model wall-clock
latency. Raw bootstrap intervals resample per-layer mean differences rather
than independent full-model repetitions.

Natural quality uses four B1/S1024 segments, BOS per sequence, and
4,092 scored next-token positions. Strassen perplexity is
6.009908864 versus Native 6.007357114
(+0.042477%). Absolute NLL delta is
0.000424681, KL 0.001784540 nats and top-1
agreement 98.5093%. Logit relative L2 is
4.8469% and maximum absolute difference
15.08203125. Agreement compares with Native predictions, not true-label
accuracy. Preserve logit diagnostics alongside task-level metrics. Natural
limits remain abs NLL delta <=0.01, KL <=0.02 and agreement >=97%.

The saved local/global profiles and complete numerical summaries are preserved
here. Raw stage samples/events will accompany the complete batch archive after
prefill finishes. Different corpus spans across batches prevent interpreting
absolute perplexity differences as a controlled batch-size effect.

B8/B16/B32/B4 are fully complete. B64's diagnostic OOM failures and validated
repair remain separate; its saved-profile retry follows B1 on the same retained
TPU. This checkpoint does not claim that B64 or the whole sweep has completed.
