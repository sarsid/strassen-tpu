# B4/S1024 resident and quality checkpoint

Tuning, representative block qualification, all 60 resident decoder-layer
measurements and both repetitive/natural quality evaluations completed.
Every arm passed all unchanged scientific gates. Serialized cached-disk
offloaded prefill is still running; no final offloaded result is claimed here.

| Algorithm | Sum of warmed resident-layer means (ms) |
|---|---:|
| Native | 562.608 |
| Tuned Native | 515.937 |
| Tuned Cubic | 533.960 |
| Tuned Strassen S1 | 466.894 |

Strassen saves 17.01% versus Native
(1.2050x) and
9.51% versus Tuned Native
(1.1050x). These sums exclude embedding,
head, compilation, preparation and transfers; they are not full-model wall-clock
latency. The raw bootstrap intervals resample per-layer mean differences,
not independent full-model repetitions.

Natural quality uses four B4/S1024 segments, BOS per sequence and
16,368 scored next-token positions. Strassen perplexity is
4.980553905 versus Native 4.980536229, a
+0.000355% change. Absolute NLL delta is
0.000003549, KL 0.001935174 nats and top-1
agreement 98.3688%. Logit relative L2 is
4.6552% and maximum absolute logit difference
14.125. Agreement compares with Native predictions, not true-label
accuracy. Preserve logit diagnostics alongside task-level metrics. Natural
limits remain abs NLL delta <=0.01, KL <=0.02 and agreement >=97%.

The saved local/global tuning profile and complete numerical summaries are
preserved here. Raw per-stage samples/events will accompany the complete batch
archive after offloaded prefill finishes. Absolute perplexities across batches
cover different corpus spans and are not a controlled batch-size comparison.

B8/B16/B32 are fully complete. B64's separate diagnostic OOM failure and its
validated pending repair remain preserved. B1 follows B4, then the prepared
B64-only retry runs on the same retained TPU.
