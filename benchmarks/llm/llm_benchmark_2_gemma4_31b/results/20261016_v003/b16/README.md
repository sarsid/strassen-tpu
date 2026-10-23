# Completed Gemma4-31B B16/S1024 benchmark — v003

All five stages completed and all scientific gates passed: tuning, representative
block qualification, resident confirmation, natural-text quality, and serialized
offloaded prefill. The archive hash was verified after download:
`168880bd72bcb1615b01034157ba5a0d39786386c5ad7ed9927e056e5e58ffae`.
Raw candidate logs, local/global profiles, samples, error metrics, input hashes
and BOS policy are preserved in `artifacts/`. No activation or weight tensors
are included.

| Algorithm | Sum of resident-layer means (ms) | Offloaded prefill mean (s) |
|---|---:|---:|
| Native | 2445.893 | 242.562 |
| Tuned Native | 2091.712 | 242.062 |
| Tuned Cubic | 2235.421 | 242.414 |
| Tuned Strassen S1 | 1858.854 | 241.168 |

Strassen's resident speedups are 1.3158x versus Native (24.00% less time) and
1.1253x versus Tuned Native (11.13% less time). Resident sums exclude embedding,
head, compilation, weight preparation and transfers; they are not whole-model
wall-clock latency. Bootstrap intervals in the raw confirmation summary resample
per-layer mean differences, not independent full-model repetitions.

Offloaded prefill includes serialized cached-disk weight loading, common weight
preparation, recurring host/device transfers, embedding, all 60 decoder layers,
and last-position logits. It excludes compilation, network checkpoint acquisition
and autoregressive decode. One warmup and three measured passes were performed
per arm, with rotated ordering; all 12 measured outputs were finite.

Strassen's offloaded mean is 1.394 s (0.575%) lower than Native and 0.894 s
(0.369%) lower than Tuned Native. Individual sample ranges overlap and the
Strassen range spans about 2.06 s. These three-pass data do not establish a
robust wall-clock improvement comparable to the resident gain. The current
serialized harness averages 58.13 s reading checkpoint bytes per pass; remaining
time includes preparation, transfers and computation, which are not fully
separated by the current instrumentation.

Natural quality passed over 65,472 scored positions from four B16/S1024 segments
with BOS per sequence. Strassen perplexity is 6.054949 versus Native 6.057986,
absolute NLL delta 0.00050140, KL 0.00186306 nats and top-1 agreement 98.3550%.
Every arm passed the unchanged limits (absolute NLL delta <=0.01, KL <=0.02,
agreement >=97%). Repetitive quality passed too. Logit relative L2 is 4.3656%
and max absolute difference 19.65625; retain these diagnostics with task metrics.
Agreement is against Native predictions, not true-label accuracy, and tiny
perplexity reductions are not evidence of improved quality. B16 covers more
held-out corpus positions than B8, so their absolute perplexities are not a
controlled comparison of batch size.

The v003 source and approved protocol are preserved in
`../../../operations/approved_validation_v002/`. Previous BF16 cross-backend
failures remain archived. B8 and B16 are now complete; B32 tuning was running
on the same owned v6e when this checkpoint was preserved.
