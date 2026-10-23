# Completed Gemma4-31B B4/S1024 benchmark — v003

All five stages completed and all scientific gates passed: tuning, representative
block qualification, resident confirmation, natural-text quality, and serialized
offloaded prefill. The downloaded archive SHA-256 was verified:
`19846b0fb783824f50fbf3f88fecff6c6639723ae566dce5541d5c45ae518d7f`.
Raw tuning candidates, local/global profiles, timing samples, error metrics,
input hashes and BOS policy are preserved in `artifacts/`. No activation or
weight tensors are included.

| Algorithm | Sum of resident-layer means (ms) | Offloaded prefill mean (s) |
|---|---:|---:|
| Native | 562.608 | 242.690 |
| Tuned Native | 515.937 | 242.869 |
| Tuned Cubic | 533.960 | 242.914 |
| Tuned Strassen S1 | 466.894 | 242.184 |

Strassen's resident speedups are 1.2050x versus
Native (17.01% less time) and
1.1050x versus Tuned Native
(9.51% less time). Resident sums
exclude embedding, head, compilation, preparation and transfers; they are not
whole-model wall-clock latency. Bootstrap intervals in the raw confirmation
summary resample per-layer mean differences, not independent full-model runs.

Offloaded prefill includes serialized cached-disk weight loading, common weight
preparation, recurring host/device transfers, embedding, all 60 decoder layers,
and last-position logits. It excludes compilation, network checkpoint acquisition
and autoregressive decode. One warmup and three measured passes were performed
per arm, with rotated ordering; all 12 measured outputs were finite.

Strassen's offloaded mean is 0.506 s
(0.209%) lower than Native and
0.685 s
(0.282%) lower than Tuned Native.
Individual sample ranges overlap. These three-pass data do not establish a
robust wall-clock improvement comparable to the resident gain. The serialized
harness averages 58.84 s reading checkpoint bytes per pass; remaining
time includes preparation, transfers and computation, which are not fully
separated by the current instrumentation.

Natural quality passed over 16,368 scored positions from four
B4/S1024 segments with BOS per sequence. Strassen perplexity is
4.980553905 versus Native 4.980536229
(+0.000355%), absolute NLL delta
0.000003549, KL 0.001935174 nats and top-1
agreement 98.3688%. Every arm passed the unchanged limits
(absolute NLL delta <=0.01, KL <=0.02, agreement >=97%). Repetitive quality
passed too. Logit relative L2 is 4.6552% and maximum
absolute difference 14.125; retain these diagnostics alongside
task metrics. Agreement is against Native predictions, not true-label accuracy.
The corpus spans differ across batches, so their absolute perplexities are not
a controlled comparison of batch size.

The v003 source and approved protocol are preserved in
`../../../operations/approved_validation_v002/`. B8/B16/B32/B4 are fully
complete; B1 tuning was running on the same owned v6e when this archive was
preserved. B64's diagnostic failure and pending validated retry remain separate.
