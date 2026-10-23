# Completed Gemma4-31B B1/S1024 benchmark — v003

All five stages completed and all scientific gates passed. The downloaded
archive SHA-256 was verified: `f334e94aa270cb1ff2284a3e67591535ff99d36c4a6e4d83b735a9594fbb802f`.
Raw tuning candidates, profiles, samples, errors, input hashes and BOS policy
are preserved in `artifacts/`; no activation or weight tensors are included.

| Algorithm | Sum of resident-layer means (ms) | Offloaded prefill mean (s) |
|---|---:|---:|
| Native | 137.584 | 241.456 |
| Tuned Native | 131.293 | 242.005 |
| Tuned Cubic | 135.821 | 242.477 |
| Tuned Strassen S1 | 121.924 | 242.330 |

Strassen's resident speedups are 1.1284x versus
Native (11.38% less time) and
1.0768x versus Tuned Native
(7.14% less time). Resident sums
exclude embedding, head, compilation, preparation and transfers; they are not
whole-model wall-clock latency. Bootstrap intervals resample per-layer mean
differences, not independent full-model runs.

Offloaded prefill includes serialized cached-disk weight loading, common
preparation, recurring host/device transfers, embedding, all 60 decoder layers
and last-position logits. It excludes compilation, network checkpoint acquisition
and autoregressive decode. Each arm had one warmup and three measured passes
with rotated ordering; all 12 measured outputs were finite.

Strassen's offloaded mean is 0.874 s
(0.362%) higher than Native and
0.324 s
(0.134%) higher than Tuned Native.
Sample ranges overlap. These three-pass data show no clear offloaded wall-clock
gain despite the resident compute saving. The harness averages 58.94 s
reading checkpoint bytes per pass; preparation, transfers and computation are
not fully separated by the remaining instrumentation.

Natural quality passed over 4,092 scored positions from four
B1/S1024 segments with BOS per sequence. Strassen perplexity is
6.009908864 versus Native 6.007357114
(+0.042477%). Absolute NLL delta is
0.000424681, KL 0.001784540 nats and top-1
agreement 98.5093%. Every arm passed the unchanged natural
limits (abs NLL delta <=0.01, KL <=0.02, agreement >=97%) and repetitive gates.
Logit relative L2 is 4.8469%, with maximum absolute
difference 15.08203125. Agreement is against Native predictions,
not true-label accuracy. Absolute perplexities across batches use different
corpus spans and do not measure a controlled batch-size effect.

B1/B4/B8/B16/B32 are now fully complete. The predecessor ended with the separate
B64 diagnostic-memory failure and retained the TPU. The validated B64-only
retry was launched on that same endpoint with the exact saved profile; no
retuning or new allocation occurred. Earlier failed attempts remain preserved.
