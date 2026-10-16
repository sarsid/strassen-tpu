# Completed Gemma4-31B B32/S1024 benchmark — v003

All five stages completed and all scientific gates passed: tuning, representative
block qualification, resident confirmation, natural-text quality, and serialized
offloaded prefill. The archive hash was verified after download:
`8ed85881bef2474718a1084615c33541fc42a8f2210955465c8cfa71ae08e3c7`.
Raw candidate logs, local/global profiles, samples, error metrics, input hashes
and BOS policy are preserved in `artifacts/`. No activation or weight tensors
are included.

| Algorithm | Sum of resident-layer means (ms) | Offloaded prefill mean (s) |
|---|---:|---:|
| Native | 4796.475 | 247.072 |
| Tuned Native | 4151.352 | 246.424 |
| Tuned Cubic | 4462.521 | 246.386 |
| Tuned Strassen S1 | 3769.584 | 246.267 |

Strassen's resident speedups are 1.2724x versus Native
(21.41% less time) and
1.1013x versus Tuned Native
(9.20% less time). Resident sums
exclude embedding, head, compilation, weight preparation and transfers; they
are not whole-model wall-clock latency. Bootstrap intervals in the raw
confirmation summary resample per-layer mean differences, not independent
full-model repetitions.

Offloaded prefill includes serialized cached-disk weight loading, common weight
preparation, recurring host/device transfers, embedding, all 60 decoder layers,
and last-position logits. It excludes compilation, network checkpoint acquisition
and autoregressive decode. One warmup and three measured passes were performed
per arm, with rotated ordering; all 12 measured outputs were finite.

Strassen's offloaded mean is 0.805 s
(0.326%) lower than Native and
0.158 s
(0.064%) lower than Tuned Native.
Individual sample ranges overlap; these three-pass data do not establish a
robust wall-clock improvement comparable to the resident gain. The current
serialized harness averages 58.78 s reading checkpoint bytes per pass;
remaining time includes preparation, transfers and computation, which are
not fully separated by the current instrumentation.

Natural quality passed over 130,944 scored positions from four
B32/S1024 segments with BOS per sequence. Strassen perplexity is
5.794087 versus Native 5.793672
(+0.00716%), absolute NLL delta
0.00007160, KL 0.00179820 nats and top-1
agreement 98.3886%. Every arm passed the unchanged limits
(absolute NLL delta <=0.01, KL <=0.02, agreement >=97%). Repetitive quality
passed too. Logit relative L2 is 4.3586% and max absolute
difference 19.65625; retain these diagnostics with task metrics.
Agreement is against Native predictions, not true-label accuracy. B32 covers
more held-out corpus positions than B8/B16, so absolute perplexities across
batches are not a controlled comparison of batch size.

The v003 source and approved protocol are preserved in
`../../../operations/approved_validation_v002/`. Previous BF16 cross-backend
failures remain archived. B8, B16 and B32 are now complete; B64 tuning was running
on the same owned v6e when this checkpoint was preserved.
