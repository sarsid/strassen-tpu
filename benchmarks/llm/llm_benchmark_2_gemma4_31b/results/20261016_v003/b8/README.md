# Completed Gemma4-31B B8/S1024 benchmark — v003

All five stages completed: tuning, block qualification, resident confirmation,
natural-text quality, and serialized offloaded prefill. All scientific gates
passed. Full candidate logs, selected local/global profiles, raw timing samples,
error metrics, exact token hashes and BOS policy are preserved in `artifacts/`.
The downloaded archive matches SHA-256
`0bfc3700ea1d6fe7e28009a08ec838c3da8c1613da3bb5ed16051c24031f277b`.

| Algorithm | Sum of resident-layer means (ms) | Offloaded prefill mean (s) |
|---|---:|---:|
| Native | 1207.597 | 240.251 |
| Tuned Native | 1065.998 | 240.110 |
| Tuned Cubic | 1097.026 | 239.402 |
| Tuned Strassen S1 | 935.688 | 239.586 |

Strassen's resident-layer speedups are 1.2906x versus Native (22.52% less time)
and 1.1393x versus Tuned Native (12.22% less time). This scope excludes embedding,
head, compilation, weight acquisition/preparation and host/device transfers.

The offloaded scope is a different measurement: serialized cached-disk weight
loading, common preparation, recurring host/device transfers, embedding, all
60 decoder layers and last-position head logits. It excludes compilation,
network checkpoint downloads and autoregressive decode. Each arm has one warmup
and three measured passes, with rotated order. All 12 measured outputs were
finite. Strassen's means are 0.276% lower than Native and 0.218% lower than Tuned
Native. Sample ranges overlap, so these small differences do not establish a
meaningful offloaded-latency benefit. Cubic has the lowest offloaded sample mean,
despite its slower resident-layer sum. Do not equate these scopes or claim that
the 22.52% resident reduction carries over to offloaded inference.

This offloaded harness spends about 57.7 seconds per pass reading 64.2 GB from
the private checkpoint cache. The remaining wall time includes host preparation,
transfers and model execution; the current instrumentation does not separate all
of those costs. The approximately one-second resident decoder sum is a small
part of the roughly 240-second serialized offloaded pass.

Natural-text quality: four independent B8/S1024 segments, each sequence starting
with BOS; 32,736 scored next-token positions. Strassen perplexity 5.713757 versus
Native 5.714063, absolute NLL delta 0.00005351, mean KL 0.00188602 nats, and
predicted-token agreement 98.3229%. All unchanged limits passed. This agreement
is relative to Native, not label accuracy. The tiny perplexity decrease is not
evidence of a quality improvement. Natural-logit relative L2 is 4.4912%, max
absolute logit difference 18.8125; retain these alongside distribution metrics.
Repetitive quality passed too (8,184 positions, agreement 99.9878%).

Software and source are frozen in the v003 source manifest under
`../../../operations/approved_validation_v002/`. CPU FP32 formula fidelity is
reused only after source/checkpoint hash verification; earlier BF16 cross-backend
failures remain archived. This batch is complete; five other batches are not
represented here. At preservation, B16 tuning was active on the same owned v6e.
