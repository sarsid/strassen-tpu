# Gemma4-31B B8/S1024 — resident timing and quality checkpoint

All four arms completed confirmation and passed the unchanged repetitive and
natural-text quality gates. The same owned v6e is running offloaded prefill;
its transfer-inclusive timing is not yet available in this checkpoint.

| Algorithm | Sum of warmed resident-layer means (ms) | Speedup vs Native |
|---|---:|---:|
| Native | 1207.596701 | 1.0000x |
| Tuned Native | 1065.998362 | 1.1328x |
| Tuned Cubic | 1097.025572 | 1.1008x |
| Tuned Strassen S1 | 935.687924 | 1.2906x |

Strassen saved 271.91 ms (22.52% time) versus Native and 130.31 ms (12.22%)
versus Tuned Native; its speedup over Tuned Native is 1.1393x. These are sums
of warmed resident decoder-layer means, excluding embedding, final head,
compilation and weight transfers. They are not a wall-clock whole-model
inference or offloaded-prefill latency measurement. The source summary's
bootstrap intervals resample per-layer mean differences; they are not intervals
from independent full-model runtime repetitions.

Natural-text results use four B8/S1024 segments with BOS per independent
sequence, scoring 32,736 next-token positions. All limits remained unchanged:
absolute NLL delta <=0.01, mean KL <=0.02, top-1 agreement >=97%.

| Algorithm | Perplexity | Absolute NLL delta | Mean KL (nats) | Top-1 agreement |
|---|---:|---:|---:|---:|
| Native | 5.714063 | 0 | 0 | 100% |
| Tuned Native | 5.716320 | 0.00039492 | 0.00157604 | 98.5062% |
| Tuned Cubic | 5.715983 | 0.00033594 | 0.00163562 | 98.4757% |
| Tuned Strassen S1 | 5.713757 | 0.00005351 | 0.00188602 | 98.3229% |

The tiny signed perplexity decrease for Strassen is not evidence of improved
model quality. Agreement compares predicted tokens with Native, not true-label
accuracy. Strassen natural-logit relative L2 is 4.4912%, with max absolute logit
difference 18.8125; these are reported diagnostics rather than stopping limits.
Repetitive quality also passed: Strassen NLL delta 0.00015646, KL 0.00007912,
agreement 99.9878% over 8,184 positions. Full raw summaries retain all errors.

The full batch archive, candidate traces and offloaded-prefill measurements
will follow separately. No other batch is completed by this checkpoint.
