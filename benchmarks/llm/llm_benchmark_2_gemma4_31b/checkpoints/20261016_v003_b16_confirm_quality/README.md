# Gemma4-31B B16/S1024 resident and quality checkpoint

Tuning, representative-block qualification, resident confirmation and both full-depth quality checks completed successfully. The same owned v6e is running offloaded prefill; that timing is still pending.

| Algorithm | Sum of warmed resident-layer means (ms) | Natural-text perplexity | Top-1 agreement |
|---|---:|---:|---:|
| Native | 2445.893043 | 6.057986 | 100.0000% |
| Tuned Native | 2091.711924 | 6.054617 | 98.5078% |
| Tuned Cubic | 2235.420683 | 6.053349 | 98.5414% |
| Tuned Strassen S1 | 1858.853954 | 6.054949 | 98.3550% |

Strassen saves 24.001% resident time versus Native and 11.132% versus Tuned Native (speedups 1.3158x and 1.1253x).

This scope sums warmed decoder-layer means and excludes embedding, final head, compilation, weight preparation and transfers. It is not whole-model wall-clock or offloaded-prefill latency. Bootstrap intervals in the raw summary resample per-layer mean differences, not independent full-model repetitions.

Natural quality scores 65,472 next-token positions from four B16/S1024 segments with BOS per sequence. Every arm passed unchanged absolute NLL delta <=0.01, KL <=0.02, and agreement >=97%. Strassen has absolute NLL delta 0.00050140, KL 0.00186306, and agreement 98.3550%. Its logit relative L2 is 4.3656% and maximum absolute difference is 19.65625; these are reported diagnostics. Repetitive quality passed all limits too. Full metrics are retained in the JSON summaries.

Agreement is with Native predictions, not true-label accuracy. A tiny perplexity decrease is not evidence of improved quality. B16 covers more held-out corpus positions than B8, so the change in baseline perplexity between batches is not a controlled effect of batch size.

This checkpoint preserves the selected local/global profiles and final stage summaries. Full candidate traces and offloaded-prefill results will follow in the immutable complete batch archive. B8 remains separately preserved; no B16 prefill conclusion is claimed here.
