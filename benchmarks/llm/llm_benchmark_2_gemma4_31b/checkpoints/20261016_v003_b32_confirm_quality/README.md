# Gemma4-31B B32/S1024 resident and quality checkpoint

Tuning, representative-block qualification, resident confirmation and both full-depth quality checks completed successfully. Offloaded prefill is running on the same owned v6e; its timing remains pending.

| Algorithm | Sum of warmed resident-layer means (ms) | Natural perplexity | Top-1 agreement |
|---|---:|---:|---:|
| Native | 4796.475498 | 5.793672 | 100.0000% |
| Tuned Native | 4151.352345 | 5.793317 | 98.5612% |
| Tuned Cubic | 4462.521437 | 5.791868 | 98.5689% |
| Tuned Strassen S1 | 3769.584207 | 5.794087 | 98.3886% |

Strassen saves 21.409% resident time versus Native and 9.196% versus Tuned Native (speedups 1.2724x and 1.1013x).

Resident sums exclude embedding, head, compilation, preparation and weight transfers. They are not whole-model wall-clock or offloaded-prefill latency. Bootstrap intervals in the raw summary resample per-layer mean differences, not independent full-model runtime repetitions.

Natural quality uses four B32/S1024 segments, BOS per independent sequence, and 130,944 scored next-token positions. All four arms passed unchanged limits: absolute NLL delta <=0.01, KL <=0.02, agreement >=97%. Strassen has absolute NLL delta 0.00007160, KL 0.00179820, top-1 agreement 98.3886%, and a relative perplexity increase of 0.007160%. Repetitive quality also passed. Natural logit relative L2 is 4.3586%, with max absolute difference 19.65625; these diagnostics remain reported alongside task metrics.

Agreement is with Native predictions, not true-label accuracy. B32 covers more held-out corpus positions than B8/B16, so absolute perplexity changes across batch sizes do not isolate a batch-size effect. Full selected profiles and metric summaries are retained here; candidate traces and offloaded measurements will follow in the complete batch archive.
