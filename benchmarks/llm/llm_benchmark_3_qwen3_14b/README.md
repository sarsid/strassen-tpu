# LLM Benchmark 3 (Qwen3-14B)

**Completed: all six batches and all four algorithms, with every registered quality gate passed.**
The TPU has been released and no further experiment is queued.

[Read the complete report](summaries/completed_20261017_v001/README.md) for timings,
all offloaded samples, quality/error metrics, tuning tiles and measurement limits.

- Batches 1, 4, 8, 16, 32, 64; sequence length 1024; one v6e.
- Native, Tuned Native, Tuned Cubic, Tuned Strassen S1; BF16 output.
- Fresh per-batch tuning; JAX/jaxlib 0.11.2 and libtpu 0.0.48 throughout.
- Strassen used **5.86% less resident-layer time at B1 and 23.08–29.83% less at B4–B64** versus Tuned Native.
- The resident metric sums 40 independently warmed decoder layers; it excludes weight transfers and is not uninterrupted whole-model latency.
- Transfer-inclusive cached-disk-offloaded prefill mean savings versus Tuned Native were **0.27–1.52%**, with only three measured passes per arm and visible variation.
- Maximum natural-text perplexity increase for Strassen versus Native: **0.1191%**; top-1 agreement was **98.64–98.75%**. Full numerical errors are retained.

See the [registered protocol](PROTOCOL.md), [frozen code](code/initial_20261017_v001/source),
[raw results](results/20261017_v001), [machine-readable report](summaries/completed_20261017_v001/summary.json),
and [completion/release evidence and full environment](operations/completed_20261017_v001).
Run identity: `20261017-qwen3-14b-v001`. Remote execution: October 17, 2026,
17:53:18–22:03:15 UTC. All 30 stages succeeded on attempt 0.

The [initial launch index](operations/initial_index_20261017_v001), all incremental
snapshots and both local validation attempts remain preserved. Prior Qwen32 and
Gemma benchmark folders are unchanged. Only code and numerical evidence are
published; no credentials, checkpoint weights or activation tensors are included.
