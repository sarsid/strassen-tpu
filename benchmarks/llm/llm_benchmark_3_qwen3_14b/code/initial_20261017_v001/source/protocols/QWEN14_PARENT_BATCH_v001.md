# LLM Benchmark 3: Qwen3-14B

Authorized October 17, 2026: use the same setup as the completed Qwen32 and
Gemma experiments. Only this model is queued. Preserve both previous benchmarks.

## Model and environment

Use the pinned parent's existing 14b registry entry, without editing parent
sources: Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18. Its official
configuration is archived under plans/qwen14_parent_batch_20261017_v001.
There are 40 layers, hidden width 5120, intermediate width 17408, 40 query
heads, eight KV heads and head width 128. Verify registry/config/plan agreement.
Parent commit: 95be1fb088656a89813b04492e1d77c66b36ccf9.

One v6e; JAX/jaxlib 0.11.2 and libtpu 0.0.48 for every batch, with the same
Qwen tokenizer/data dependencies as the completed Qwen32 B64 run. Archive
installation and freeze records. BF16 inputs/output, parent FP32 accumulation,
DEFAULT precision, original rounding and fused operations. No Gemma adapter,
S2, new fusion ablation, full-resident serving experiment, or v5e run is added.

## Workloads and tuning

S=1024; B=1,4,8,16,32,64. Execution order 8,16,32,64,4,1. Four arms:
Native (default XLA), Tuned Native (scoped VMEM), Tuned Cubic, Tuned Strassen
(parent S1). Main paper comparison remains Tuned Native versus Strassen.

Fresh tuning for every batch. Reuse the Qwen32 v004 candidate axes, analytic
memory filtering, 2 warmups/5 samples and mean selection. Native VMEM choices
are default,32,48,64,96,112,128 MiB. Gate/up and O/down tiles are independently
selected for Cubic and Strassen; site constraints use Qwen14 dimensions.
Q/K keeps the parent published fused tile (BM clipped for B1); V stays native.
The initial O tile uses K=5120, matching Qwen14 rather than Qwen32's 8192.
Gate/up and O/down tile screening uses seeded synthetic operands; confirmation
uses official weights and propagated real activations. Do not label Q/K tuned.

Freeze each profile, recording model/revision/workload and its SHA-256 before
qualification, confirmation or held-out quality. Reject another model's profile.
Keep every candidate, compile failure, rejection and timing. Do not retune on
confirmation/quality results. Kernels and timing loops are unchanged from the
completed current-stack Qwen32 implementation. Full-batch segmented diagnostic
capture is retained; no microbatching or silent batch reduction.

## Measurements and error gates

Resident projections and block0: 3 warmups/20 samples. Each of all 40 resident
layers: 2 warmups/5 samples with rotated/reversed arm order and each arm's own
propagated state. Sum warmed layer means, excluding embeddings/head, compilation,
weight preparation/transfers. This is not uninterrupted whole-model latency.
Report raw milliseconds, time saved and speedups versus both Native baselines.
Preserve parent layer bootstrap and its limited interpretation.

Secondary cached-disk-offloaded prefill: 1 warmup and 3 full passes per arm,
rotating arm order. Includes common cached weight reads, packing, recurring H2D,
embedding, all layers and last-position logits. No network downloads or compile
inside timing. Common loader includes Strassen preparation for every arm;
this is not an independently optimized native serving baseline. No decode.

Use parent repetitive prompts and four contiguous WikiText2 test segments with
pinned data/tokenizer and token hashes. No token repetition to fill corpus.
Keep finite, relative L2, max error, NLL/perplexity, KL and top1 agreement.
Native reconstruction relative L2 <=.01. Predictive gates unchanged:
repetitive absolute NLL <=.01, KL <=.005, agreement >=.99;
natural absolute NLL <=.01, KL <=.02, agreement >=.97.
All tunable arms must pass; failed gates remain visible. Agreement is against
Native, not ground-truth accuracy. As with Qwen32, this study follows the parent
Qwen computation and does not add a new cross-framework model-equivalence claim.

## Operations and preservation

Use new versioned files and a new immutable run directory. Commit before
validation, archive local checks, freeze source before allocation. Maintain the
existing dashboard at port8790 with explicit batch, stage, arm, tile, layer and
quality/prefill test progress. Status distinguishes allocation from computation.

Use the tested retained-TPU controller: one tracked allocation, persistent
capacity retries, no duplicate launch after a timeout, incremental profile and
summary checkpoints, per-batch hash-verified exports and scoped commits.
One fresh-process retry per infrastructure/stage failure; never retry numerical
failures merely for a pass. Other batches continue after a failed batch. Retain
the owned TPU for troubleshooting on errors; release only after successful
verified archival or explicit user request. Keep idle sleep prevented during
execution, while acknowledging physical lid closure/network loss can interrupt
local monitoring. Publish new code/results in llm_benchmark_3_qwen3_14b.
