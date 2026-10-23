# Model order and benchmark overhead, October 3

The user requested finishing each model before moving to the next. The active
Qwen3-14B B1/S512 workload finishes first, explicitly confirmed by the user.
The remaining queue then follows this order, retaining the existing workload
order within each model and skipping completed precision groups:

1. Qwen3-8B
2. Qwen3-32B
3. Gemma3-12B
4. Gemma3-27B
5. Mistral Small 24B
6. Qwen3-14B
7. Mistral-7B

All 70 workload identities and 700 planned comparisons remain. Historical
attempts, measured results, frozen sources and the active controller are
preserved. The ordering change does not change kernels, tuning budgets,
precision, timing samples or quality thresholds. This supersedes the original
workload-major scheduling in LLM_QUEUE_v001.md. The queue reads job order from
its persisted state on each iteration; no measurement source replacement is
needed. The local watchdog starts a fresh supervisor after the scheduling edit;
the existing controller and TPU continue without interruption.

## Why the current campaign takes hours

Measured Qwen3-8B B1/S512 took 207.32 minutes including setup, archive and release.
Its two confirmation groups each contain 75 samples. Mean event spacing was
72.19 seconds for BF16 and 68.66 seconds for FP32. In the current Qwen3-14B
workload the first 24 confirmation events averaged 85.28 seconds between events,
including 76.25 seconds of host preparation per event. A forward is about 2.3
seconds, and each event follows three warmups plus one measured forward.

`llm_host_pool_v001.ModelPool.activate` evicts the previous algorithm's prepared
host matrices. On every return to it, `Checkpoint.layer` reads and transposes
every layer again, joins gate/up, and `pack_host` constructs the projection
layouts. The LM head is transposed again too. The policy bounds memory to one
prepared model, but 15 randomized rounds over five algorithms and two stores
repeat this expensive work roughly 150 times. This is avoidable harness work,
not a cost intrinsic to Strassen. Preparation is outside reported inference
latency; serialized recurring H2D is inside it.

The frozen parent script at commit 95be1fb, lines 509 onward in
`third_party/strassen_tpu_public_95be1fb_v001/experiments/qwen3/benchmark_qwen3_32b_streamed_inference.py`,
loads one layer's parameters, performs two warmups and five repeats for three
arms while they are resident, then advances the hidden states and loads the
next layer. It scores the resulting final hidden states afterward. Its headline
is the sum of resident layer compute, explicitly excluding weight transfers,
embedding and final scoring. Our current five-arm/two-store full-forward study
asks a broader timing question and adds workload-specific tuning and official
checkpoint qualification. Matching the parent timing loop alone would change
the meaning of the end-to-end results.

The archived parent reproduction makes the wall-time difference concrete:
the current-stack `current-streamed` stage took 200.11 seconds and the separate
`current-quality` stage took 195.59 seconds, excluding checkpoint/environment
setup. Those stages have the narrower parent scope above. The original-stack
streamed stage took 996.62 seconds; therefore even parent wall time depends on
the software environment. Evidence is in
`results/v6e/parent_llm_replication_20261002_v001/summary.json`.

## Proposed optimizations, not yet deployed

First cache canonical transposed checkpoint tensors and deduplicate packed
layouts by checkpoint revision, layer, projection and the exact layout metadata.
Keep the common LM head once. Share byte-identical buffers across algorithms and
output tracks when their layout is identical. A bounded memory cache with a
private disk-backed cache can handle models that cannot retain all layouts in
RAM. Page-in must finish before warmup and timing. Raw weights and caches remain
outside result archives. This preserves the existing randomized 15-round timing
and all accuracy gates. Validate bitwise identity to existing packing and then
measure real host preparation, peak RAM/disk and inference timing on a pilot.

At the current Qwen3-14B cadence, confirmation preparation alone is approximately
150 * 76 seconds, or 190 minutes. Reusing layouts could remove much of that cost;
it is not yet a measured speedup. Cache misses, disk throughput and larger-token
quality/tuning costs prevent a firm revised duration estimate today.

Second use a parent-style layer-resident measurement pass for the compute
breakdown, keeping a separately labeled complete-forward transfer-inclusive
confirmation. It is useful for identifying kernel gains but cannot replace a
full-forward sample or be reported as serving latency. Changing sample counts
or using screening/confirmation subsets is a protocol change and should be
explicit; no such changes have been made here.

Finally, retaining one allocated runtime and checkpoint across a model's ten
workloads could reuse downloads, qualification and compiled shapes when valid.
This requires stronger checkpoint/recovery boundaries and careful shape-keyed
cache invalidation. It is secondary to eliminating repeated host transposes.
