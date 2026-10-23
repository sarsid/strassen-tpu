# Revised research sequence and execution boundary

User decision: September 29, 2026 evening (America/New_York).
This supersedes the execution order in `RESEARCH_SEQUENCE_v001.md` and the
automatic v6e-to-v5e transition in the September 27 architecture/recovery plans.
Those documents and all executed measurement sources remain historical evidence.

## Current authorization: finish v6e shapes, archive, release, stop

Complete the existing 168 development geometries on v6e using the frozen
measurement contract. Keep FP32 and BF16 output separate; preserve the default
Native, tuned Native, independently tuned cubic, S1 and S2 comparisons, errors,
failures, timing samples, settings, source hashes and allocation identities.
Continue the running controller; no kernels, tuning menus, seeds or gates change.

Verify all whole comparisons and finalize the report in the dedicated v6e
results directory. Release the owned TPU before CPU-only final reporting, as
already required by the recovery policy. If reporting fails, retry reporting
without allocating a new TPU. Mark completion only after the final evidence
is verified and committed. Then stop the supervisor and delete the scheduled
monitor when it observes verified completion and runtime release.

There is no automatic v5e transition and no automatic LLM launch. Previous
follow-up requests, including old probe repeats, are removed from the execution
queue. Unrelated projects' tasks are outside this scope. Preserve existing v5e
datasets and prepared source/configuration; they are not newly authorized work.

## Proposed next phase: v6e LLM fusion

Discuss and freeze the application experiment after reviewing the completed
shape study. The purpose is to establish whether measured matrix speedups
survive integration into realistic model workloads, and what numerical/model
quality cost accompanies them. Do not queue or launch this phase automatically.

Recommended design decisions for that discussion:

1. Compare the standard application baseline with equally fused/tuned Native,
   cubic and Strassen variants. Separate gains from fusion itself from gains
   attributable to the matrix algorithm.
2. Tune the actual fused operation contracts and fixed model/workload profiles.
   Keep prefill and decode results separate. Record kernel, resident-layer and
   complete-model timing, memory and the fraction of work actually accelerated.
3. Use real weights and activations. Measure layer/logit errors, distribution
   changes, perplexity and appropriate prediction/task metrics. Declare useful
   speedup and acceptable quality changes before the final evaluation inputs.
4. Freeze implementations and selections before final evaluation. Archive wins,
   losses and inconclusive outcomes alike; a preference to pursue positive
   applications is not permission to discard negative evidence.
5. Preserve the completed standalone v6e matrix dataset unchanged. If fusion
   work leads to a different matrix implementation, version that separately.

## Later decision: replicate both parts on v5e

After reviewing the v6e LLM results, decide whether to replicate both the shape
study and fused application workloads on v5e. Use the agreed algorithm families,
precision contracts, inputs and evaluation criteria while tuning for v5e's own
hardware/software constraints. Do not copy v6e tile sizes or memory limits by
default. Distinguish architecture effects from compiler and implementation
differences. This phase also requires fresh instructions and is not queued.

Only after that evidence is available should the research narrative combine
standalone matrix behavior, application carryover, numerical cost and the
architecture comparison.
