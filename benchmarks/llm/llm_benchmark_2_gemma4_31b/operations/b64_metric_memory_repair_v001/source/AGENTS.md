# Project progress

While working in this project, follow `status/PROGRESS_WORKFLOW_v002.md` to keep
the user's live progress view current with concise actions, results, blockers,
and next steps. Append status updates at the start, at material milestones,
and before ending a turn. Report actual execution counts; do not substitute
preparation or an unrelated campaign for the requested experiment's progress.

Preserve executed source versions and experiment results. Archive validation
and experiment runs with scoped commits that exclude unrelated active work.

# Active experiment scope

October 16 B64 diagnostic-memory recovery: v003 B64 tuned and qualified, but
both confirmation attempts exhausted HBM while constructing an untimed 5.25 GiB
FP32 projection-error difference. Prepare stage v004 bounded error diagnostics
under protocols/GEMMA4_B64_METRIC_MEMORY_REPAIR_v001.md. Let the existing v003
worker finish B4/B1, retaining its owned TPU, then automatically retry B64 only
on that endpoint with its saved profile. No retuning, duplicate worker,
reallocation, timing/kernel change or relaxed quality gate is authorized.

October 16 approved validation split supersedes the pending decision below.
The user approved the parent-style FP32 formula qualification and unchanged
same-TPU BF16 algorithm gates, plus the natural-text BOS correction. Use
protocols/GEMMA4_PARENT_VALIDATION_v002.md and versioned v003 stage/worker,
v004 controller/starter. Resume only this Gemma four-arm batch sweep. Retain
on errors. Colab stopped listing the prior retained endpoint at 14:33:47 UTC;
our controller did not release it. Reconcile allocations before replacement.

October 16 numerical diagnosis completed: all 60-layer CPU/Torch/TPU traces and
189 matched-operation records are preserved in runs/20261016-gemma4-numerics-v001.
The independent real-weight FP32 control in runs/20261016-gemma4-fp32-semantics-v001
agrees within 4.94e-7 relative L2. No additional formula mismatch was identified.
The remaining BF16 gate failures involve accumulated cross-framework drift and
one exact-tie-sensitive prediction. The user has been asked to choose the proposed
parent-style validation split in protocols/GEMMA4_PARENT_VALIDATION_PROPOSAL_v001.md.
Until they answer, keep the owned TPU retained, preserve the current failed gates,
and do not repeat the completed diagnosis, start tuning, or change stopping rules.
The natural-text BOS correction remains necessary before future quality runs.

October 16 retained-TPU troubleshooting supersedes automatic release on failure:
the user instructed retaining the TPU when an error occurs and fixing the issue
on that machine. Preserve failed attempts, reconcile the same owned endpoint,
and repair/retry without releasing or allocating a replacement. The active
numerical diagnostic also retains the TPU after collecting evidence so repairs
can follow. Release after successful completion of the requested work or explicit
user instruction. See protocols/GEMMA4_RETAIN_ON_ERROR_v001.md.

October 16 numerical repair: the user explicitly authorized fixing Gemma4's
error mismatch and restarting. Use versioned BF16-score attention, unchanged
accuracy thresholds, historical-path diagnostics, and a new immutable run.
Validate locally before allocation, then pass the full official-checkpoint
gate before tuning. See protocols/GEMMA4_ATTENTION_REPAIR_v002.md.

October 16 Gemma4-31B supersedes the completed Qwen B64 scope. The user
authorized LLM Benchmark 2 and a clearer live dashboard. Run the pinned dense
Gemma4 text checkpoint with Native, Tuned Native, Tuned Cubic and parent S1,
BF16 output, batches 1/4/8/16/32/64 at sequence 1024. Qualify official-weight
model semantics before timing; tune local/global layer types separately. Keep
older queues inactive, preserve every attempt, commit results and release the
owned TPU after completion. See protocols/GEMMA4_PARENT_BATCH_v001.md.

October 16 B64 tuning reuse supersedes the repeat-tuning decision below. The
user requested reusing completed tuning; two saved frozen_profile events
restore the exact original profile SHA-256. Use controller v007, starter v008,
worker v005 and the unchanged measurement stage v004 to skip tuning and run
qualification, confirmation, quality and prefill. Keep v6e/compiler versions,
idle-sleep prevention, persistent capacity wait, stage checkpoints and all
scientific settings. See protocols/QWEN32_B64_PROFILE_REUSE_v007.md.

October 16 UTC runtime-loss recovery continues the active B64-only home
resumption. macOS idle sleep stopped keep-alives after v005 tuning and
qualification passed; Colab later confirmed the owned endpoint was absent.
Preserve partial results and recover with controller v006, stage checkpoints,
starter v007 --keep-waiting and a process-scoped caffeinate -i assertion.
Measurement stage/worker v004 and all scientific settings remain unchanged.
See protocols/QWEN32_B64_SLEEP_RECOVERY_v006.md.

October 15 home resumption: the user explicitly requested restarting and
running the B64 experiment and waiting until a v6e allocation is available.
The travel pause below is lifted for this B64-only restart. Use validated
controller v005 / starter v006 and unchanged measurement stage/worker v004,
retune from scratch, maintain the live log, archive and commit results, and
release the owned TPU afterward. Continue capacity retries until allocation;
do not reactivate any other batch/model/architecture queue.

October 15 travel pause supersedes automatic B64 recovery below: the user
explicitly said not to restart and will resume after reaching home. The v004
TPU is already released. Finish local preparation/validation and preservation
only; do not allocate a TPU, launch a worker, or schedule an automatic restart
until the user explicitly resumes. Keep the B64 recovery ready but paused.

October 15 monitoring recovery: current-stack B64 v004 was interrupted by an
unhandled local polling timeout; its controller released the owned TPU. Resume
the still-authorized B64 task with controller v005 after fault-injection
validation. Keep measurement stage/remote worker v004 and all scientific
settings unchanged, retune from scratch, and preserve the failed run and its
partial tuning evidence. See protocols/QWEN32_B64_CONTROLLER_RECOVERY_v005.md.

October 15 B64 compiler follow-up: v003 full-size segmented capture succeeded,
but unchanged Native block compilation then stack-overflowed twice on the
original parent stack. Under the user's repair-and-run authorization, v004
keeps parent kernels/math and runs B64 on JAX/jaxlib 0.11.2 and libtpu 0.0.48,
retuning every tunable arm on that runtime before qualification/confirmation.
Store this as a separate software-stack extension; never pool it with the
parent-stack B1-B32 sweep. Scope remains B64 only. See
protocols/QWEN32_B64_CURRENT_STACK_v004.md.

October 15 B64 repair and run: the user explicitly authorized fixing the
remaining Qwen3-32B B64 failure and running that experiment. Prepare versioned
segmented capture on the pinned parent stack, validate locally and at full
B64/S1024 size, then run the four frozen-profile arms through confirmation,
quality and offloaded prefill. Preserve all historical sources and failures,
keep the live log current, archive results and release the owned v6e afterward.
No other batch/model or retuning campaign is included. See
protocols/QWEN32_B64_RETRY_v003.md.

October 15 office resumption: the user explicitly authorized starting the
prepared B32/B64 retry. The preparation-only pause below is lifted for
`runs/20261015-qwen32-large-batch-v002` only. Use the validated frozen v002
measurement sources and saved B32/B64 profiles, run qualification before
confirmation/quality/prefill, preserve new results separately, keep the live
log current, and release the owned v6e afterward. No other batch, model,
retuning sweep or v5e experiment is queued by this resumption.

October 15 large-batch repair is preparation only until explicit resumption.
The six-batch Qwen32 sweep ended with B1/B4/B8/B16 complete and B32/B64
confirmation memory failures; its TPU was released. The user authorized fixing
the B32/B64 harness and local validation, and will ask to run after reaching
the office. Do not allocate a TPU, download weights, launch a remote worker or
reactivate an allocation supervisor yet. Prepare the versioned retry described
in protocols/QWEN32_LARGE_BATCH_RETRY_v002.md, retaining all earlier evidence.

October 14 Qwen-only study supersedes the preparation-only scope below. The user
selected Qwen3-32B first, larger batch sizes, four algorithms (Native, Tuned
Native, Tuned Cubic, Tuned Strassen), the supplied measurement-scope guidance,
a live log, and immutable committed results. Prepare and run only this study
under protocols/QWEN32_PARENT_BATCH_v001.md; keep all older queues paused.
Other LLMs follow only after discussion of this study's outcome.

October 13 follow-up: prepare a parent-based study for Gemma4-31B dense text
and the requested Qwen4-32B/14B identities. Follow the pinned original parent
for design choices. Qwen4 public checkpoints are unverified; Qwen3 metadata is
an unselected alternative pending clarification. Batch and sequence are unset.
This is preparation only: do not allocate, download weights, tune, run models,
or reactivate any queue. The parent arms/method replace the old custom campaign
for this draft. See plans/parent_llm_models_20261013_v001/README.md.

October 13 parent restart completed (02:52 UTC October 14): all eight parent
measurement/tuning stages finished, the complete archive was downloaded and
hash-verified, and the owned v6e was released with absence verified. The
published profile reproduced the headline: 1.2106x block and 1.2937x summed
resident-layer speedup; all its scientific gates passed. Retuning achieved
1.2969x layer-sum speedup and passed quality, but failed the additional block
scheduling comparison against standard Strassen. Preserve that failed sub-gate.
See results/v6e/parent_exact_restart_20261012_v001/completed_v001/README.md.
No further TPU allocation, tuning repeat, larger LLM campaign or v5e run is
queued; discuss the larger experiment design with the user next.

October 12 parent-only restart: the user authorized a fresh reproduction of
the parent's LLM benchmark and its tuning before redesigning the larger study.
Run the pinned public-preview Qwen3-32B B8/S1024 v6e benchmark on its original
software stack, preserving the parent computation and measurement contracts.
Keep published-profile reproduction and fresh parent-tuner results separate.
This lifts the pause only for this bounded parent replication, not the broad
LLM queue, v5e, or our custom-kernel sweep. Release the owned TPU afterward.
See protocols/PARENT_LLM_RESTART_20261012_v001.md.

October 4 operator pause supersedes automatic continuation below. The user
requested pausing the LLM campaign and releasing its TPU to reconsider the
strategy in the morning. Leave the queue, watchdog and monitor paused until
explicit user resumption. Preserve all 80 completed BF16 comparisons and the
interrupted Qwen3-32B B4/S1024 diagnostics; never pool partial rounds with a
future runtime. See the queue's `operator_pause_20261004_v001` receipts.

October 3 four-model revision supersedes the seven-model scope below. Keep
Qwen3-8B, then Qwen3-32B, Gemma4-31B, then Qwen3-14B: 40 workloads and
200 BF16 comparisons. Defer both Gemma3 models and both Mistral models; preserve
all earlier evidence. Keep the active Qwen8 remote session uninterrupted. Use
queue v008 with an explicit qualification barrier before Gemma4; never run it
through the Gemma3 adapter or silently skip it. See
docs/LLM_FOUR_MODEL_SCOPE_v001.md.

October 3 BF16 narrowing supersedes the two-store LLM requirements below:
the user authorized BF16 output only for the remaining LLM campaign. Preserve
all historical FP32 data and the original 700-entry plan. The active target is
350 BF16 comparisons, with five algorithms, unchanged timing rounds and error
gates. Stop the active legacy FP32 phase after preserving its completed BF16
group and partial diagnostics; archive and release normally, then continue the
same model order using queue v007 and the qualified cached model-session code.
See docs/LLM_BF16_SCOPE_v001.md. Do not interpret this as changing accumulation
precision or authorizing weaker accuracy thresholds.

October 3 cached sessions: the user approved all three runtime optimizations in
docs/LLM_CACHED_MODEL_SESSIONS_v001.md. Preserve the current Qwen3-14B workload;
qualify new versioned cache/resident/session code and activate it for subsequent
workloads. Keep all 700 identities, all five algorithms, both stores, 15 full
confirmation rounds and all error gates. Old sources/results remain preserved.

October 3 model order: finish the current Qwen3-14B B1/S512 workload, then run
all remaining workloads model-by-model: Qwen3-8B, Qwen3-32B, Gemma3-12B,
Gemma3-27B, Mistral Small 24B, Qwen3-14B, Mistral-7B. The user explicitly kept
the latter two after the five priority models. Preserve all 700 identities and
completed groups. This supersedes workload-major ordering. See
docs/LLM_QUEUE_ORDER_AND_OVERHEAD_v001.md. Optimization proposals in that document
are not deployed measurement changes.

October 2 late evening: the user explicitly requested that the full 700-comparison
v6e LLM sweep be queued unattended. Adopt the running first workload and continue
all seven models × ten prefill workloads × five algorithms × two stores. Save
whole five-arm groups, retry bounded failures, preserve qualification gates,
archive and commit results, and release each owned TPU. No v5e, decode or
downstream task expansion is implied by this queue. See docs/LLM_QUEUE_v001.md.


October 2 evening: the user authorized starting the broad v6e LLM campaign,
maintaining a live log and committing code/results. This supersedes the earlier
no-new-LLM boundary. Initial suite: Qwen3 8B/14B/32B, Mistral 7B/24B, Gemma3
12B/27B text; five algorithms, separate BF16/FP32 stores, DEFAULT precision.
Validate model semantics before tuning and held-out evaluation. Retain streaming
with transfer-inclusive timing and resident compute as separate metrics. No v5e
work is queued. See docs/LLM_CAMPAIGN_v001.md.

October 2 parent S1 adoption: the user authorized making the reproduced parent
S1 implementation the default matched Qwen3-32B v6e path. Use the new v002
adapter/evaluation entry points and the adopted profile; preserve historical
v001 comparisons and explicitly label S2/FP32 as extensions. See
`docs/PARENT_S1_ADOPTION_v001.md`. This integration and evidence audit does not
queue a TPU allocation or a Gemma experiment.

October 2 parent replication is now complete. The parent headline Qwen3-32B
result was reproduced on its original and current stacks; our tuned S1/S2
BF16/FP32 policies and the declared timing-repeat/down-budget follow-up were
measured. See `results/v6e/parent_llm_study_20261002_v001/README.md`. All evidence
is downloaded; the owned v6e was released with verified absence. No additional
TPU, v5e, LLM, tuning or scheduled follow-up work is queued. Historical scope
and failure evidence below remain preserved.

October 2 parent LLM replication: the user now explicitly authorized replicating
the parent LLM experiment and testing whether our implementation matches or
improves its results. Start with the headline Qwen3-32B v6e workload (batch 8,
sequence 1024), preserve exact parent reproduction separately from our tuning,
and retain speed/error evidence and a live log. This supersedes the previous
no-new-LLM-work boundary for this task. No v5e replication is queued.

October 2 fusion adaptation: after completion and publication of all 168 v6e
shapes, the user authorized inspecting the parent's public-preview fused LLM
kernels and making them compatible with this repository. Implement and qualify
that integration in new versioned modules. Preserve the completed study and
its sources. This does not queue v5e replication or a broad LLM campaign.

Per the user's 2026-09-23 instruction, future MM tuning and LLM comparisons use
Native and Strassen depths 1 and 2 only. Exclude depths 3 and 4 unless the user
explicitly requests them again. Preserve their historical sources and results.

Per the user's 2026-09-24 instruction, future MM studies must treat FP32 output
and BF16 output as separate comparisons. Match input, accumulation and output
precision across Native, S1 and S2 within each comparison. Include any final
output conversion in timing and numerical error; do not pool results across
output dtypes. Broader BF16 comparisons follow the current full-contraction
probe, whose limited parent-kernel BF16 reproduction remains a separate group.

Per the user's September 29, 2026 evening instruction, automatic execution now
ends after the current 168-shape v6e FP32/BF16 study: verify and commit the final
results, release the owned TPU, stop the supervisor, and remove its scheduled
monitor. All prior instructions to automatically proceed to v5e or any other
follow-up experiment are superseded. Do not automatically launch LLM fusion,
old probe repeats, or another study. See `docs/RESEARCH_SEQUENCE_v002.md`.
LLM fusion on v6e and later replication of both shapes and fusion on v5e are
proposed subsequent phases requiring fresh user instructions. Preserve all
historical code, plans, results and failed attempts; cancellation is not deletion
of scientific evidence.

September 30 travel pause: finish the already-running arch-130 comparison and
its dedicated exports, then park the existing local controller and supervisor.
The heartbeat is PAUSED. Do not restart or resume either process until the user
explicitly requests resumption. Inspect the supervisor folder's
`operator-pause-request.json` and its evidence before acting. The existing TPU
is retained for this short pause. Preserve the frozen controller/measurement
sources; do not launch a duplicate worker. After authorized resumption, the
v6e-only completion boundary above still applies.

September 30 office resumption: the user explicitly requested restart after
returning. The travel pause is lifted. The saved TPU endpoint was no longer
allocated at the pre-resume check, so retire only the parked obsolete local
controller and let the existing v6e-only supervisor reconcile saved comparisons
and obtain a replacement v6e. Re-enable the same recovery heartbeat. Preserve
the pause evidence and resume only unfinished whole comparisons; no v5e or LLM
experiments are authorized by this resumption.

September 30 evening pause supersedes that resumption: the user requested
stopping after arch-136 and not starting 137. At inspection, arch-136 was already
saved and arch-137-screen had started. Stop that incomplete phase, preserve its
partial evidence, release the owned TPU, and leave orchestration and the
heartbeat paused until explicit user resumption. Do not count partial 137 as a
completed comparison or reuse its partial timings on a new runtime.

September 30 home resumption: the user explicitly requested restart after
returning home. The evening pause is lifted. Retire the obsolete stopped cohort
controller, resume the existing v6e-only supervisor, and re-enable the same
recovery heartbeat. Verify the released runtime is absent and continue with a
new v6e allocation from whole comparison arch-137. Preserve the completed 136
comparisons and the interrupted 137 evidence. The scope still ends after the
168 v6e shapes; no subsequent v5e or LLM work is queued.
