# Project progress

While working in this project, follow `status/PROGRESS_WORKFLOW_v002.md` to keep
the user's live progress view current with concise actions, results, blockers,
and next steps. Append status updates at the start, at material milestones,
and before ending a turn. Report actual execution counts; do not substitute
preparation or an unrelated campaign for the requested experiment's progress.

Preserve executed source versions and experiment results. Archive validation
and experiment runs with scoped commits that exclude unrelated active work.

# Active experiment scope

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
