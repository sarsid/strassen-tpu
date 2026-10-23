# Project progress

While working in this project, follow `status/PROGRESS_WORKFLOW_v002.md` to keep
the user's live progress view current with concise actions, results, blockers,
and next steps. Append status updates at the start, at material milestones,
and before ending a turn. Report actual execution counts; do not substitute
preparation or an unrelated campaign for the requested experiment's progress.

Preserve executed source versions and experiment results. Archive validation
and experiment runs with scoped commits that exclude unrelated active work.

# Active experiment scope

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
