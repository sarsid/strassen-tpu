# Unattended 700-comparison v6e queue

The user's October 2 late-evening request supersedes the initial single-workload
launch limit. Scope: 7 pinned models × 10 workloads × 5 algorithms × BF16/FP32.
The currently running Qwen3-8B B1/S512 workload is adopted without interruption.
Workload-major ordering gives each model a small qualification workload early.
No v5e, incremental decode or downstream-task evaluation is queued by this change.

The frozen protocol, kernels, tuning menus, quality gates and timing remain
unchanged. A new input loader extracts the official Gemma text oracle without
allocating the vision model. CPU generated-fixture qualification is separate
from the mandatory actual-checkpoint per-layer TPU qualification at execution.
Every workload still qualifies before tuning, freezes profiles, measures held-out
quality, and confirms all five algorithms in 15 randomized paired rounds.

A serial local supervisor persists 70 job records, corresponding to 700 unique
comparison IDs. Each output precision is a five-arm atomic checkpoint; incomplete
rounds never mix across devices. Successful checkpoints are downloaded, hashed
and merged idempotently. Finished groups are skipped on retry. Failed accuracy
gates remain measured outcomes, never reclassified as speed/quality successes.
Three total workload attempts are allowed; then that workload is recorded failed
and independent jobs continue. Missing Gemma credentials block Gemma jobs only.
Colab allocation failures use 2-minute to 1-hour backoff without marking any
measurements complete. Unexpected existing allocations block another allocation.
An unreleased owned TPU blocks queue advancement until release succeeds.

New case time budgets: ten hours for measurement, twelve hours for its controller.
The current first workload keeps its original frozen limits. A dead local
controller is reattached to the same run up to twice. The local watchdog restarts
a dead supervisor; its caffeinate process prevents idle system sleep. The laptop
must remain powered, awake (lid open) and connected for reliable orchestration
and result downloads. Colab credits, provider availability and session expiry
can still interrupt the sweep. This queue is not a promise of overnight completion.

Each case uses a committed source archive plus an immutable job specification,
a fresh v6e, pinned dependency versions, private checkpoint storage, and verified
release. The existing authorized HF credential is staged only for official Gemma,
outside artifacts and deleted with the runtime. Results and failure evidence are
committed to the dedicated campaign folder; this repository has no configured
Git remote, so these scoped commits are local unless explicitly published later.
The dashboard distinguishes measured comparisons from queued, failed and blocked
workloads. Live state is in queue_v001/state.json and remains resumable after a
local process restart. Future source changes require a new version and evidence.
