# Faster Strassen TPU: Codex Cloud handoff, October 5

The project is paused while the user reconsiders the LLM methodology. Preparing
or opening this project in Codex Cloud does **not** authorize resuming the queue,
allocating a TPU, downloading model checkpoints, or changing numerical gates.
No accelerator is needed to inspect, verify, or reconstruct the saved results.

This snapshot contains the committed research code and all **80 completed active
BF16 comparisons**, plus five historical FP32 comparisons stored separately.
It supersedes the 75-comparison October 4 publication for current status; older
snapshots remain immutable. The interrupted Qwen32 B4/S1024 workload is not a
completed comparison and must not be pooled with a future runtime.

## Start here

- Repository: `sarsid/strassen-tpu`.
- Branch: `codex/benchmarks-20260929`; do not start from the parent `public-preview`
  branch when continuing this study.
- This handoff: `benchmarks/llm/20261005_v001/`.
- Current research source: [code/](code/), with its own `src/`, `tools/`, `tests/`,
  `configs/`, `runtime/`, design notes and frozen parent implementations.
- Latest results: [timing-scope report](results/compute_scope_analysis_v001/README.md),
  [machine-readable measurements](results/compute_scope_analysis_v001/measurements.json),
  [raw active ledger](results/completed_entries.jsonl).
- Read [operator pause](code/docs/LLM_OPERATOR_PAUSE_20261004_v001.md),
  [four-model scope](code/docs/LLM_FOUR_MODEL_SCOPE_v001.md), and
  [project instructions](code/AGENTS.md) before any execution. Newest dated
  instructions supersede older historical instructions to run automatically.

All paths in this document are relative to this snapshot unless specified.
Historical absolute laptop and TPU paths in logs identify provenance; they are
not paths that can be used in the cloud container.

## Current state and research decisions

| Model | Completed workloads | Completed comparisons | Remaining status |
|---|---:|---:|---|
| Qwen3-8B | 10 / 10 | 50 | Complete |
| Qwen3-32B | 5 / 10 | 25 | Paused before completing B4/S1024 |
| Gemma4-31B | 0 / 10 | 0 | Official-checkpoint/runtime qualification gate closed |
| Qwen3-14B | 1 / 10 | 5 | B1/S512 retained; remaining work follows Gemma4 |

The intended order remains Qwen8, Qwen32, Gemma4, Qwen14. There are 200 active
comparisons, not the earlier 700. All five algorithms are retained: Default
Native, Tuned Native, Tuned Cubic, S1, S2. Use BF16 output for the current LLM
campaign, DEFAULT arithmetic and FP32 accumulation where the kernels require
it. BF16 output does not authorize changing accumulation or relaxing errors.
Depths 3/4, additional FP32 LLM runs, Mistral/Gemma3, and v5e follow-ups are deferred.

Ten workloads per model: B1 with S512/1024/2048/4096, B4 with S512/1024/2048,
and B8 with S512/1024/2048. Every completed workload has a whole five-arm group
with 15 matched confirmation rounds. Preserve fixed held-out inputs and all
quality gates. Tune to the target architecture; do not carry v5e settings to
v6e without architectural review. A general-purpose selector is not required;
LLM-specific workload tuning is the intended approach.

The queue/controller/watchdog and Codex monitor were stopped. The TPU was
released, with an empty allocation inventory verified on October 4. See
[release/pause receipt](pause/completion.json) and [archive verification](pause/archive-verification.json).
These are dated receipts, not a fresh query of the Google account.
[Continuation state](continuation.json) is a planning summary, not a runnable
controller state. Local automations, authentication and localhost services have
not been migrated.

Gemma4's last host-pool and resident-adapter drafts were unexecuted and are not
included in the committed code snapshot. Earlier synthetic qualifications do
not establish real-checkpoint accuracy or performance. Do not substitute the
Gemma3 adapter or bypass the qualification gate.

## What the results establish

Keep four different scopes separate: selected isolated fused-projection tuning
screens; individually warmed layer aggregates; layer compute during streaming;
and full streamed forward with recurring H2D plus all-token logits.

Warmed layer aggregates exclude H2D, embeddings and vocabulary head while
including attention, norms, residuals, epilogues and device-memory traffic.
They are not uninterrupted, fully resident model inference. Qwen8 B1/S512 and
Qwen14 B1/S512 predate the warmed pass; do not fill those gaps with another scope.
There are no full resident-model, last-token-only prefill or decode measurements.
The isolated projection screens cover Cubic/S1/S2 only; Native tuning measured
whole layers, so it is not an isolated Native GEMM baseline.

Qwen8 S1 beats Tuned Native in 3/9 available warmed workloads; S2 in 0/9.
Qwen32 S1 shows approximately 5–11% warmed compute reductions over Tuned Native
on its three larger completed workloads. At Qwen8 B8/S2048, S1 is 1.41x faster
than Default Native yet slower than Tuned Native. Do not conflate baselines.
All recorded predictive gates passed, but intermediate errors are nonzero and
can be appreciable. Quality uses held-out WikiText2 next-token scoring, not
downstream task accuracy. The detailed report preserves every error metric.

Current custom tuning selects isolated projection winners, then reuses
Native-selected whole-layer compiler options. It does not jointly retune the
assembled custom layer. This is a limitation to investigate, not a demonstrated
explanation of every loss. Parent results used another workload/protocol; see
[parent replication](parent_study/parent_llm_study_20261002_v001/README.md).

The completed 168-shape v6e MM study and historical v5e data remain in the parent
repository's `benchmarks/results/`. Keep their architecture and dtype groups
separate. Do not rerun them merely to set up this environment.

## Cloud setup and first checks

Use Python 3.12 for offline analysis. From this snapshot directory:

```sh
python3 verify.py
python3 -m venv /tmp/strassen-cloud-analysis
/tmp/strassen-cloud-analysis/bin/python -m pip install -r requirements-analysis.txt
/tmp/strassen-cloud-analysis/bin/python rebuild_report.py --output /tmp/strassen-report-rebuild
```

`verify.py` verifies file/evidence hashes and complete comparison groups.
`rebuild_report.py` materializes the sixteen event inputs from lossless evidence,
reconstructs the report, and checks numerical outputs against the saved report.
Use a fresh output directory on repeated runs. The analysis dependency pins are
not the TPU benchmark environment. No GPU/TPU package or model download is
required by these checks.

`restore.py` lists evidence groups or restores a selected group; for example:

```sh
python3 restore.py
python3 restore.py --group qwen3_32b-b4s512 --output /tmp/qwen32-b4s512
```

The cloud setup should initially support code review, analysis and CPU checks.
Do not add queue launch commands to installation/startup scripts. Before future
TPU work, obtain explicit resumption, establish authenticated remote runtime
access, adapt laptop-specific paths, verify the pinned runtime and target device,
and prove a small round trip with durable result collection. The existing
controller relies on local Colab tooling; cloud portability is not yet validated.
Do not copy credential files into Git, setup scripts, reports or prompts.

The old dashboard was `http://127.0.0.1:8789/` on the laptop. That URL does not
become reachable from the cloud. Maintain a live progress journal in the new
workspace using `code/status/PROGRESS_WORKFLOW_v002.md`; a cloud-accessible log
view can be configured separately. Keep logs and result artifacts durable.

Follow [current Codex Cloud setup instructions](https://learn.chatgpt.com/docs/cloud)
and [environment configuration](https://learn.chatgpt.com/docs/environments/cloud-environments).
Select this repository and branch, configure the analysis dependencies, verify
the snapshot, and publish the environment. Authentication and publishing the
environment must be completed in the user's account; this repository is the
handoff material, not proof that a cloud environment exists.

## First task prompt

> Continue the Faster Strassen TPU research project from
> `benchmarks/llm/20261005_v001/README.md` on branch
> `codex/benchmarks-20260929`. Read the operator pause and timing-scope report,
> verify the evidence, and reconstruct the offline report. Summarize the current
> 80/200 state and discuss the next LLM methodology with me. Keep all TPU
> allocation, checkpoint downloads, queues and monitors paused. Do not resume
> experiments until I explicitly ask. Preserve executed source versions and
> results, retain all five algorithms and errors, and maintain a live log.

## Preservation and publication

`checkpoint.json` identifies the source commit. `MANIFEST.json` hashes each
exported file; `evidence/index.json` maps canonical artifacts to deduplicated
gzip blobs with original filenames and hashes. Frozen measurement source bundles
remain separate from newer code; never retroactively attribute results to new code.

This export includes all completed cases and recorded compute-scope analysis.
The full local Git history, unrelated dirty work, local credentials, model
weights/caches, virtual environments and 139 partial B4/S1024 diagnostic files
remain on the original laptop. The partial diagnostics' verified archive is
recorded in the pause receipt. Nothing local was deleted. Do not claim a complete
byte-for-byte migration of the laptop, conversation, or active processes.

Commit source before meaningful validation, archive its execution and results,
and use scoped commits. Historical directories are evidence, not a live workspace
to edit in place. For new kernels or orchestration changes create new versions.
