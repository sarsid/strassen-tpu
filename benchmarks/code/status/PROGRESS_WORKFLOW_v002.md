# Live progress reporting

The user wants to see concrete progress while work is underway. The current
view is `http://127.0.0.1:8766`; the older results browser remains on port 8765.

At the start of a work session, after a material action/decision/result, and
before ending the turn, append a concise factual update using
`tools/record_progress_v001.py`. Never edit old journal entries. For example:

```sh
python tools/record_progress_v001.py --kind activity --state working \
  --title 'Preparing the three-model shape experiment' \
  --detail 'Selecting matrix shapes and recording the comparison protocol.' \
  --next-step 'Freeze the protocol before launching measurements.'
```

Use activity id `current`. Use model ids `qwen`, `mistral`, `gemma` for the
new larger-shape mini experiment, and agent ids `/root`, `/root/protocol`,
`/root/kernels`, `/root/runtime` for those agents' reported assignments.
Supply actual completed/total counts only when a denominator is known.
Preparation, previous experiments, or a completed dashboard do not count as
starting the new experiment. Record actual model names using `--title`.

Add evidence paths with `--evidence`. Record plans, actions, observable findings,
decisions, blockers and next steps; do not log private deliberation. Mark an
activity idle or complete when it ends. The site does not run, queue or monitor
assistant turns itself. Its refresh only reads local evidence.

The grid count is read automatically from the separate cohort's `group_complete`
events. Partial journal lines are ignored until complete. An old update is shown
as stale, not as proof the job is running or stopped. Agent cards show the last
reported assignment, not an independently verified running process.

The initial Gemma panel reads the preserved v001 full-checkpoint comparison.
New qualifications must be added explicitly as new source-linked evidence;
do not quietly replace the failed historical result.

To serve the dashboard from the working tree:

```sh
python status/server_v009.py --port 8766
```

For immutable deployment, run the same file from a committed execution snapshot
with `STRASSEN_PROJECT_ROOT` set to this project directory. Bind to loopback only.
Keep experiment sources/results and the other campaign's dirty files intact.
Commit code before validation and archive meaningful validation runs with the
scoped archive runner. Commit this workflow's append-only journal separately
from the older shared `status/events.jsonl`.

Current version: server v009 reuses the validated v008 data feed and serves
responsive UI v002. The v001 page and every failed browser check remain archived.
