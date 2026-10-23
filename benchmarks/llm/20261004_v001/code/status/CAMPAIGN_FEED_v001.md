# Measured campaign progress feed

Serve the new read-only page on **8767**, leaving 8766 unchanged:

```sh
STRASSEN_PROJECT_ROOT='/absolute/path/Strassen_MM_Focus' python status/server_v011.py \
  --port 8767 --campaign-state '/absolute/path/Strassen_MM_Focus/runs/NEW-COHORT/progress.json'
```

The controller writes `progress.json` by atomic replacement and retains its own
append-only events. The dashboard never writes the feed or controls a worker.
Absolute and project-relative JSON paths are accepted inside the project only.
Run the server from a frozen source snapshot with `STRASSEN_PROJECT_ROOT` set to
the canonical project when archiving deployment. The server imports `server_v008`
and serves `progress_v003.html`; preserve all three files in the snapshot.

The API remains `GET /api/progress`, with the new `campaign` object alongside
the existing actual agent records, activity, work log, and commits. Missing feed
means unknown; old campaign counters are never substituted.

## Controller snapshot schema

```json
{
  "schema_version": 1,
  "campaign_id": "registered-campaign-id",
  "title": "LLM shapes then 168-shape comparison",
  "cohort_path": "runs/NEW-COHORT",
  "state": "preparing",
  "stage": "preflight",
  "updated_utc": "2026-09-21T00:00:00Z",
  "heartbeat_utc": "2026-09-21T00:00:00Z",
  "worker_last_data_utc": null,
  "detail": "Preparing the registered execution; no measurements completed.",
  "next_step": "Run the first frozen LLM stage.",
  "evidence": [],
  "stages": [
    {"id": "preflight", "label": "Preflight", "kind": "preparation", "state": "preparing", "detail": "Preparing inputs; excluded from measurement progress.", "evidence": []},
    {"id": "LLM-screen", "label": "18 LLM shapes · screening", "kind": "measurement", "state": "not_started", "completed": 0, "expected": null, "succeeded": 0, "failed": 0, "measured": 0, "worker_last_data_utc": null, "detail": "Denominator will come from the frozen candidate plan.", "evidence": []}
  ],
  "selected_results": []
}
```

The example has no fabricated execution count or denominator. Register all
LLM and 168-shape screen/confirmation stages before running; derive each expected
count from its frozen plan, including seeds and scheduled native controls.

- States: `unknown`, `not_started`, `pending`, `preparing`, `running`, `waiting`,
  `succeeded`, `completed`, `failed`, `attention`, `blocked`, `cancelled`.
- Stage kinds: `preparation`, `measurement`, `analysis`. Only measurement stages
  enter the progress denominator. Unknown numbers stay JSON `null`.
- Count **distinct `(group_id, seed, arm_id)` terminal `call`-scope outcomes**.
  Do not double-count `prepared_kernel`. Starts, compilation, queued candidates,
  heartbeat ticks, and preparation are never completed measurements.
- `completed = succeeded + failed`. `succeeded` means the registered numerical
  and sample-coverage contract passed; numerical/compile/OOM failures are retained
  in `failed`. `measured` counts resolved attempts with actual timing samples,
  including numerically failing attempts; it can be smaller than `completed`.
- A completed stage may have failed scientific outcomes. Use `completed` for
  workflow completion and preserve the succeeded/failed counts. Never turn an
  ineligible result into a successful speedup claim.
- Use `heartbeat_utc` for a responsive local controller, **separately** from
  `worker_last_data_utc` for newly received worker evidence. Stage timestamps may
  also be provided. The dashboard labels active status stale after 120 seconds
  without a valid heartbeat; quiet worker evidence is a separate warning.
  Stale evidence does not prove termination. Terminal states remain terminal.
- Stages may include `current_candidate`, `detail`, and project-file `evidence`.
  Evidence entries are `{ "label": "Journal", "path": "runs/.../results.jsonl" }`.
  Only public project evidence files are linked; private or escaping paths fail.

## Selected result rows

Each row in `selected_results` has:

```text
shape_id: string
shape_mkn: [positive M, positive K, positive N]
stage: stage ID
scope: call | prepared_kernel
selection_status: screen_selected | confirmed | unavailable
method: native_default | native_tuned | cubic | strassen1 | strassen2
candidate_id: selected candidate ID or null
tile: [BM, BN, BK] or null for compiler managed
mean_ms: finite positive mean, or null
eligible: true | false | null
status: recorded terminal/eligibility detail
evidence: [{label, path}, ...]
```

Keep failed selections and absent family winners as unavailable rows. A retained
timing may coexist with `eligible: false`; it is visibly ineligible, not a valid
performance claim. Screening and fresh confirmation must remain distinguishable.
The view provides stage, scope, method, eligibility, and shape/candidate filters.
It renders 100 matching rows at a time with a button to reveal more.

## Agent cards

Use the existing `tools/record_progress_v001.py --kind agent --id AGENT_ID ...`
workflow. IDs and names are generic, including `llm_campaign` and
`shape_campaign`. Cards display the actual latest journal assignment, timestamp,
and reported state. Old active assignments become visibly stale after 120 seconds;
the page does not fabricate worker or agent process status. Parent orchestration
owns these updates; preparing this dashboard does not advance experiment counts.

## Root-owned validation

The new offline test file is ready for the root archive manager to execute:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests \
  -p 'test_campaign_progress_v001.py' -v
```

It imports the server without binding a port and uses temporary fixture JSON.
Capture `status/server_v011.py`, `status/server_v008.py`, `progress_v003.html`,
this contract, and the test source before the validation. No test has been run
by the dashboard author.

Do not launch TPU work to test this page. In a unique archived local validation,
compile/import the new server plus its frozen v008 dependency, feed fixture
snapshots, and verify: missing/invalid feed is unknown; preparation has no counts;
two scopes count only one controller-reported outcome; invalid totals hide the
progress bar; unknown pending denominators stay unknown; stale heartbeat differs
from quiet worker data; terminal completion remains terminal; generic agent IDs
appear with stale active assignments labeled; mean zero/missing is unavailable;
traversal/private/symlink escape evidence cannot be served.

Then start a disposable loopback server with a fixture feed and browser-check
the main page, five-method filters, scope switching, 100-row pagination, empty
results, narrow/mobile layout, and disconnect message. The deployment command
above binds only to 127.0.0.1 and does not restart or kill the older server.
