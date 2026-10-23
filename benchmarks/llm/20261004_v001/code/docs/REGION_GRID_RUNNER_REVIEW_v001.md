# Region-grid runner reuse review, v001

Review date: 2026-09-20. Local source and sealed historical evidence were read.
This review did not query current account usage, allocate a TPU, connect to a
runtime, or run a benchmark. The root task owns the new cohort and lifecycle.

## Reuse recommendation

Use `runtime/launch_phase_v004.py` with a new
`strassen_mm.benchmark_region_grid_v001` preflight wrapper that delegates to the
unchanged `benchmark_power_grid_v001` timing implementation. Freeze one source
archive before smoke and reuse its exact bytes for screen and confirmation.
Retain `kernels_v002` and its versioned dependency. Do not include the concurrent
optimized-Strassen experiment in this comparison or substitute its kernels.

The launcher creates its run directory and files exclusively, checks the uploaded
source archive hash, extracts source beneath the run directory, sets PYTHONPATH
to that snapshot, launches a fresh child, uses a nonblocking exclusive TPU lock,
records bounded failures, and seals an exclusive completion archive. Use a new
UTC/random run ID for every attempt, including failed or timed-out attempts.

`runtime/colab_control_v001.py` is preferable to raw Colab CLI file transfer:
its upload rejects an existing remote path and its download opens the local file
exclusively. The ordinary Colab CLI upload can replace existing remote content.
Pass the frozen `--expect-endpoint` to every session action. The parent Jupyter
kernel must not import JAX; the detached benchmark child owns the TPU.

## Cohort and holdout protection

The existing runner already requires confirmation to match screen's campaign,
shape manifest, complete screen journal, Python source hashes, deterministic
selection replay, and exact environment identity. A screen selection from the
old allocation cannot validly seed confirmation on a new one. Generate a new
campaign ID/cohort ID and capture a new provisioning identity, then use the
smoke environment as the expected identity for subsequent runs.

The existing generator excludes holdout IDs, but the runner itself does not
independently enforce holdout exclusion. The new wrapper must check IDs and
actual (M,N,K) coordinates, including renamed aliases of old reserved shapes,
before importing JAX. It must require all phase shape inventories to agree with
the new frozen manifest and keep new holdouts out of smoke, screen and confirm.

`snapshot_sources` copies every `src/strassen_mm/*.py` file. Rebuilding source
from the live checkout between phases can therefore fail confirmation even
when the edited file belongs only to the concurrent optimized experiment.
A single frozen archive for all phases avoids that race and establishes which
implementation produced the measurements.

`tools/archive_v002.py` runs global `git add -A`; it can capture unrelated
concurrent edits. Use a cohort-specific explicitly enumerated snapshot and
source manifest. Do not modify or append to older experiment artifacts.

## Timing and analysis constraints

The original screen groups each candidate exactly once per shape. Only the first
four of 24 groups include a native preset. A direct margin computed from a
custom candidate's screen mean and the separately measured default-native mean
is an unpaired descriptive ratio; matching round numbers in different groups
does not make it paired. Confirmation places frozen family winners and default
native in the same headline group and supports its recorded paired intervals.
Keep confirmation entirely separate from margin-model fitting.

Do not add default native to every screen group without a new selection protocol:
`eligible_rows` requires exactly two scoped results per candidate/shape and
would reject those repeated reference rows. The chosen wrapper intentionally
preserves the original timing groups and selector.

The old `audit_power_grid_v002.py` hardcodes 60 exploratory shapes, 12 holdouts,
72 manifest rows and 6,240 scoped screen results. It cannot be used unchanged
as the new cohort's acceptance test. Reuse its numerical, timing, source-hash
and deterministic-selection checks only through a new validator with dynamic
manifest-derived counts. Never interpret an old-count assertion as a scientific
failure of a new shape set.

## Capacity and cost planning

The registered runtime budget is a conservative 8 GiB estimated live-device
limit and 48 MiB custom-kernel VMEM limit. `TuningRunner` preflights original
inputs, largest padded inputs, distinct prepared-input buffers and reserved
outputs/temporaries per group. Executable/runtime peak memory is unmeasured;
new shape admission is not proof that a compiler/runtime allocation will fit.
Memory skips and compile failures must remain in the journal. No silent retries.

Historical observed benchmark wall times were 210.7 s for three-shape smoke,
7,070.3 s for a 60-shape/52-candidate screen, and 3,946.8 s for its confirmation
(489 groups). These are rough planning anchors only: the new shape distribution,
compilation behavior and host can change runtime substantially. They do not
measure billable time or establish current account balance.

`colab-setup/tpu_usage_monitor.py --once --output-dir NEW_DIRECTORY` exposes a
sanitized read-only account snapshot with paid compute units, account CU/hour,
and estimated paid hours remaining. Root should inspect current assignments and
usage before allocation; prefer reusing an appropriate idle allocation only
after coordinating with its owner. Active.lock protects cooperating campaign
workers, not arbitrary processes from another task. No concurrent TPU timing.

The old runtime was a logical single-device TPU v5 lite, JAX/jaxlib 0.7.2,
libtpu 0.0.21.1. These are historical facts only. Endpoint, hostname and boot ID
establish logical runtime continuity, not a permanent physical TPU serial.

## Concrete launch interface

Execute this inside the selected existing Colab runtime, with all placeholders
resolved to new cohort-specific frozen paths and IDs:

```text
python /content/Strassen_MM_Focus/NEW_UPLOAD/launch_phase_v004.py
  --archive /content/Strassen_MM_Focus/NEW_UPLOAD/source.tar.gz
  --archive-sha256 SHA256
  --run-id NEW_IMMUTABLE_RUN_ID
  --phase GRID-screen
  --campaign-relative configs/NEW_BUNDLE/campaign_region_grid_v001.json
  --benchmark-module strassen_mm.benchmark_region_grid_v001
  --expected-identity /content/Strassen_MM_Focus/NEW_UPLOAD/smoke-environment.json
  --allocation-id CURRENT_ENDPOINT
  --timeout-seconds BOUNDED_SECONDS
```

For confirmation, use the same source archive and campaign, a new run ID, phase
`GRID-confirm`, and pass the new cohort screen's selections with
`--runner-arg=--selection --runner-arg=/content/Strassen_MM_Focus/runs/NEW_SCREEN/artifacts/selections.json`.
The screen journal and source manifest must remain beside that selection.
Download each sealed completion archive before proceeding to the next phase.
Keep the new cohort's tables in new files; later comparison may align geometries
and dimensionless speedups with old evidence, but should retain allocation and
cohort identity and must not pool raw milliseconds by default.

## Implemented guard and local verification

`src/strassen_mm/benchmark_region_grid_v001.py` now validates the explicit cohort
ID and no-pooling declaration, role inventories, per-phase geometry membership,
unique shape coordinates, all twenty new holdouts, and the original twelve
holdouts reconciled against a hash-verified legacy manifest copy in the new
bundle. It checks candidate/seed consistency and selection campaign/shape
provenance before delegating to the original engine without modifying arguments.

The wrapper accepts the original provisioning identity subset for confirmation
and checks its comparable fields against screen's complete identity. It does
not mistake missing pre-JAX provisioning fields for an environment change; the
engine still captures and compares complete runtime identity before timing.

Fifteen new offline guard tests and twelve unchanged grid protocol tests passed
using `PYTHONPATH=src python3 -m unittest tests/test_region_runner_v001.py
tests/test_power_grid_v001.py`. Tests cover renamed old-holdout coordinates,
new-holdout phase leakage, duplicate coordinates and IDs, altered protection
sources, cohort mixing, seed reuse, changing candidate catalogs, retained output
files, and setup/full identity schemas. The actual generated new bundle also
passed an offline preflight with 120 exploratory shapes and 20 reserved shapes.
No benchmark was invoked by these checks.
