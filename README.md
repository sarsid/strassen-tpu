# Strassen MM Focus

Fresh, self-contained matrix multiplication study. The original project remains
untouched; reused implementations are copied here with provenance.

**Current results: [Start here — completed N5–N9 and fresh-v5e replication](START_HERE_v003.md).**

The subsequent N5–N9 study and its separately reported fresh-v5e replication
are documented in the [N5–N9 protocol](protocols/N5_N9_v1.md),
[decision log](decisions/N5_N9_v001.md), and versioned `START_HERE` review notes.
The scope below records the initial N1–N4 authorization.

## Initial authorized campaign

Execute N1–N4 on one TPU v5e allocation first. v6e replication and N5 onward
are later phases. Two-level Strassen is an optional future note, not an active
experiment. No LLM inference or model-quality claim is made by N1–N4.

1. N1: basic native XLA, full-tile cubic, matched quadrant cubic and one-level
   Strassen across square, rectangular, model-derived and partial-tile shapes.
2. N2: bounded, matched tile sweeps, retaining all failures.
3. N3: pure-MM optimization ablations at a fixed tile.
4. N4: numerical sensitivity, including difficult synthetic inputs.

Protocols and shape manifests live in `protocols/` and `configs/`.

## Preservation and machine consistency

- All project code, run snapshots, protocols, results and reports live here.
- Source files may be developed before their first execution. Once used, their
  exact bytes are preserved in each run's source snapshot. Later executed code
  revisions receive new versioned files; completed snapshots and results are
  never overwritten. Failures are retained as executions too.
- Every execution has a unique directory under `runs/`, its source/configuration
  hashes, logs, raw records, environment and completion record. Files are created
  exclusively; live record streams are append-only and sealed when complete.
- Commit source before execution and commit code plus collected results after
  every execution, including failed attempts, before the next experiment.
- Agents develop and review in parallel; only one process benchmarks the TPU.
- Reuse one v5e allocation. Record allocation/session identity, host, boot ID,
  TPU model/count and package versions. If that identity changes, stop the
  campaign rather than silently combining machines. Colab does not guarantee
  that a later reconnect supplies the same physical chip.
- Freeze requested numerical precision and compiler settings. Preparation costs
  and resident-kernel latency are distinct measurements. Selection and numerical
  thresholds are recorded before measuring, not adjusted to favor an arm.

`status/` is a local, read-only dashboard backed by append-only status events and
run artifacts. Its displayed view may refresh; archived source and results do not.

No remote Git publication is configured. Commits are in this folder's local Git
repository.
