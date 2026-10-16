# Launcher recovery, October 16, 2026

The v6e was allocated at 04:27 UTC. Initial launch failed because Colab exec-file
executes a source string without the frozen bundle's sibling modules on sys.path.
The original monitor accepted an empty status snapshot, hiding that failure.

At 04:41 UTC, the validated standard-library launcher extracted the exact original
bundle before invoking its worker. The same owned TPU now runs official-weight
qualification. No kernel, model, tuning or measurement source was changed.
The original bundle SHA is retained in the launch receipt. No timings are claimed.

The separate controller v002 uses this launcher and rejects missing-worker
snapshots, with timeout, runtime-loss and empty-snapshot regression checks.
The active v001 controller resumed tracking the successfully launched worker;
controller v002 is preserved for subsequent starts, not substituted mid-run.
