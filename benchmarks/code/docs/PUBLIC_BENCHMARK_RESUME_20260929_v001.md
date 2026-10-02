# Resume the paused architecture studies

No allocation is authorized by opening or verifying this checkpoint. The owner
will create/authenticate a new Colab account later. At publication, the automatic
supervisor was stopped, its lock released, and the two-hour heartbeat paused.
No TPU was owned. An unattributed CPU runtime was left untouched.

## Remaining measurements

- v6e: 113 complete whole screen/confirm pairs (arch-001 through arch-112,
  plus early boundary/large-shape preflight arch-131). The remaining 55 units
  are listed exactly in `checkpoint.json`.
- arch-113 has an interrupted screening attempt in recovery evidence. It is
  not a completed comparison. Restart its whole screen/confirm pair on the new
  qualified runtime; never pool partial samples from the lost runtime.
- v5e: after completing v6e, run the separately frozen historical v5e code and
  tuning menu with FP32 and BF16 output. New v5e measurements have not started.

## When the owner is ready

1. Authenticate the intended Colab account privately, outside Git and archives.
   Inspect its live inventory and remaining compute units. Do not reuse cached
   credentials, proxy tokens or endpoint/session identifiers from the old account.
2. Reconcile the old local pending allocation request before restarting any
   supervisor. Do not adopt or terminate an unrelated CPU/runtime. Preserve the
   original pause and recovery evidence; record the account transition separately.
3. Reuse the existing local study history if working in the original research
   workspace. The active supervisor source is identified in
   `provenance/supervisor/active-source.json`; archived absolute paths are historical.
   On another machine, reconstruct portable phase/cohort paths from the bundles
   and receipts before attempting a continuation. This export supports offline
   report replay directly; it is not a drop-in live session state directory.
4. Allocate one v6e runtime, install the frozen stack, record its new identity,
   and pass all 72 arithmetic qualification cases. Keep measurement kernels,
   precision, candidate menus, seeds and decision gates unchanged. The copied
   runtime/supervisor tools implement bounded transfer retries, lost-launch
   reconciliation and whole-comparison recovery. A new device identity requires
   a new cohort, not editing the old one.
5. Continue only missing complete comparisons. Archive every phase and failure,
   maintain the live log, and release the owned TPU after retrieval. Re-enable
   the paused heartbeat only after the authenticated continuation is ready.
6. Run the v5e study with its own code/compiler/memory settings; do not transplant
   the v6e search policy. Keep FP32 and BF16 comparisons separate. Publish another
   versioned checkpoint/final report without rewriting this snapshot's evidence.

In the original workspace, `tools/start_arch_supervisor_v003.py` is the guarded
starter. Its `start --folder runs/20260927-arch-supervised-v001` command is only
appropriate after the account and stale-request reconciliation above. Merely
running it against the archived old-account state is not a valid migration.
