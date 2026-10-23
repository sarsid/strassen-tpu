# v6e 168 shapes: joint tuning, FP32 and BF16 output

## Completed October 2, 2026

All 168 geometries are complete in both output precisions: 336 precision
groups with default Native, tuned Native, tuned cubic, S1 and S2. The phase
archives, tuning choices, timing samples and numerical errors are committed.
The combined report is [RESULTS.md](report/RESULTS.md), the tuner rationale is
[TUNER_DESIGN.md](report/TUNER_DESIGN.md), and [COMPLETE.json](COMPLETE.json)
records full coverage and the final results checksum.

The owned TPU has been released and verified absent. The supervisor finished
and its recovery heartbeat was deleted. No subsequent experiments are queued.
These final results are committed in the local research repository; the
earlier GitHub benchmark publication has not yet been updated to this snapshot.

The dated checkpoints below are retained as historical evidence.

Status as of 2026-09-27 00:30 UTC: running. Hardware qualification passed
72/72 exact cases. The first shape has completed both output contracts;
the boundary preflight is screening. This text is a timestamped checkpoint,
not a live status assertion.

Live dashboard: http://127.0.0.1:8788/
Controller: `runs/20260927-arch-v6e-168-v001/progress.json`.

Authorized protocol: [architecture-specific studies](../../../docs/ARCHITECTURE_STUDIES_20260927_v001.md).
Five methods per output contract: default Native, tuned Native, independently
tuned cubic, S1 and S2. Both precisions stay separate. Kernel bodies come from
the completed v6e joint pilot, with ordinary timed boundary padding/cropping.

Verified phase bundles will be committed under `phases/<cohort>/`; the exact
common source and protocol under `provenance/<cohort>/`. Final reports and
tuning/error tables will be under `report/`. Absence of `COMPLETE.json` means
the full study is not complete. A complete phase is only part of the study.
The controller records allocation release separately.

## Recovery checkpoint: 2026-09-27 03:37 UTC

A lost Colab launch acknowledgment stopped the first controller. Its remote
worker completed successfully; the checksum-verified confirmation archive
was recovered without rerunning measurements. All 63 cases were numerically
eligible. The interruption left 28 geometries complete in both output dtypes.
See [diagnosis and recovery](recovery/20260927-lost-launch-ack-v001/DIAGNOSIS.md).

The same allocation resumed in
`runs/20260927-arch-v6e-continuation-v001`; the live dashboard remains
http://127.0.0.1:8788/. Measurement source, tuning and precision contracts are
byte-for-byte unchanged. The new transport reconciles lost acknowledgments
by reading the existing worker state; it never repeats the launch.
At this checkpoint the next screening phase had 19 of 346 cases completed,
all successful. The final report will verify all original, recovered and
continuation phases. This is a timestamped checkpoint, not a live status.

## Automatic recovery checkpoint: 2026-09-27 21:46 UTC

Sixty complete geometries are preserved. The arch-060 screening worker
finished remotely, but its result download failed and the runtime was later
absent; the partial local evidence is retained in
[the download-failure archive](recovery/20260927-archive-download-v001/DIAGNOSIS.md).
The incomplete comparison will restart on a replacement TPU.

Automatic supervision is active under `runs/20260927-arch-supervised-v001`.
It handles transfer retries, existing-worker reconciliation, runtime replacement
after verified absence, checkpoint recovery and the subsequent v5e study.
Fourteen recovery/lifecycle checks passed. Replacement setup completed with the
original pins; hardware qualification is now running. Kernel and tuning sources
are unchanged. The dashboard remains http://127.0.0.1:8788/.
A separate two-hour task heartbeat checks for unexpected supervisor failure.
This paragraph is a timestamped checkpoint; the live log is authoritative.
