# v5e 168 shapes: historical code with BF16 output added

Status: queued after the v6e study; no new measurements yet.
The validated one-shot handoff starts this study only after v6e reports
complete and its allocation is verified released. It stops on failure.
Handoff evidence: `runs/20260927-arch-handoff-v001/execution-v002/`.
The live dashboard will be http://127.0.0.1:8789/ once launched.

Use the historical September 21 MAIN kernels and tuning menu. Add only a final
BF16 conversion to the existing FP32 complete call; retain FP32 accumulation,
the original padding/cropping, 48 MiB custom limit and historical Native flags.
Do not substitute the v6e joint tuner or its compiler stack.

Five methods per output contract: default Native, tuned Native, tuned cubic,
S1 and S2. Rerun FP32 and BF16 separately. Preserve tuning, errors, raw samples,
failures, source and environment identity. The existing
[historical FP32 archive](../168_shapes_fp32_20260921_v001/README.md) remains
unchanged and does not count as new measurements here.

Recovery checkpoint, 2026-09-27 03:37 UTC: v5e has not started. The original
handoff stopped when the v6e controller lost its launch acknowledgment.
`runs/20260927-arch-handoff-v002` now waits for the verified v6e continuation
to finish and release its allocation before starting the unchanged v5e plan.

Checkpoint 2026-09-27 21:46 UTC: the persistent supervisor in
`runs/20260927-arch-supervised-v001` replaces the blocked one-shot handoff.
It will start this unchanged historical v5e experiment after the v6e results
are complete and the v6e TPU is released. It also recovers v5e failures using
whole screen/confirmation blocks and keeps output precisions separate.
