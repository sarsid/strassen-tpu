# Prepared B64 diagnostic-memory recovery

Stage v004 changes only the untimed error reducer and its method metadata. The
model, tuned kernels, full B64 activation shapes, timing functions, token inputs
and scientific thresholds remain unchanged. Projection errors are reduced in
chunks of at most 1,048,576 elements, with FP32 device tile sums and host FP64
combination. The reduction grouping differs; the metric definitions are the same.

Six local checks passed for numerical equivalence, BF16/FP32/tail cases,
zero/nonfinite values, bounded tiles and unchanged timing/scientific functions.
Seven local recovery-admission tests passed, including rejection of an active
predecessor, other failed batches, altered profiles, missing qualification/archive,
and an occupied exclusive TPU worker lock. These are CPU checks; the repaired
B64 experiment has not yet run on the TPU.

The frozen recovery bundle is prepared for the existing recovery heartbeat to
launch after B4/B1 complete successfully. It reuses the retained v003 runtime,
checkpoint cache and exact B64 profile, repeats qualification and all remaining
measurements, and does not retune or allocate/release a TPU. See the frozen
[runbook](source/protocols/GEMMA4_B64_RETRY_RUNBOOK_v001.md) and
[scientific repair protocol](source/protocols/GEMMA4_B64_METRIC_MEMORY_REPAIR_v001.md).
All original failures are preserved separately. No activation, weight or
credential tensors are included.
