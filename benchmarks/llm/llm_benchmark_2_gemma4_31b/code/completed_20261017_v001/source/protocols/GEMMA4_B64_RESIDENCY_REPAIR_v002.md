# B64 recovery: bound diagnostic buffer residency

The first bounded-error retry passed qualification and completed all local
projection confirmations plus decoder layers 0–4. Both attempts then failed at
the first Native global-Q projection warmup. Its compiled temporary workspace
is about 20.03 GiB, but 18.31 GiB was available because unrelated captured
activations and propagated states remained resident. This differs from the
original 5.25 GiB FP32 error-difference allocation; the first repair worked.
Preserve every failed run and do not pool partial timings.

Stage v006 keeps the original model, all four kernels, full B64/S1024 inputs,
saved tuning, compiler options, timers, warmup counts, rotated sample order,
BF16 output/FP32 accumulation, tokenization and quality thresholds unchanged.
It changes only storage lifetime outside measured calls:

- Park all propagated arm states on host while isolated projection diagnostics
  run, then restore exact BF16 states before resident block/layer timing.
- Capture and park diagnostic inputs/residuals on host; only the current full
  projection inputs are placed on the TPU. Release each measured output before
  the next call. Copy final outputs to host only after stopping the timer.
- Keep inactive natural-quality segment/arm states on host. Execute each full
  B64 layer on the TPU, retaining exact BF16 outputs on host between calls.
  Quality evaluation carries no performance timing claim. Its metric formulas,
  scored tokens and gates remain unchanged.

The isolated projection timings exclude these diagnostic transfers, as they
already exclude input preparation. Resident layer timing and transfer-inclusive
prefill functions remain unchanged. Device/host storage policy is explicitly
logged. This is not microbatching, smaller model computation or a new algorithm.

CPU tests must verify bit-identical propagated states and projection outputs,
unchanged scientific/timing functions, no device-output retention between timed
projection calls, and safe retry admission. Commit sources before validation.
Repeat qualification and complete all B64 measurements in a new immutable run,
reusing the original tuning and same retained v6e. Verify the original five
batches passed and the prior retry is finished/archived before launch. The
exclusive remote lock prevents concurrent TPU workers. No new allocation or
retuning is needed; retain the TPU on failure and release only after all results
pass and are preserved.

The initial CPU validation of v005 caught an ordering change: JAX device_get
sorts dictionary keys, changing the projection site enumeration. Stage v006
saves the original site order before host transfer. All numerical comparisons
remain strict. The failed v005 validation is retained separately.
