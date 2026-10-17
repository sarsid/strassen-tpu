# First B64 diagnostic-memory retry: failed

This immutable retry reused the original tuning and retained TPU. The bounded
error reduction passed the original local gate/up allocation failure. All six
local projection confirmations and decoder layers 0–4 completed. Both attempts
then failed at the first Native global-Q projection warmup: its compiled
workspace required 20.03 GiB while 18.31 GiB was available. Inactive captured
inputs and propagated arm states remained resident. This was memory exhaustion,
not a reported prediction-quality failure. Partial timings are not full B64
results and must not be pooled with later attempts.

The complete numerical/log archive is SHA-256
`352f30358ba9fa4cea4c2b81f06d254d6cb1c443e76ce8aef6f2c6a9bf61f018`.
No natural-quality or offloaded-prefill result completed. The TPU was retained.
See the separate residency-repair operation for the next attempt.
