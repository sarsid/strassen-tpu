# B64 partial attempt: diagnostic out-of-memory failure

B64 tuning and representative block qualification completed and passed. Both
resident confirmation attempts failed after local gate/up projection timing,
while the untimed error checker formed a full FP32 difference array. It tried
to allocate 5.25 GiB with 2.29 GiB free in the second attempt.

This is an infrastructure/diagnostic-memory failure, not a failed numerical
quality gate or evidence that timed GEMM execution failed. The partial projection
samples are retained for diagnosis and must not be pooled with a new attempt.
No full resident aggregate, natural quality or offloaded prefill result exists
for this attempt. The raw logs, saved tuning profiles, successful qualification
and both failed confirmation traces are preserved unchanged.

The TPU was retained. The existing worker proceeded to B4 and then B1. A separate
[bounded-diagnostic repair](../../../operations/b64_metric_memory_repair_v001/)
has passed local checks and is prepared for a fresh B64-only retry on that same
TPU after those batches finish, reusing the original tuning. No GPU/TPU execution
of the repair is claimed by this publication checkpoint.
