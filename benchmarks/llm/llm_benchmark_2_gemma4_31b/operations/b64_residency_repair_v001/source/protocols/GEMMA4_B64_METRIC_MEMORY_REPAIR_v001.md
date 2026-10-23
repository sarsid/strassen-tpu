# Gemma4 B64 error-diagnostic memory repair

The v003 B64 run completed tuning and block qualification. Both confirmation
attempts failed after timing the local gate/up projection, in the untimed
`error()` diagnostic at `d = v - r`. Each full FP32 buffer was 5.25 GiB; the
second attempt had only 2.29 GiB free. This is a diagnostic-memory failure,
not evidence of an accuracy failure or a failed timed matrix multiplication.
Preserve both attempts and their partial timings; do not pool those timings
with a retry.

Stage v004 replaces only this diagnostic with chunks of at most 1,048,576
elements. Each device tile reduces FP32 squared error/reference norms and max
errors, and returns five scalars. Squared-norm scalars are combined using host
FP64 `math.fsum`. The mathematical metrics and all quality limits remain the
same; floating-point reduction grouping changes and is recorded explicitly.
Kernels, model formulas, profiles, full B64 activation shapes, timing scopes,
sample counts, natural-text BOS policy and token hashes remain unchanged.

Reuse the complete hash-verified B64 profile from the same owned v6e, software
stack and source-pinned model. Do not retune. A fresh immutable retry must run
the FP32 evidence check, B64 block qualification, all confirmation measurements,
natural quality and offloaded prefill. Nothing from the failed partial
confirmation qualifies as a completed result.

The v003 worker is currently running B4 and then B1 on the same TPU. Allow it
to finish; do not interrupt it or start concurrent TPU execution. Its B64
failure forces the existing controller to retain the allocation afterward.
The authorized recovery heartbeat must then run this B64-only retry on that
same endpoint after verifying the predecessor is finished and the remote
exclusive lock is available. No new allocation is authorized while it exists.
Keep the existing keepalive and sleep-prevention supervisor alive. Release
only when B8/B16/B32/B4/B1 plus the B64 retry are complete, all gates pass, and
all artifacts are downloaded, hash-verified and committed. No other model or
architecture queue is included.

Before remote execution, commit sources and archive local tests of chunk
boundaries, BF16/FP32 numerical equivalence, zeros/nonfinite values, bounded
tile size and unchanged timing/scientific functions. The original v003 source
and result directories stay immutable.
