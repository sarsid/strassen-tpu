# Accuracy study monitoring recovery — October 19 UTC

The v001 worker launch was acknowledged at 2026-10-18 21:57:39 UTC. Its
observer then exited at argument parsing: the starter passed `--expect-endpoint`
but the observer requires `--endpoint`. No observer read or result download
occurred. At 2026-10-19 00:42 UTC Colab listed no allocation; direct recovery
of the old owned endpoint returned HTTP 404. The available records do not
establish when or why the remote runtime disappeared, or how far its evaluation
advanced. Count zero verified downstream results from that attempt. Preserve
its launched source, receipts and failure evidence without fabricating outcomes.

Recovery changes only host orchestration: starter/observer v002 pass the exact
observer CLI, require a live PID and a matching readiness acknowledgement,
then require a successful remote status read before declaring launch success.
The starter remains a supervisor and restarts a crashed observer without
duplicating the remote worker. Contents-API transient errors remain retryable.
Offline tests execute the real child CLI, reject the historical bad option,
and check process/readiness matching. The full dataset driver is also exercised
with explicitly fake outputs to verify all task paths and checkpoint resumption.

The actual model/algorithm/task settings and frozen remote source remain
unchanged unless a separately recorded implementation failure requires repair.
Retry in a new run folder, preserving the failed attempt. Observe the real
worker through admission and the first saved benchmark pairs before relying on
unattended execution again. Keep the live view accurate; unverified remote work
must never be counted as completed accuracy results. Retain on errors and
release only after all requested artifacts have been verified locally.
