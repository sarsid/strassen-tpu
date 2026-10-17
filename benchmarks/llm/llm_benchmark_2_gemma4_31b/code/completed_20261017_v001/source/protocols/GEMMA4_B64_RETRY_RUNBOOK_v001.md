# B64 retry on the retained v003 TPU

This is part of the existing authorized Gemma sweep, not a new model campaign.
Read `GEMMA4_B64_METRIC_MEMORY_REPAIR_v001.md` for the scientific contract.

The immutable recovery specification is
`configs/gemma4_b64_metric_retry_v001.json`. The prepared local run is
`runs/20261016-gemma4-b64-metric-retry-v001`, with a frozen source tar and hash
manifest. It reuses the existing v003 private environment/cache and exact B64
profile SHA-256
`e72c892479c8389be57c28c8d48d70fef61f0711300f8772aac9cc66395cc413`.

Do not launch until the existing v003 remote worker is finished, its archive is
ready, and B8/B16/B32/B4/B1 all passed. The remote launcher enforces this and
the worker takes the shared nonblocking exclusive `active.lock` before any TPU
execution. It refuses duplicate run IDs and changed profile/source hashes.
No allocation, setup/install, release or retuning is performed by the retry.

After verifying the same Colab session/endpoint is still owned, upload the
prepared source tar to a new path under `/content/Strassen_MM_Focus/`. Execute
the frozen `runtime/gemma4_b64_retry_remote_v001.py` with `--archive`, its
`--sha256`, and the registered `--run-id`. Preserve the local RPC launch
receipt. If a response is lost, poll before retrying: never blindly relaunch.

Use the existing frozen `runtime/poll_gemma4_batch_v001.py` with the retry run
ID and the original v003 session/endpoint. Its snapshot and stage-export
format is unchanged. While retry stages run, point the live dashboard to the
new local run, retaining the original full-sweep evidence. Reuse the existing
v003 keepalive and supervisor sleep-prevention scope, which retain the TPU
after the predecessor's failure. Do not start a second allocation supervisor.

Download and hash-verify completed stage/batch/full archives to new local
paths. Preserve both original failures and the new retry. Publish numerical
artifacts and frozen sources only. If all five original successful batches
and this B64 retry pass and are preserved, release the original owned TPU
with verified absence and stop the obsolete retained supervisor. Otherwise,
retain it and troubleshoot the new evidence. Update the existing recovery
heartbeat; do not create a duplicate automation.
