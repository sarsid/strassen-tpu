# Execute the actual accuracy worker in the notebook cell

Three detached-worker allocations disappeared before any GSM8K batch completed.
The first two ended approximately 21 minutes after launch. The third started
at 06:56:15 UTC, received an additional real diagnostic notebook execution at
06:59:55, and disappeared between the last acknowledged keepalive at 07:21:02
and the absent-assignment check at 07:22:02. Keepalive requests returned HTTP
200 throughout. The host's temporary system-sleep guard was active during this
third loss. A separate earlier host sleep interruption is documented, but does
not explain these allocation losses.

The observations suggest notebook execution lifecycle matters; the internal
Colab termination reason is unavailable. Treat this as a hypothesis, not a
proven timeout policy. Do not synthesize activity or bypass account/service
limits. Instead execute the real experimental worker synchronously in the
notebook cell, the normal execution model for notebook workloads.

`resume_accuracy_remote_v002.py` restores the same hash-verified saved data and
calls the original frozen worker directly, instead of detaching that worker
and immediately finishing the cell. The worker still launches the exact same
isolated scientific subprocesses. The host client waits for the real worker,
with a 24-hour request limit. Host starter v005 dispatches this client once,
verifies launch provenance through the contents API and starts the existing
checkpoint observer. Lost responses are reconciled; no duplicate launch.

The scientific source bundle, checkpoint seed, environment pins, models,
subsets, settings, prompts, margins, algorithms and analysis are unchanged.
Exact contract matching and fresh runtime admission remain mandatory. Original
MMLU/perplexity results are retained; incomplete GSM8K must restart at its first
unsaved batch. No artificial keepalive loop runs in the notebook. The existing
sanitized transport keepalive and temporary AC-only host sleep assertion remain.

Retain any allocated TPU on faults. Verify actual kernel/worker progress and
saved batches before claiming stability. Respect allocation refusals and runtime
limits. Release only after verified final archival or explicit user direction.
