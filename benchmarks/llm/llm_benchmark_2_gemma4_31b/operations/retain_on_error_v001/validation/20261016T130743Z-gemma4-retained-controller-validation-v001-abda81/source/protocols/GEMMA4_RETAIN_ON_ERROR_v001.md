# Retain the TPU during troubleshooting

On October 16 the user instructed: "once an error occurs don't release the
TPU and try to trouble fix it". This supersedes automatic release on failure
in the preceding Gemma4 protocols.

Preserve every failed attempt and reuse the owned endpoint to inspect, repair,
validate and retry. Do not allocate another machine while this endpoint is live.
Neither a polling error nor a scientific gate failure permits automatic release.
Release after the requested work is successfully finished or when the user asks.

The current numerical diagnostic retains the TPU even after successful evidence
collection, because diagnosis must be followed by any necessary repair. Local
controller v002 saves archives and a retained-allocation receipt; it never calls
the release command. Verified provider-side runtime loss is recorded separately.
Starter v002 keeps its sleep-prevention scope active and checks only that same
allocation while it is retained. Polling timeouts are unknown state, not absence.
The existing provider keep-alive remains active. Provider reclaim remains possible.

The original remote source bundle and measurement code stay unchanged. A separate
controller override manifest pins the new local orchestration source. The pending
allocation request is allowed to settle; the old supervisor is retired only after
preserving its process and request receipts. No duplicate request is launched.

Validate timeout recovery, missing worker, remote failure, corrupted export,
successful diagnostic, verified runtime loss and reuse of an existing allocation
before activating the replacement. No accuracy gate or performance queue changes.
