# Recovered source-upload failure

At 04:41 UTC on September 29, 2026, uploading the source archive for shape 104 returned TransportError. Launch had not been attempted and there were no failed-attempt measurement samples. No HTTP status or service explanation identifies a deeper cause.

The supervisor verified that the same TPU allocation was present, the remote run directory did not exist and the execution lock was free. It automatically started a new cohort at 04:44 UTC, passed the smoke checks and completed shape 104 screening and confirmation by 05:37 UTC. Both phases have matching device identities. Shape 105 is now running; 105 whole comparisons are saved, including the early shape-131 anchor.

No manual restart, new TPU, kernel change, search-space change or sample pooling was needed. Original failed files remain unchanged; this folder preserves copies, hashes, reconciliation receipts and the successful retry identities.
