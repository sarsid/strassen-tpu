# Runtime loss during shape 113

At 13:12 UTC on September 29, the control API reported that the existing TPU endpoint was no longer allocated. The supervisor confirmed the missing allocation at 13:13 UTC. The provider did not explain the disappearance. 113 complete shape comparisons were saved; the incomplete shape-113 screen had 558 of 639 local candidate outcomes. These partial outcomes are preserved and excluded from whole-comparison reuse.

Replacement requests at 13:14, 13:28 and 13:55 UTC each returned generic HTTP 400 Bad Request responses. This is different from the previous runtime-loss incident, whose replacement requests timed out. The response does not identify a credit, quota, capacity or entitlement cause.

The supervisor remains alive with its exclusive lock held and continues automatic retries and late-assignment checks. No duplicate allocation or worker was started. The read-only status snapshot includes a fresh provider inventory. Original failed evidence remains unchanged; the local failed-phase archive is verified member by member, and immutable sources are referenced by hash.
