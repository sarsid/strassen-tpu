# Controlled allocation diagnosis — October 19 UTC

No downstream evaluation has started in this recovery. Thirteen requests using the original persistent notebook identity completed the initial handshake quickly but returned HTTP 502 after approximately 240 seconds on the allocation POST. The user confirmed sufficient compute credits on the authenticated controller account.

The existing retry supervisor and scheduled recovery were paused; its in-flight request was allowed to finish. A committed and tested diagnostic reconciled the old request, confirmed zero allocations, then sent exactly one POST using a fresh recorded identity with the same v6e hardware request and authentication. It also returned HTTP 502 after **240.041 seconds**, with a generic temporary-error HTML response. Both identities and the assignment list were reconciled afterward: **zero assignments**.

Changing the notebook identity did not resolve the failure. These observations do not establish whether the upstream cause is capacity, an account restriction, or a failure specific to the automated allocation path. The next requested check is a manual v6e connection in Colab on the same account; automatic requests remain paused to avoid competing allocations during that check. Browser-control tooling could not initialize, so the manual result is pending user input.

The diagnostic preserves only allowlisted response classifications, content type, length and hash; no raw response bodies or credentials are archived. Four safety checks passed, covering refusal to allocate alongside an existing assignment, recovery of the old identity without another POST, exactly one fresh POST after reconciliation, and omission of secrets from response summaries.

The model checkpoints, fixed test subsets, Native/S1 algorithms, BF16 precision, statistics and frozen scientific source are unchanged. This is operational evidence, **not a model-accuracy result**. Earlier published snapshots remain unchanged.
