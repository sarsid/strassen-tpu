# Keepalive diagnostics after two runtime losses

Both allocated runtimes disappeared approximately21minutes after launch during
the first GSM8K batch. Root cause is not established. The completed Qwen14 MMLU
and perplexity task results remain unchanged and preserved.

The old keepalive discarded stdout/stderr and the installed SDK suppresses read
timeouts. Host controller v004 now logs sanitized HTTP outcomes and exceptions,
without headers, tokens or bodies; only an actual2xx response is acknowledged.
Four validation checks passed. This adds evidence collection, not a claimed fix.
The next recovery is queued with exactly the same scientific source and saved
batch seed. No new GSM8K result is claimed. Earlier snapshots are unchanged.
