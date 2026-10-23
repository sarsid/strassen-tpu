# Foreground runtime loss and checkpoint recovery

The v005 worker began at 07:49:10 UTC on October 19. Its last acknowledged
keepalive was at 11:17:45 UTC; the next assignment check at 11:18:45 reported
the endpoint absent. The observer recorded runtime loss at 11:20:17. An
independent assignment listing also returned empty. The provider's internal
reason is unknown. This approximately 210-minute foreground run differs from
the earlier approximately 21-minute detached-worker losses; do not attribute
it to the same mechanism without further evidence.

Qwen3-14B MMLU and perplexity remain complete. The first eight paired GSM8K
examples are preserved in parent commit `19043cb`; the second GSM8K batch was
incomplete and cannot be recovered from per-token state. No model-level or
full-study completion is claimed. The controller did not release this TPU.

Recovery v006 reuses the validated starter v005 and foreground launch wrapper
v002 with the identical scientific source bundle. Its verified seed contains
20 paired batch files plus the original manifest and execution contract.
This adds the completed first GSM8K batch to the previous seed. Fresh
train-only admission and exact regenerated execution-contract equality remain
mandatory before saved results can be reused.

The obsolete host execution client and dashboard were retired after confirming
allocation absence. A single replacement-allocation supervisor started at
12:50:42 UTC, using the already authorized alternate account. At this snapshot
it was waiting for allocation. No settings, scoring rules, model checkpoints,
dataset examples or scientific source files changed. Earlier evidence remains
intact. The live dashboard continues at localhost port 8791.
