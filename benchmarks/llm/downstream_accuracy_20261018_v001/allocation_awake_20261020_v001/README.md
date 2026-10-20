# Temporary AC protection during replacement allocation

This additive operational snapshot preserves the allocation-wait sleep guard and its validation. It does not change kernels, datasets, scoring, scientific source bundles, saved results, or the existing allocation supervisor.

As of October 20, 2026 at 17:49 UTC, Colab had not provided a replacement TPU. Attempt 23 was pending, with one serialized request. Earlier requests returned HTTP 502 or timed out. Local laptop sleep had also paused requests. The provider termination and allocation causes remain unconfirmed; the active account has user-confirmed Pro+ and credits.

Version 001 passed five lifecycle checks. Subsequent review found a possible gap between bounded sleep assertions. Version 002 renews before expiry, starting the replacement assertion before stopping its predecessor; all six lifecycle checks passed. Both versions and checks are preserved. Version 002 was attached to the existing supervisor, and an actual renewal and operating-system assertion were verified.

The guard operates only on AC power, releases its owned assertion on battery power, and exits when the supervisor exits or an allocation is recorded. It hands off to the existing allocated-runtime guard. It changes no permanent power setting and does not prevent deliberate sleep or lid closure.

Five of nine model-task evaluations remain complete. Sixteen of thirty-two Qwen3-32B GSM8K pairs are saved. Reacquisition will resume only unfinished work using the previously published exact-source recovery and fresh admission checks. No new scientific result is claimed here.
