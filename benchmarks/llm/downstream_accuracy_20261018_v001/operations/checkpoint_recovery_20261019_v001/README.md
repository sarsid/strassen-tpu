# Checkpoint recovery after allocation disappeared

Qwen3-14B completed MMLU and perplexity before the runtime disappeared during
GSM8K. Two complete tasks are preserved in the separate results snapshot.
No completed GSM8K batch exists. Cause of allocation disappearance is unknown.

This adds a host-side resume wrapper and a data-only recovery archive. The
original frozen remote scientific source bundle remains byte-identical. The
original evaluator requires exact contract equality, repeats runtime admission,
and reuses saved pairs rather than repeating MMLU/perplexity. No benchmark
settings or statistical criteria are changed.

The first validation failed on a macOS temporary path alias in the test fixture
and a missing archived import dependency. Both failures are retained. The
corrected validation passed eight tests, including the original driver's
injected-failure resume test, exact-byte restoration, rejection of corrupt or
unexpected archive members, duplicate-run rejection and observer CLI parsing.

The new supervisor is queued on the already authorized alternate Colab account.
This operations snapshot records queued recovery, not proof of remote execution.
Read newer live receipts for allocation/execution status.
