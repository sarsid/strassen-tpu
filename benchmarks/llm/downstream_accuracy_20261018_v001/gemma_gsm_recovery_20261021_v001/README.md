# Gemma final GSM8K recovery

This additive operational update resumes only the last16 of32 paired Gemma
GSM8K examples after the runtime disappeared. Both Qwen evaluations and Gemma
MMLU/perplexity are complete. All21 saved Gemma result batches are preserved
locally; this folder publishes recovery code and validation, not new task scores.

The numerical source bundle and evaluation worker are unchanged. The new seed
validator restores114 MMLU pairs,32 perplexity windows and16 GSM8K pairs, verifies
their hashes and execution origins, and requires fresh preflight/admission and
exact scientific-contract equality before reuse. The first13 MMLU batches retain
the disclosed original admission-receipt gap; later batches retain actualv011
admission/preflight provenance. New receipts do not retroactively fill that gap.

Eight recovery checks passed. To rerun them without accelerator access:

```sh
cd code
python3 tools/check_accuracy_gemma_recovery_v002.py
```

Allocation retries remain serialized, with one foreground worker and a read-only
observer. The bounded AC-only sleep guard changes only the expected supervisor
identity. Retain runtime on errors; release only after the complete study and
all result archives have been verified locally.

This publication includes experiment source and test results only. It contains
no model weights, downloaded model checkpoints, recovery seed archives,
credentials, or raw local power/process diagnostics. Prior result snapshots
remain unchanged. No scientific score, precision, algorithm, dataset or decoding
setting is modified by this recovery update.
