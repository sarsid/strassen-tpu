# Gemma checkpoint recovery after runtime loss

Both complete Qwen downstream studies remain verified and unchanged. Gemma's
first 13 MMLU batches (104 paired questions out of 114) were saved locally
before the replacement TPU disappeared. This snapshot preserves that partial
data and the validated recovery machinery. It contains no completed Gemma
benchmark or new accuracy-preservation conclusion.

The controller did not release the TPU. Local observation was interrupted after
October 21 01:25 UTC, and an independent check after resumption found no
allocation. The original termination cause is unknown. The first replacement
request returned HTTP 502; a serialized supervisor continues retrying under the
same user-authorized account. Account credits and Pro+ were confirmed by the
user; this evidence does not establish exhausted credits or a fixed time limit.

## Preserved evidence and its limit

The seed includes both complete Qwen model archives, Gemma's original saved
manifest and execution contract, and all 13 paired MMLU batch files unchanged.
The saved Gemma manifest records `admission_status=passed`. However, the detailed
original numeric runtime-admission and preflight receipts were not downloaded
before provider loss. That gap is explicitly recorded in the seed's
`artifacts/gemma4_31b/provenance/original-runtime-evidence.json`; no original
receipt is fabricated or replaced by a later check.

The new runtime must pass the original train-only query preflight and fresh
runtime admission before resuming. The frozen evaluation driver must regenerate
exactly the original execution-contract digest before it can reuse saved
batches. It then evaluates the remaining 10 MMLU questions, the registered
32 perplexity windows and 32 GSM8K problems. Both Qwen studies are inherited,
not recomputed. The publication of any later Gemma result must retain this
provenance limitation.

## Frozen experiment and operational changes

Scientific source SHA-256:
`ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`.
The source archive, checkpoints, input examples, precision, S1 tile/depth,
generation settings, extraction, scoring, confidence methods and margins are
unchanged. Recovery seed SHA-256:
`30ce1d1aa8be9ed642d47171a0f5375eae4278d3307a57907919276bf745b991`
(2,427,723 bytes).

The versioned operational worker changes the queue to Gemma only, initializes
the two Qwen models as complete, and publishes an index of actual admission and
preflight receipt filenames for backup. It invokes the same frozen preflight
and evaluation commands. The legacy worker remains unchanged inside the
scientific source bundle and contract; the actual operational worker has its
own separately recorded hash. This distinction is explicit in the launch
receipt and `operational-manifest.json`.

The supervisor reconciles the persistent session identity and retries one
allocation request at a time. Bootstrap and worker files are uploaded and
hash-verified before foreground execution, preserving the repair for the prior
notebook `__file__` problem. The new observer saves actual admission/preflight
receipts in addition to paired batches. It retains the TPU on errors and only
releases after all nine tasks and all three model archives are verified locally.
Temporary AC-only allocation-wait assertions do not change permanent power
settings or override intentional sleep.

## Validation and reproduction

Seven checks passed against the final operational sources. They cover exact
restoration, duplicate-output rejection, changed hashes/contracts/order,
incomplete pairs, forbidden paths, explicit receipt-gap handling, unchanged
scientific commands, launch identity, and refusal to score after failed
preflight. The actual saved-data seed builder also verified original inputs,
source-file hashes, both completed archive receipts and all saved batch
contracts. Commands, source snapshots and logs are in `validation/` and
`seed-preparation/`. These are recovery checks, not TPU performance results.

Run the checks locally from this directory with Python:

```sh
python validation/source/tools/check_accuracy_gemma_recovery_v001.py
```

`prior-runtime-loss/` preserves the observed failure and original client
receipt. `initial-allocation-v001.jsonl` preserves the failed provisioning
attempt. `active-recovery-at-publication.json` is a historical handoff snapshot,
not a claim about the current allocation. The complete recovery protocol is in
`operational-source/protocols/LLM_ACCURACY_GEMMA_CHECKPOINT_RECOVERY_v014.md`.

This folder is additive. It does not replace earlier results, failures, source
snapshots or the complete Qwen publications.
