# Exact-source recovery after runtime loss

Colab stopped listing the running TPU at 00:55:57 UTC on October 20, 2026. An independent allocation query confirmed no assignments. The controller did not request a release. This occurred about twelve hours after allocation; the provider supplied no definitive reason.

Five of the nine model-task evaluations were complete. Qwen3-14B was fully archived; Qwen3-32B had completed MMLU and perplexity, plus two of four GSM8K batches (16 paired examples). The third GSM8K batch was interrupted and has no complete checkpoint. Gemma4-31B had not started. Existing published results remain unchanged.

## Preserved execution and recovery

The scientific archive is byte-identical to the repaired v008 bundle: SHA-256 `ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`. The separate operational bootstrap installs the same pinned environment, restores saved files, and synchronously calls the original byte-verified repaired worker. Fresh training-only validation and runtime admission are required; the evaluator also rejects any change to the saved execution contract before reusing a batch. No model math, checkpoint, precision, prompts, examples, stopping, extraction, or statistical criteria were changed.

The seed SHA-256 is `e0a58841017ad7a4e369716c94342c12feffed438022c61745e870a9ee0bb72e`. It contains the original completed Qwen14 archive, Qwen32 manifest and execution contract, 21 completed paired batch files (15 MMLU, four perplexity, two GSM8K), and original successful admission provenance. A new admission is recorded separately on replacement hardware. Remaining work is Qwen32 GSM8K batches three/four, then all Gemma evaluations.

Eight local recovery checks passed, covering exact byte restoration, duplicate refusal, changed source/checkpoint/contract rejection, unsafe-member rejection, inherited archive and frozen worker verification, synchronous dispatch, failed-install retention, allocation ownership, observer CLI and the actual prepared seed. These are orchestration checks, not new TPU accuracy measurements. Operational code commit: `ac16fb772`; validation archive commit: `3d6175196`; recovery launch preservation: `c45ced1b1`.

## Allocation state at preservation

The first replacement request returned HTTP 502. Its progress-logging thread also encountered a relative-path error, preserved in the original log; the main request completed with the recorded HTTP response. The validated supervisor uses absolute paths and retries only one reconciled request at a time. At the 01:36 UTC check, there was still no allocated TPU. The laptop slept from 01:18:55 to 01:35:00 UTC, pausing the local allocation client; the client resumed on wake. This later sleep does not explain the preceding TPU loss.

No new scientific completion is claimed by this recovery snapshot. Retain the allocated TPU on experiment errors; release only after all nine evaluations and all three model archives are locally hash-verified, or an explicit user instruction. Future results are added separately.

## Files

- `source.tar.gz` and `source-manifest.json`: original repaired scientific bundle.
- `resume-seed.tar.gz` and `seed-receipt.json`: immutable completed data and provenance for recovery.
- `operational-source/`: separate bootstrap, supervisor, observer, validation code and protocol.
- `validation/`: archived local recovery checks and their exact source.
- `runtime-loss/`: original loss and ownership evidence.
- Allocation receipts preserve the failed request and later sleep/wake diagnosis.
- `artifact-manifest.json`: hashes and sizes of all other files in this snapshot.
