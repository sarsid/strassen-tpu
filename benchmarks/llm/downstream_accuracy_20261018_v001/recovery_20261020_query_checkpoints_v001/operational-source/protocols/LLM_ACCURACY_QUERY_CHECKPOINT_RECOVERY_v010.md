# Recovery of the repaired accuracy study

On October 20 at 00:55 UTC, Colab stopped listing the owned runtime. A separate
assignment query confirmed no allocations. No release was requested by the
controller. The disappearance occurred about twelve hours after allocation;
the provider did not supply a definitive termination reason.

Recover in `20261020-accuracy-study-v009` using the exact v008 scientific archive
SHA `ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`.
Do not rebuild that bundle or substitute the original pre-repair evaluator.
The recovery bootstrap and acquisition supervisor are versioned and archived
separately; neither changes the measured model code or statistical protocol.

Restore the original Qwen3-14B archive unchanged, with its original provenance.
Restore Qwen3-32B's 15 MMLU batches, four perplexity batches and two complete
GSM8K batches, together with their original manifest and execution contract.
The interrupted third GSM8K batch has no complete checkpoint and is recomputed.
The original passed admission remains under a provenance subdirectory; a fresh
admission is required on the replacement hardware before any new test scoring.

The bootstrap installs the same pinned environment into a private directory,
then synchronously invokes the byte-verified v008
`accuracy_query_block_recovery_v001.worker`. That exact worker performs the
training preflight and invokes `accuracy_study_v002.py`, whose existing strict
contract comparison rejects source, hardware/software or numerical-setting
changes before reusing a saved batch. Only Qwen3-32B then Gemma4-31B execute.

`resume_accuracy_query_remote_v001.py` rejects unexpected seed members, content
hash changes, a changed completed-model archive, changed batch contracts and
duplicate run directories. `start_accuracy_study_v006.py` reconciles one named
allocation, persists capacity retries, observes the genuine foreground worker
and retains the machine on errors. An AC-only process sleep guard is optional;
persistent system power settings are unchanged.

Preserve all original checkpoints and failed/interrupted attempts. Publish new
whole-task results separately. Release only after all nine evaluations and all
three model archives are locally hash-verified, or a later user instruction.
