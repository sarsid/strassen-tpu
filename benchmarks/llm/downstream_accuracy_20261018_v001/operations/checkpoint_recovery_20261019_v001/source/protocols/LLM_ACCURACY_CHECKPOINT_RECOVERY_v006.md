# Restore completed accuracy tasks after runtime loss

The alternate account obtained v6e at 04:51 UTC on October 19. Its worker was
observed performing GSM8K at 05:12:24, after completing MMLU and perplexity.
At 05:12:55 the controller no longer found that allocation. Subsequent checks
agreed; the observer recorded runtime_lost and the supervisor exited. There
is no evidence of a deliberate release or a captured inference exception.
The provider-side cause remains unknown. No completed GSM8K batch was saved.

All 15 MMLU and four perplexity batches were downloaded. Preserve the original
run and the previously published first-80-question snapshot. Verify the full
saved tasks against their original inputs, execution contract, source bundle,
passed admission and each batch's order and contract hash. Report only these
complete tasks using the unchanged registered analysis; retain the original
full manifest alongside an explicitly task-filtered analysis manifest.

The recovery uses a fresh run ID after independently confirming assignment
absence. It uses the alternate account and the exact original scientific
source bundle, SHA-256
162116e4ef16f4f7b24a713211e5d7b70c91b739b42d7a3d88734ae097bc091d.
Only host orchestration and the remote launch wrapper change. The wrapper
restores a hash-verified archive containing the original manifest, execution
contract and complete paired batch files. No summary, executable, checkpoint
weight, credential or other file is accepted from this archive. It then
launches the original worker from the frozen source bundle.

The original driver regenerates its execution contract and requires exact
equality, performs fresh train-only runtime admission, and reuses saved batch
results only if the contract and example order match. It does not rerun the
completed MMLU/perplexity tasks. Different software, hardware identity or flags
cause recovery to fail rather than silently mix conditions. GSM8K restarts at
its first incomplete batch. Both scientific arms and all registered tasks,
checkpoints, subsets, prompts, precision, generation budget, margins and
analysis methods remain unchanged.

Retain allocated resources on recoverable errors. Continue checkpointing and
monitoring. Release only after all nine task evaluations and three model
archives are locally verified. Publish additive snapshots and failure receipts.
