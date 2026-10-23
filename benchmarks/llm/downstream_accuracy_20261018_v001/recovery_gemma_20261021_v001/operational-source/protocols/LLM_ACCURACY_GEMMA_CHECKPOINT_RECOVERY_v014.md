# Gemma-only recovery after the second runtime loss

The retained v009 session disappeared while v010 was evaluating Gemma. No
release was requested. The provider cause is unknown. Local observation was
interrupted between October 21 01:25 and 05:29 UTC; the first independent check
after resumption showed no allocation. Preserve this uncertainty rather than
attributing the loss to exhausted credits, a subscription tier or a fixed limit.

Both complete Qwen model archives remain unchanged and verified. Gemma has
13 complete MMLU batches (104 paired examples), its execution contract and its
passed-status manifest saved locally. The detailed original Gemma runtime
admission and preflight receipts were not downloaded before the loss. This gap
is explicitly recorded in the recovery seed; no replacement receipt is invented.

Run v011 restores both Qwen archives byte-for-byte and only the saved Gemma
prefix. It uses the identical scientific source archive, checkpoint revision,
software pins, inputs, numerical precision, S1 tile/depth, generation settings,
scoring and confidence methods. Its Gemma-only operational worker differs from
the frozen queue in completed-model bookkeeping and receipt backup, not in
the preflight or evaluation command. The legacy frozen worker remains unchanged
inside the scientific bundle and contract; the actual operational worker has
its own recorded source hash in the launch receipt.

Before any scoring or checkpoint reuse, rerun the existing train-only query
preflight and fresh runtime admission. The frozen evaluation driver must
regenerate exactly the saved execution-contract digest before reusing any
batch. Any mismatch stops the run and retains the TPU for investigation.
Resume only the last 10 MMLU questions, then the same 32 perplexity windows and
32 GSM8K problems. Do not retune, expand benchmarks, normalize answers or
replace either completed Qwen result.

The new observer backs up actual admission and preflight receipts through the
contents API. They are named by a stable metadata index produced by the
operational worker. The original scientific receipt and event files are not
edited. Both completed Qwen archives and the final Gemma archive must be locally
hash-verified before release at 9/9 tasks.

The new supervisor adopts any single in-flight allocation request, reconciles
its persistent session identity before retries, and never overlaps requests.
Use uploaded, hash-verified bootstrap and worker files; do not repeat the prior
inline `__file__` failure. Keep one real foreground notebook execution and one
read-only observer. Retain the runtime on failures. Use temporary AC-only
idle-sleep protection while waiting, without changing permanent power settings
or overriding intentional sleep. Keep the existing fifteen-minute monitor
active; it may use shorter checks during this recovery.
