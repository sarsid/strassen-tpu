# Gemma 3 1B implementation qualification

Implement Gemma in new versioned source files, preserving all executed Qwen,
Mistral and MM code. Model revision:
`google/gemma-3-1b-pt@fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`.
The official reference is Transformers 4.56.2, eager attention, BF16 CPU,
evaluation mode, no KV cache. The JAX CPU path is a correctness check, not a
TPU timing result. TPU qualification and larger-shape timings remain separate.

Independent tasks: kernel agent implements the adapter; protocol agent writes
official-model comparisons and qualification harnesses; runtime agent reviews
model semantics; the primary agent manages inputs, environments and execution.
All executed sources and results receive immutable snapshots and commits.
The scoped archive runner stages only its own execution directory so it cannot
accidentally commit another active campaign's files or working changes.

Before timing, check a deterministic tiny official Gemma model with local/global
attention and sequences exceeding its window; then check the pinned real
checkpoint using an official 64-token full-model reference and a 529-token
real-weight layer check. Exercise cubic and one-level Strassen in Pallas CPU
interpretation, including padding. CPU interpreter checks establish algebraic
behavior, not compiled TPU correctness or performance.

Freeze qualification limits before execution: finite outputs, relative L2 at
most 0.03, mean KL at most 0.01, absolute mean NLL difference at most 0.05,
and top-1 agreement at least 0.90 for full-model reference comparisons. Primitive
and layer checks use stricter explicit harness tolerances. Preserve failures;
do not relax limits after observing outcomes. These initial implementation
qualification limits do not certify arbitrary inputs or downstream task quality.

Checkpoint files and authentication tools live in ignored `.runtime_private/`;
archive model revisions, file hashes and public request provenance instead of
credentials or gated checkpoint payloads. The existing Hugging Face credential
is read only by the official-source acquisition process. No credentials appear
in source, command arguments, logs or committed artifacts.
