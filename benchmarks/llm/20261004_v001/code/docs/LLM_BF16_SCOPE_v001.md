# BF16 output scope, October 3

The user narrowed the remaining LLM campaign to BF16 output. The active study
now contains seven models, ten prefill workloads and five algorithms: 350
comparisons. Preserve the original 700-entry plan, all completed FP32 results,
and partial FP32 evidence. FP32 is out of scope for future scheduling, not a
failed BF16 comparison. Existing BF16 entry identities remain unchanged.

Keep DEFAULT arithmetic, existing BF16 storage boundaries, accumulator types,
15 randomized confirmation rounds, three warmups, screening budgets, independent
qualification and all quality gates. Accepting a small precision tradeoff does
not establish a new numerical threshold; continue recording perplexity/NLL,
KL, prediction agreement, logit errors and hidden-state errors as before.
BF16 output does not imply BF16 accumulation. This is a research scope choice,
not a finding that BF16 always wins for every model or algorithm.

Qwen3-14B B1/S512 already has a complete saved BF16 group. Stop its unfinished
FP32 measurement process, retaining its owner so the normal archive/download/
release path runs. Do not promote partial timing rounds as completed results.
If FP32 completes before the stop arrives, retain those results as historical.
Resume the existing model order with the qualified cached model-session worker.

Queue v007 filters remaining output groups through each job's
`active_output_dtypes`. Original `groups` remain intact for provenance and the
existing archive validator. Counts and completion use only authorized IDs;
historical FP32 entries cannot inflate progress or prevent completion. Every
future session job explicitly passes `--output-dtypes bfloat16` to the unchanged
qualified runner. The effective BF16 protocol is archived separately; the old
two-store configuration remains the capability/identity reference used by the
existing validated runtime, with its supported explicit store-subset argument.

Keep the weight/layout cache and separate resident pass. The first workload in
each new model session still runs the cached/original equivalence pilot for its
selected BF16 store. No v5e, S3/S4 or new workloads are added.
