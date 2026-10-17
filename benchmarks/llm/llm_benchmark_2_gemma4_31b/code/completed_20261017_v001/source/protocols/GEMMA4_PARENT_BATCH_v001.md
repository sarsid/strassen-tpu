# LLM Benchmark 2: Gemma4-31B dense text

User-authorized October 16, 2026. Use the public official checkpoint
`google/gemma-4-31B@5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`, text only.
The four arms are Native, Tuned Native, Tuned Cubic and Tuned Strassen S1.
The pinned parent kernels at `95be1fb088656a89813b04492e1d77c66b36ccf9`
remain unmodified. This is a model-specific adapter, not an exact reproduction
of an existing parent Gemma4 experiment.

## Workloads and software

Sequence length 1024; batches 1, 4, 8, 16, 32, 64. Execute 8, 16, 32, 64,
4, 1 to emphasize the larger workloads. One v6e, JAX/jaxlib 0.11.2,
libtpu 0.0.48 throughout, BF16 model storage and output, FP32 GEMM accumulation,
DEFAULT operand precision. The newer stack avoids the old-stack B64 compiler
failure. Do not pool these timings with the earlier Qwen software stack.

Gemma has 60 layers, hidden width 5376, intermediate width 21504, GELU gating,
direct learned RMS scales, a layer scalar, tied scaled embeddings, and a
softcapped output head. Five local layers followed by one global layer repeat.
Global K/V shares a raw projection; local/global head dimensions differ.
Retain all of these semantics and tune representative layers 0 and 5 separately.

## Qualification and selection

First compare official checkpoint native outputs with the independent pinned
Transformers 5.5.0 reference, including both attention types and final logits.
Synthetic tests alone are not checkpoint qualification. A qualification failure
stops performance claims and remains in the archive.

Tune each batch and layer type independently, using the parent blocked Cubic
and interleaved product-aware S1 implementations. Qwen tiles are not Gemma
profiles. Log every candidate, rejection, tile, compiler option, raw timing and
selection rule. Tuned Native varies scoped VMEM; default Native has no override.
Choose the fastest finite candidate using mean warmed timing, then freeze the
profile before held-out confirmation. Record whether each projection uses a
fused epilogue or external postprocessing. No depth 2/3/4 extension is included.

## Measurements

Keep isolated projection, complete resident block, sum of warmed resident
layer means, and serialized offloaded prefill separate. The resident layer sum
excludes transfers, embedding and output head; it is not full-device-resident
32B inference. Offloaded prefill includes recurring cached weight reads,
packing, transfers, all layers, embedding and last-position logits. Use the
same prepared-weight loader for all arms and prohibit network downloads in
timed passes. No decode or serving-throughput claim follows from these scopes.

Use real weights and native projection inputs; propagate each arm's own hidden
states through the full model. Rotate measurement order, preserve raw samples,
and use the Qwen warmup/repetition counts in the registered JSON plan. Compare
repetitive prompts and four WikiText segments; preserve token hashes, positions,
hidden/logit relative L2 and max error, finite checks, NLL/perplexity change,
KL and top-1 agreement. Keep the Qwen accuracy gates unchanged. Failed gates
remain visible and cannot be retried merely to obtain a pass.

## Operations and reporting

The dashboard at port 8767 explicitly separates setup, checkpoint qualification,
tuning, confirmation, quality, offloaded prefill and archival. Show current
batch, arm, layer type, test, candidate/tile and heartbeat age. Completed batch
counts require saved artifacts. A stale heartbeat is not a claimed failure.

Freeze source and protocol before execution. Keep bounded stage retries and
incremental stage/profile downloads; preserve failed attempts. Commit only
scoped files, release only the owned TPU and verify release. No other model or
v5e campaign is queued. Publish a new dedicated Benchmark 2 folder after data
exist; never overwrite Benchmark 1 or historical Gemma attempts.
