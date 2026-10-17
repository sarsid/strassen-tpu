# Qwen3-32B: four algorithms across batch sizes

Authorized October 14, 2026. Run one LLM first; no Gemma, Qwen14 or v5e queue.
Use the pinned parent kernels and numerical semantics. This is a versioned
extension of the reproduced parent experiment, not an unchanged reproduction.
The user specifically adds Tuned Native and requires four algorithms.

## Workloads and controls

One v6e, Qwen/Qwen3-32B revision 9216db5781bf21249d130ec9da846c4624c16137,
BF16 output, DEFAULT operand precision with the parent's FP32 accumulation and
rounding boundaries. S=1024; B=1,4,8,16,32,64. Run B8 first to connect with the
reproduction, then B16/32/64, then small controls B4/1. Large-batch memory failures
are results, not permission to silently reduce B, alter attention, or mix timings.
No padding study, S2, decode, or other model is included.

Native uses the parent native layer with default XLA memory settings. Tuned
Native uses that same computation and selects scoped VMEM from default and
32/48/64/96/112/128 MiB using the complete layer, two warmups and five samples.
The parent 48 MiB setting is explicitly included; this distinction is logged.
Tuned Cubic and Tuned Strassen use the parent block implementations, gate/up
tuners and O/down routing with independently selected tiles. Custom blocks retain
the parent's 48 MiB XLA scope; their Pallas limits remain 124/120 MiB.
Strassen is the parent's one-level product-aware fused SwiGLU and fused Q/K
normalization/RoPE path. V remains native. Fused Q/K keeps the parent's tile,
reducing only BM to 1024 for B1. Do not describe this fixed Q/K choice as tuned.

## Tuning and frozen confirmation

Use the parent's exact gate/up candidate axes, memory estimates and two-warmup,
five-sample mean selection. Its tile screen uses seeded synthetic operands;
record this honestly. Confirmation uses official weights and real propagated
activations. The site screen covers the deployed O/down sites; screen Cubic
independently as well as Strassen. O is timed as a plain projection, matching
its actual parent use; down includes the actual residual epilogue. This corrects
the original site's O residual mismatch for this new extension and is logged.
Keep parent native V and published fused Q/K rather than tuning unused Q/K/V
plain GEMMs. Archive kept/pruned candidates, compile failures, every sample,
and selected tiles. Freeze all profiles before confirmation; never retune on
the confirmation or held-out quality result.

Each measurement invocation gets a new output directory. Stage failures get at
most one fresh-process retry with both attempts preserved. Numerical gate
failures are retained without retrying to obtain a pass. A failed mandatory
stage blocks dependent stages for that batch; the worker continues other
batches. Complete four-arm groups are the unit of comparison.

## Three separate measurement scopes

1. **Resident projections:** Q, K, V, O, gate/up+SwiGLU, and down+residual on
   layer-0 native-path activations and real weights. All arms receive identical
   inputs. Measure three warmups/twenty samples, keep errors versus native, and
   include the fused postoperations used by each arm. Fixed native V is labeled.
2. **Transformer computation (primary):** full block-0, then all 64 layers with
   each arm's own propagated activations. Block uses three warmups/twenty samples;
   the layer sum uses two warmups/five samples per layer. Rotate/reverse arm order
   as in the parent. Transfers, compilation, embeddings, final norm and vocabulary
   scoring are excluded. Device-memory traffic remains inside kernel timing.
   Report absolute milliseconds, time saved and ratios versus both Native and
   Tuned Native. The sum of individually warmed layers is not uninterrupted
   model latency. Retain the parent's bootstrap over layer differences and all
   raw samples; its interval describes this procedure, not deployment variation.
3. **Offloaded prefill (secondary):** separately time complete serialized
   prefill through all 64 layers, including cached-checkpoint reads, recurring
   weight transfer/packing, embeddings, final norm and last-position logits.
   The checkpoint is cached locally before this stage; downloads and compilation
   are excluded. One full warmup and three complete passes per arm, rotating
   order. Use the same common prepared-weight loader for all arms; log that it
   includes shared Strassen packing even for native. This is the benchmark's
   specified common offload pipeline, not an optimized native serving system.
   All-token logits are computed separately for quality, never charged to
   generation prefill. This scope does not measure decode or KV-cache serving.

## Accuracy and reproducibility

Preserve repetitive text checks and four contiguous WikiText-2 test segments,
with pinned parent dataset/tokenizer revisions and token hashes. Repetitive
prompts cycle the parent's eight strings to fill B, so B8 is the original corpus.
For the natural test, never repeat tokens to conceal an undersized corpus.
Record finite checks, maximum absolute error, RMSE, relative L2, relative max
error, NLL, perplexity, logit drift, KL and top-1 agreement. Apply the parent's
natural thresholds (abs NLL <= .01, KL <= .02, top1 >= .97) to every candidate,
including Tuned Native; retain the stricter repetitive thresholds separately.
Report failures alongside speedups. Quality is relative to the matched Native
path; an official Transformers model-equivalence claim is outside this study.

Pin JAX/jaxlib 0.7.2, libtpu 0.0.21.1 and the original parent tokenizer stack.
Archive original parent source hashes, new wrapper source commit, environment,
tokens, profiles, memory/compile diagnostics and raw outputs. Never modify
executed sources or earlier results. Commit every completed or failed batch's
archive locally as it is downloaded. Update the live journal throughout setup,
downloads, candidate screening, compilation, each layer and quality scoring.
Release the owned TPU after this study or a terminal controller failure.
