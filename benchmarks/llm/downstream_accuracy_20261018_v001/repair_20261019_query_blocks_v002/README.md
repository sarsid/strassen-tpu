# Shared-query-block repair for pending downstream accuracy evaluations

The original Qwen3-32B Native runtime-admission failure is preserved separately in `diagnostics/qwen3_32b_admission_20261019_v001`. No Qwen32 or Gemma test example was scored before this repair. Completed Qwen3-14B results retain their original source and execution contract.

## Implementation change

The evaluator now invokes the same compiled decoder on16 query tokens at a time for both prefill and cached decoding. Prompt blocks are processed in causal order, carrying KV state. Short decode blocks use future-position padding excluded by causal and sequence-length masks. Both Native and Strassen use this layout. The original projection functions, S1 tile128/256/256, BF16 boundaries, DEFAULT precision, weights, attention, normalization, activations, tokenizer, prompts, batch8, generation budget, stopping, extraction and registered statistical analysis remain unchanged. This is an accuracy-evaluator repair, not a timing or tuning result.

The Native-only diagnostic matched cached and full-prefix execution bit-for-bit at every layer and at logits for64,129 and1024 training tokens. The repaired actual two-arm engine is independently checked before downstream scoring; results from those checks are stored separately from test scores.

## Validation and preservation

Six local checks passed: Qwen variable-length/block-boundary behavior, Gemma local/global attention, independent histories and padding, no Qwen14 rerun, failed-preflight retention/no scoring, and preflight event handling. Earlier reduced-Gemma fixtures failed their configuration checks and are preserved. Their corrections did not change the scientific engine. A preflight event-dispatch issue was found before remote launch; the initially prepared v007 bundle was never uploaded or executed and is preserved as superseded preparation. The v008 source bundle contains the corrected, tested preflight. Four core executed source files are byte-identical to the final local validation snapshot.

The frozen v008 bundle SHA256 is `ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`. Every file from the original source bundle remains byte-identical; the repair adds new versioned files. Each validation receipt includes its original source.tar, so source trees can be reconstructed without duplicated loose files.

## Execution scope

Continuation v008 launched on the existing retained v6e endpoint at19:56UTC on October19. It reuses the existing environment and canonical checkpoint cache. Only Qwen3-32B and Gemma4-31B are scheduled. Each must pass Native cached/full-prefix L2<=0.005 and finite two-arm logits on64/129/1024 training tokens; the test process then repeats the original64-token admission. Native-versus-Strassen quality is not an admission filter. Failures retain the TPU and caches. Qwen14's completed archive is inherited bit-for-bit under its original provenance. Release requires all9 evaluations and all3 locally hash-verified model archives.

This snapshot documents the repair and validation, not completed Qwen32 or Gemma downstream accuracy. Those results will be added separately.

Actual Qwen32 two-arm preflight completed successfully: Native cached/full logits match exactly at64,129,1024 tokens; S1 cached/full logits match exactly at129 and1024 tokens, and both arms remain finite. These compare each algorithm with its own recomputation, not Native against S1 accuracy. Source hashes were verified against the frozen bundle. The queue automatically advanced to the evaluation process after this gate.
