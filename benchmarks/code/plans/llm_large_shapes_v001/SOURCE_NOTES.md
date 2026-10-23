# Larger LLM-derived matrix shapes, v001

This is a **pure matrix-multiplication experiment using synthetic BF16 values**. The three larger models supply verified dimensions; their weights are not downloaded or loaded for this study. Results cannot establish full-model speed, perplexity, output quality, attention performance, or real-activation numerical behavior.

The main plan contains **18 matrix shapes: three models × three M values × two projection sites**. Use M = 2,048, 8,192, and 16,384 as numbers of flattened token rows. M does not by itself specify batch size, sequence length, or an actual attention context. Execute first on the same identified v5e allocation; v6e replication is a later, separately identified stage.

| Official configuration | Hidden width H | Intermediate width I | Concatenated gate/up (M,K,N) | Down (M,K,N) |
|---|---:|---:|---|---|
| Qwen3-8B | 4,096 | 12,288 | (M, 4,096, 24,576) | (M, 12,288, 4,096) |
| Mistral-7B-v0.3 | 4,096 | 14,336 | (M, 4,096, 28,672) | (M, 14,336, 4,096) |
| Gemma3-12B text backbone | 3,840 | 15,360 | (M, 3,840, 30,720) | (M, 15,360, 3,840) |

An individual gate or up projection has shape (M,H,I); these are two distinct logical projections sharing the same geometry. This bounded study explicitly chooses the concatenated matrix B = [B_gate, B_up], shape (H,2I), and measures its single matrix product plus a separate down matrix product. It does not apply GELU, SiLU, multiplication of gate/up activations, normalization, or a residual. Nor does it claim the official model always executes a fused concatenated projection. The down input is freshly generated synthetic data, not an activation produced by the gate/up measurement.

## Verified primary sources

Official `config.json` response bytes were read on 2026-09-21 UTC; their lengths and SHA256 values were computed before writing this plan. Qwen and Gemma revisions were resolved from the official Hugging Face model API. Mistral deliberately reuses the previously verified revision in `configs/generated_n9_v001/model-01-model_manifest.json`; its config was fetched again and its SHA256 exactly matches that preserved manifest. No model weight request was made. Existing authorized Hugging Face credentials, where available, were used only against the official provider and were not printed or stored in this plan.

| Model | Immutable revision | Config bytes | SHA256 of original config bytes |
|---|---|---:|---|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 728 | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| mistralai/Mistral-7B-v0.3 | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | 601 | `affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99` |
| google/gemma-3-12b-pt | `295efb63d01a7017928f273a94ebb86105c9526f` | 876 | `61501b2259b5efa7c5bb50a9eea9fd195c449b6e995b16bf8e3f52d3b80b2f77` |

Source links: [Qwen configuration](https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/config.json), [Mistral configuration](https://huggingface.co/mistralai/Mistral-7B-v0.3/resolve/caa1feb0e54d415e2df31207e5f4e273e33509b1/config.json), [Gemma configuration](https://huggingface.co/google/gemma-3-12b-pt/resolve/295efb63d01a7017928f273a94ebb86105c9526f/config.json).

`model_shapes.json` embeds each parsed official configuration, the immutable URL and original-byte hash, the exact extraction path, and every matrix shape. An embedded parsed object is semantically the same configuration, but its reserialized bytes are not claimed to reproduce the source-byte hash.

Qwen3-8B and Mistral-7B share H=4,096, **but their intermediate widths differ**. A Mistral-shaped matrix must not be labeled an exact Qwen3-8B gate/up or down shape. Gemma's dimensions come from `text_config.hidden_size` and `text_config.intermediate_size`, not the vision configuration. The official 12B repository is multimodal; only its dense text feed-forward dimensions are represented here. Its text configuration has linear RoPE scaling, which is outside the qualified Gemma1B adapter's supported scope. That limitation does not affect shape-only MM and this plan makes no Gemma12B adapter qualification claim.

## Fixed input and comparison scope

Use the existing `configs/distributions_v1.json` Gaussian definition: NumPy PCG64, generate A then B in float32, A ~ N(0,1)/sqrt(K), B ~ N(0,1), then quantize each once to BF16. Reuse the exact quantized operands across native XLA, cubic, and one-level Strassen within a shape/stage. The separate campaign must freeze screen/confirmation seeds and equal candidate-attempt budgets before execution. Fresh confirmation follows independent per-shape family selection. This manifest contains no observed winner and makes no selection from prior measurements.

Keep BF16 inputs/pre-adds, FP32 accumulation/output, and native precision DEFAULT consistent. Report both complete-call latency, including padding/trimming, and prepared-kernel latency. The actual Gemma dimension 3,840 must remain in the logical shape; implementation padding to 4,096 is an incurred cost, not a rewritten model dimension. Preserve every OOM, unsupported case, numerical failure, and timing failure rather than dropping cases. One Gaussian input scope does not certify cancellation-sensitive or real-model data.

## Memory bounds

The largest logical output is Gemma concatenated gate/up at M=16,384: 16,384 × 30,720 FP32 entries, or **1.875 GiB**. Its BF16 A/B plus FP32 C total is **2.2119140625 GiB** before padding, scratch buffers, retained copies, compilation, or executable memory. These figures are arithmetic estimates, not measured peak HBM or a guarantee that an arm compiles within VMEM.

The campaign/runner must preflight padded candidate estimates against its 8 GiB experiment budget, hold only the intended sequential arm buffers, use sampled reference rows/columns with all K, and keep full-output finite checks on device. No complete giant FP32 reference output is required. A failed preflight or allocation remains an explicit result. This document schedules no concurrent TPU timings and performs no benchmark execution.
