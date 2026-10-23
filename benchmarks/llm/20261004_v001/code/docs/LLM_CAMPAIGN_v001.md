# Multi-model v6e LLM experiment, version 1

## Scope and sequence

1. Resolve immutable checkpoint/config revisions for Qwen3 8B/14B/32B, Mistral 7B-v0.3/Small-24B-Base-2501, Gemma3 12B/27B PT text. Verify checkpoint access and geometry.
2. Qualify DEFAULT-precision shared cubic/S1/S2 epilogues on v6e, then validate each model adapter against pinned Transformers 4.56.2 / PyTorch 2.8.0 BF16. Cover Gemma local/global attention, non-unit normalization weights and Mistral's explicit head dimension.
3. Run a bounded pilot on real layer weights and real token-derived activations; record compilation, download, packing, transfer and synchronized compute separately. Use this to estimate sweep cost.
4. Tune the five arms per model/workload/store using calibration activations. Freeze choices and all offered/pruned/failing candidates. Confirm whole-layer choices; isolated projection wins do not suffice.
5. Run independent randomized paired confirmations and held-out quality. Save raw timing samples and errors, then bootstrap paired speed ratios.
6. Measure complete streamed prompt forward, including recurring transfers, embedding, all layers, final normalization/logits and consumer casts; report resident compute separately. One-time downloads/compilation are separate. Apply identical transfer policy to all arms.
7. Measure downstream accuracy and validate a KV-cache implementation before secondary generation timing. Then B1/B8, prompts 512/2048, 128 new tokens; prefill/decode/total separate.
8. Commit results, source hashes, profiles and failure evidence after each bounded stage; release owned runtime after archival or bounded failure. v5e remains deferred.

## Comparison grid

Ten prefill workloads: B1/S512,1024,2048,4096; B4/S512,1024,2048; B8/S512,1024,2048. Seven models × ten workloads × five algorithms × two output stores = 700 main entries. Qualification, retries and pilots do not increment this count.

Default Native, tuned Native, tuned cubic, S1 and S2 are distinct arms. Default Native has no tuned compiler options. Tuned Native searches whole-function VMEM settings; cubic receives equal fusion opportunity. Retain the exact adopted parent Qwen32B B8/S1024 BF16 path as a separately identified anchor. Parent-only cubic and our matched cubic are not interchangeable.

All matrix products use BF16 inputs, DEFAULT dot precision and FP32 accumulation. BF16/FP32 are separate projection-store experiments; model consumers remain BF16 and conversions are timed. Accumulator epilogues may change BF16 intermediate rounding; record that contract and compare quality to the unchanged Native model. Do not erase the older strict-rounding or HIGHEST experiments.

## Tuner decisions to retain

Use v6e-aware candidate geometries, emphasizing 256-wide MXU leaves. Search shorter K panels and full contraction, one/two input buffers and product/output accumulators at depths 1/2. Record actual leaf sizes, padding, estimated VMEM, compiler outcomes and measured time. Estimates prune candidates but are not proof of infeasibility. Include the parent profile where compatible. Gemma post-projection RMSNorm requires a full output-row tile or an explicitly timed unfused fallback. Mistral has no Q/K normalization.

Store every selected site algorithm, tile, depth, layout, fusion, rounding, pipeline buffers, VMEM/compiler flags, precision and weight preparation. Native fallbacks are explicit: a mixed policy is not a pure Strassen result. Confirm the composite layer before promoting an isolated winner. No general-purpose selector is required.

## Quality and data independence

Freeze token IDs, tokenizer revision, corpus revision and hashes. WikiText-2 raw v1 train supplies calibration; test supplies held-out scoring. Keep parent fixtures as a separate replication set. Proposed downstream set: 200 deterministic HellaSwag validation examples with length-normalized option likelihoods, no tuning on this set; report paired accuracy and disagreements. Pin its dataset revision before execution.

Native qualification: finite outputs and per-layer relative L2 ≤ 0.02 against the official BF16 implementation; failures block that model's tuning. This is a numerical gate, not a claim of bitwise equivalence across hardware. Tiny synthetic model fixtures check semantics only; actual checkpoint qualification is required before quality claims.

Candidate eligibility: finite values, projection relative L2 ≤ 0.10 on calibration; final held-out absolute ΔNLL ≤ 0.01, mean KL ≤ 0.02, top-token agreement ≥ 0.97. Publish every failure and magnitude, including projection/hidden/logit relative L2, max absolute, RMSE, normalized max error, NLL/PPL/KL and token agreement. Thresholds are prospective criteria, not measured facts, and may not be relaxed after seeing results without a new protocol. Downstream accuracy is reported with paired uncertainty, not silently used to tune.

## Recovery and live log

Append events and atomic status snapshots. Distinguish preparation, qualification, pilot and main measurements. Retry transient transport failures up to twice; preserve attempts. Deterministic compilation/numerical failure rejects a candidate and proceeds to independent candidates/models. Missing valid selections block dependent evaluation. Never replace failed entries with Native under another label. Preserve a full matched comparison within one runtime; re-run interrupted comparisons instead of pooling partial samples across chips.

This version starts with qualification and a bounded pilot. The 700-entry sweep and cache-decoding stages remain gated until their runners and correctness checks are ready. The dashboard must show that explicitly.
