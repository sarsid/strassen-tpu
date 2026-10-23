# N9: real-model quality passes, mixed timing results

Both frozen custom policies pass the registered corpus-quality gates on Qwen3-0.6B and Mistral-7B-v0.3. Strassen's resident-layer mean is **2.81% slower than native on Qwen** and **9.56% lower on Mistral**. Transfer-inclusive full-forward means remain close: Strassen is 0.59% higher on Qwen and 0.028% lower on Mistral. There are only three streamed repeats per policy; these are descriptive differences, not statistical speedup claims. Gemma was access-blocked and has no measured result.

The run completed with six successful model-policy outcomes and one retained access blocker. The machine audit passed, and independent reconciliation found no missing quality windows, timing rounds, altered thresholds or provenance mismatch.

Run: `20260919T081646Z-N9-v5e-v004-0fadc6`; archive commit `0184622a2c510f06834a246d7103e5092276e37e`.

Audit: `20260919T083449Z-N9-evidence-audit-v002-358ac4`; commit `cbc985bd0c56adac01101b28110cd147b0275d8c`.

## What was evaluated

This study uses actual official checkpoint weights and natural WikiText-2 raw test text. Each model scores the first 32,768 tokens produced by its official tokenizer in 32 separate 1,024-token windows: **32,736 next-token positions per policy**. Policies propagate their own states through every layer. Context resets at window boundaries. The source text is identical across models, but tokenizers differ, so these are not identical textual spans and their perplexities should not be used to rank the models.

Native attention and the vocabulary head remain in every policy. The two custom policies replace the MLP gate/up and down paths using N8's frozen complete-call selections at M=1024. Both models use the same selected configurations:

| Policy | Gate/up path | Down plus residual path |
|---|---|---|
| Native | Native XLA | Native XLA |
| Cubic | Full cubic, plain, unfused epilogue | Quadrant cubic, interleaved, fused residual |
| Strassen | One-level Strassen, interleaved, unfused epilogue | One-level Strassen, interleaved, fused residual |

All custom tiles are (BM,BN,BK)=(1024,1024,512). No early-finalization variant is used. Operands and model boundaries are BF16, multiplication accumulates in FP32, and dot precision is DEFAULT. The selected paths were frozen before N9 quality observations; no policy was changed using these outcomes.

## Native implementation qualification

Before corpus scoring, each native JAX implementation was compared with an independent official Transformers CPU BF16 forward pass on the first 64 tokens, scoring 63 next-token positions. The fixed gate requires finite outputs, logit relative L2 ≤0.03, mean KL ≤0.01, absolute NLL difference ≤0.05 nats/token and top-1 agreement ≥90%.

| Model | Logit relative L2 | NLL difference | Mean KL | Top-1 agreement | Gate |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 0.023991 | +0.002593 | 0.002169 | 96.8254% | Pass |
| Mistral-7B-v0.3 | 0.008229 | −0.004561 | 0.000679 | 96.8254% | Pass |

Final-hidden relative L2 is also recorded—0.022643 for Qwen and 0.016926 for Mistral—but is not an additional qualification gate. This short check establishes agreement within the specified tolerance, not exact equivalence over every context. Full-corpus custom-policy quality below is measured against qualified native JAX, not a full-corpus CPU Transformers reference.

## Corpus quality

The frozen corpus gate requires finite outputs, **absolute NLL difference ≤0.01 nats/token, mean KL ≤0.02 and top-1 agreement ≥97%**. Logit relative L2 is reported diagnostically and is not a corpus gate. All four custom-policy outcomes pass.

| Model and policy | NLL difference vs native | Mean KL | Top-1 agreement | Logit relative L2 | Perplexity |
|---|---:|---:|---:|---:|---:|
| Qwen — cubic | +0.0002394 | 0.0013447 | 97.9900% | 0.018184 | 23.217537 |
| Qwen — Strassen | +0.0002256 | 0.0016817 | 97.7273% | 0.021081 | 23.217215 |
| Mistral — cubic | +0.0000258 | 0.0002660 | 99.1752% | 0.005725 | 6.141676 |
| Mistral — Strassen | +0.0002972 | 0.0010361 | 98.9431% | 0.013670 | 6.143343 |

Native Qwen NLL/perplexity is 3.14466846 / 23.21197851; native Mistral is 1.81507180 / 6.14151711. Native compared with itself has zero KL/error and 100% agreement by construction; the independent native qualification is the preceding check.

The passes are tolerance-based, not identical predictions. Strassen changes the top-1 token at 744 of 32,736 Qwen positions and 346 Mistral positions. Its observed maximum absolute logit errors are 6.1875 and 8.03125 respectively. The aggregate gates do not bound every token's error or establish downstream task accuracy. Known N4 cancellation failures are not superseded by these real-model passes.

## Timing scopes and complete coverage

Resident measurements time a synchronized, compiled **complete transformer layer** with weights already on the device. Every policy receives the same native incoming state from the first corpus window at each layer. There are two warmups and seven timed rounds per layer: 28×7=196 samples per Qwen policy and 32×7=224 per Mistral policy, **1,260 resident samples total**. Layers are different workloads, not independent statistical replicates.

Streamed measurements time a full forward pass on the first 1,024-token window, including checkpoint reads, host layout/copies, device transfers, all layers, final normalization, vocabulary projection and host-loop overhead. Downloads, tokenization and compilation are excluded. Each policy has an untimed full-width warmup followed by **three measured forwards: 18 samples total**. This is a prototype that streams weights, not a fully resident production-serving benchmark.

| Model | Policy | Mean resident layer, ms | Mean streamed forward, s | Streamed range, s |
|---|---|---:|---:|---|
| Qwen3-0.6B | Native | 0.578887 | 3.529799 | 3.505023–3.551368 |
| Qwen3-0.6B | Cubic | 0.602900 | 3.549702 | 3.541806–3.555336 |
| Qwen3-0.6B | Strassen | 0.595163 | 3.550673 | 3.549576–3.551895 |
| Mistral-7B-v0.3 | Native | 3.456635 | 55.108262 | 55.066431–55.158384 |
| Mistral-7B-v0.3 | Cubic | 3.263295 | 55.039892 | 55.006848–55.072589 |
| Mistral-7B-v0.3 | Strassen | 3.126094 | 55.092791 | 55.088434–55.095426 |

On Qwen, both custom resident means are higher than native: cubic by 4.15%, Strassen by 2.81%. On Mistral, they are lower by 5.59% and 9.56%, corresponding to native/custom ratios of 1.059× and 1.106×. Strassen's resident mean is below cubic's in both models, with cubic/Strassen ratios of 1.013× and 1.044×. These are descriptive ratios of recorded means, without new significance tests.

That ordering does not yield a broad full-forward gain. Qwen's custom streamed means are about 0.56–0.59% higher than native. Mistral's cubic and Strassen streamed means are only 0.124% and 0.028% lower. Three observations cannot establish a reliable benefit or equivalence. The streamed and resident scopes differ substantially; N9 contains no timing breakdown that assigns the gap to a particular transfer, disk or host cost. The result field `eligible_for_speedup_claim: true` means the quality/completion prerequisites were met, not that a speedup was observed.

## Input repairs and preserved evidence

Qwen uses its existing v002 tokenization and official-reference artifacts unchanged. Mistral tokenization initially failed because the isolated model-tools environment lacked protobuf. The new v003 environment retains every prior installed package/version and adds only protobuf 6.32.1: independently checked inventories contain the original 29 packages plus that one addition. Its diagnostic successfully loaded the tokenizer, after which new Mistral tokenization and official-reference runs completed. The core JAX environment and allocation remained unchanged.

The initial v003 input finalizer then failed on an incorrect assumption that the older preparation adapter emitted a canonical artifact seal. That failed execution is retained. Version v004 recognizes the actual outer-execution plus child-preparation seals, validates all covered bytes, and preserves the original Qwen/Gemma entries and failed steps. The successful finalization is `20260919T081552Z-N9-input-finalization-v004-a8f5bc`, commit `30bd41ceaf6980d66df2346ce047a4d808280651`.

The frozen N9 config points to the exact official checkpoint, token and reference manifests; both models use the same corpus text digest. Its N8 policy artifact SHA256 remains `b178f8b5daa9ed8a4d3822e2bd1f6e70c1b79928db689f0cffa51bbd28ea3f3e`. N9 config SHA256 is `dbe74551000efbbbcaeebff256677c5dcd965dac34b563e74bfdac48ff60ba0b`; final journal SHA256 is `7da0b1d48278eb9da8ec520c8358924f5be28eb8c64ab9d23ceed13a100316f1`.

The N9 environment identity equals N8's selection identity on logical allocation `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, with JAX/JAXlib 0.7.2 and libtpu 0.0.21.1. No timing cohorts were pooled. Gemma3-1B remains explicitly `blocked_access`: the official access probe denied the available credentials, and no model quality or latency was measured for it.

## Audit and interpretation limits

Independent read-only inspection verified the canonical/source/outer hashes and committed manifest, exact thresholds and frozen input/policy links. It reconciled all 64 window records with per-model totals—counts, NLL/KL sums, top-1 counts, squared errors, finiteness and maximum errors—and checked all 1,260 resident and 18 streamed samples against model summaries. Every layer/repeat key and streamed repeat 0–2 is present once per applicable policy. The 1,423-record journal contains six successful policy outcomes and the Gemma blocker, with no model-policy failures. No model outputs were regenerated for this review.

The supported result is preservation of the registered quality tolerances on this fixed real-text sample, together with model-dependent resident-layer performance. It is not a generation, decode, task-accuracy, long-context, throughput or production-serving result. Two model checkpoints, one corpus prefix per tokenizer, one v5e allocation and three streamed repeats limit generalization. The short official reference and full-corpus native comparison have distinct scopes. Fresh-allocation and v6e replication remain separate later experiments.

No benchmark/test workflow was executed, no existing source/result file was edited and no commit was made for this review.

Evidence: [N9 summary](../runs/20260919T081646Z-N9-v5e-v004-0fadc6/artifacts/summary.json), [Qwen model summary](../runs/20260919T081646Z-N9-v5e-v004-0fadc6/artifacts/model-00/model_summary.json), [Mistral model summary](../runs/20260919T081646Z-N9-v5e-v004-0fadc6/artifacts/model-01/model_summary.json), [raw journal](../runs/20260919T081646Z-N9-v5e-v004-0fadc6/artifacts/results.jsonl), [frozen N9 configuration](../configs/generated_n9_v001/campaign_applications_n9_v001.json), [successful input bundle](../plans/n9_inputs_final_20260919_v004/input_bundle.json), and [machine audit](../runs/20260919T083449Z-N9-evidence-audit-v002-358ac4/artifacts/audit.md).
