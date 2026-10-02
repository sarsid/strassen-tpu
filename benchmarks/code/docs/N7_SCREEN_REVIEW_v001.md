# Independent N7 held-out screening review

The N7 screen preserves the frozen selector and follows the registered candidate-selection rules. Independent inspection found no provenance, selection-eligibility or candidate-retention issue that blocks the separate N7 evaluation. This screen tunes the independent comparator implementations on reserved geometries; it does not train or update the already frozen selector.

Run: `20260919T065405Z-N7-screen-v5e-v004-6ef489`.

Archive commit: `6ee015900da6d4efed2fb8031b52318b7b83599a`.

Source commit: `ebb810d6209eec399a56e962694bb72f0fdce8f0`.

Selection SHA256: `5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37`.

Journal SHA256: `3a465c188b13d5bde74f1e6cfc5278120957b62dbb87bec3a96322010aba82ea`.

## Selector frozen before observations

The selector's byte hash is **`18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77`**. Identical bytes were verified in the original N6a fit artifact, current stored selector, N7 source snapshot, that snapshot's source commit, and the canonical N7 artifact copy. The recorded provenance and N7 selections reference the same hash.

The original fit (`20260919T064008Z-N6a-selector-fit-v002-6ef3f3`) completed at 06:40:08.488433 UTC. N7's source snapshot was created at 06:54:05.577880 UTC. Journal sequence 1, `run_start`, recorded the selector hash at 06:54:11.505068 UTC; the first measured timing is sequence 21 at 06:54:17.490374 UTC. The runner checks and records the selector before compilation or timing. Thus the archived rule was already frozen before this screen's observations, and the supplied evidence shows no replacement or update during screening.

The rule retains 16 training prototypes with labels four Strassen, two cubic and ten native. Its provenance links the exact N5 confirmation journal, N5 screen selections, selector source and original campaign. The reserved-shape manifest retains SHA256 `e5ed73f42fdba354ef869f60081edb2f63dc43d1ae984ba6446ad5434a6dac0f`. All 16 held-out (M,K,N) tuples match the rule's reservation and are disjoint from both the 16 training tuples and the 44-shape N1–N4 manifest. LLM-shaped cases here still use synthetic Gaussian operands.

The screen, its choices and the frozen rule share the same full environment identity on allocation `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`. No cohort pooling or selector retraining is justified by these results.

## Complete attempt and eligibility accounting

All 320 groups completed. Each of 16 shapes has exactly 20 attempted cubic candidates, 20 attempted Strassen candidates and one native candidate in both scopes: 656 candidate–shape attempts and **1,312 scoped results: 1,225 `ok`, 87 `oom`**. No numerical failures, omitted candidates, duplicate results or other error statuses were observed.

Each successful result contains seven positive finite raw samples, rounds 0–6 exactly once. The 8,575 raw samples reproduce their reported arithmetic means. Recorded successful numerical metrics satisfy the unchanged finite-output, relative-L2 ≤0.02 and maximum-error ≤`0.001 + 0.05 × max_abs_reference` gate. The largest observed successful Strassen relative L2 error is 0.004444. This audit checked recorded metrics; it did not independently regenerate operands or numerical references.

Independently reconstructed two-scope eligibility and minimized `(complete-call mean, candidate_id)` within each family. All 32 selected custom candidates and 16 native entries match the artifact. Every shape has 19 eligible cubic candidates, 18 eligible Strassen candidates and one native: totals 304, 288 and 16 respectively. The nine extra successful complete-call results whose prepared scope failed are correctly excluded from selection.

## Selected custom configurations

Matrix shapes are (M,K,N); tiles are (BM,BN,BK). P = plain, O = output_accumulator, I = interleaved, IO = interleaved_output_accumulator. These are screening selections, not confirmed performance outcomes.

| Shape ID | Matrix (M,K,N) | Cubic tile; variant | Strassen tile; variant |
|---|---|---|---|
| held_square_768 | 768,768,768 | 1024,1024,512; P | 1024,1024,1024; IO |
| held_square_1536 | 1536,1536,1536 | 512,512,256; P | 512,1024,512; I |
| held_square_3072 | 3072,3072,3072 | 1024,1024,1024; O | 1024,1024,1024; O |
| held_square_6144 | 6144,6144,6144 | 1024,1024,1024; O | 2048,2048,512; IO |
| held_rect_wide | 1024,3072,6144 | 1024,1024,1024; P | 1024,1024,1024; I |
| held_rect_tall | 6144,3072,1024 | 1024,1024,1024; O | 1024,1024,1024; I |
| held_rect_short_k | 3072,1024,6144 | 1024,1024,1024; O | 1024,1024,1024; I |
| held_rect_deep_k | 3072,6144,1024 | 512,1024,1024; O | 1024,1024,512; IO |
| held_small_m16 | 16,4096,8192 | 128,512,512; O | 512,1024,512; IO |
| held_small_m128 | 128,3072,6144 | 128,512,512; P | 512,1024,512; O |
| held_tail_a | 1001,3001,5003 | 1024,1024,1024; O | 1024,1024,1024; O |
| held_tail_b | 2051,4099,1537 | 512,1024,512; P | 512,1024,512; IO |
| held_qwen_gateup | 256,5120,51200 | 512,1024,1024; O | 512,1024,512; IO |
| held_qwen_down | 1024,25600,5120 | 512,1024,1024; O | 1024,1024,1024; IO |
| held_mistral_gateup | 1024,4096,28672 | 1024,1024,1024; O | 1024,1024,1024; O |
| held_gemma_gateup | 1024,3584,28672 | 1024,2048,512; P | 1024,1024,512; IO |

Cubic selects O on ten shapes and P on six. Strassen selects IO on eight, O on four and I on four. Joint tile/variant selection does not isolate a variant's causal contribution.

## The 87 scoped VMEM failures

All failures occurred during compilation and are scoped VMEM rejections. They affect three candidate configurations across all 16 shapes:

| Family | Candidate ID | Call failures | Prepared-kernel failures | Distinct candidate–shape attempts |
|---|---|---:|---:|---:|
| Cubic | c_plain_2048_2048_512 | 13 | 16 | 16 |
| Strassen | s_plain_2048_2048_512 | 13 | 16 | 16 |
| Strassen | s_interleaved_2048_2048_512 | 13 | 16 | 16 |
| Total | Three configurations | 39 | 48 | 48 |

For each candidate, `held_square_768`, `held_square_1536` and `held_rect_tall` succeed in complete-call scope but fail prepared-kernel compilation. The remaining 13 shapes fail in both scopes. Hence there are 39 attempts with two failed scopes plus nine attempts with one failed scope, totaling 87 rows. These are 29 cubic and 58 Strassen failed scoped results, not 87 distinct parameter choices.

The errors report 81 allocations of 56 MiB and six allocations of 52 MiB against the fixed 48 MiB limit. All 48 prepared failures are 56 MiB; the six 52 MiB failures are in complete-call scope. This is not evidence of device HBM exhaustion. Output-accumulator variants using the same large tile can remain feasible, so the tile itself must not be described as universally unsupported.

## Interpretation and audit limits

The independent alternatives were tuned with equal attempted budgets on these held-out geometries after the selector was frozen. This is compatible with evaluating a fixed rule against independently tuned comparators; the alternative choices must not be copied into that rule. Fresh N7 evaluation should compare frozen-rule execution, selected cubic, selected Strassen and native together on the registered new seed and retain any numerical or compilation failures.

Screening minima are affected by selection noise. Native was timed only in the first candidate-pair group per shape; selected candidates can come from later groups. Screening ratios are therefore not fresh paired confirmation estimates. Report equal attempt counts alongside unequal feasible counts and the configured VMEM limit. Gaussian numerical passes do not remove known cancellation failures or certify real activations/model quality.

Verified root/canonical/source-manifest hashes, committed root manifests, selector source-commit bytes and selection provenance links; all checked values matched. No benchmarks were rerun, no frozen files were edited and no commits were made for this review.

Primary evidence: [selections](../runs/20260919T065405Z-N7-screen-v5e-v004-6ef489/artifacts/selections.json), [journal](../runs/20260919T065405Z-N7-screen-v5e-v004-6ef489/artifacts/results.jsonl), [selector provenance](../runs/20260919T065405Z-N7-screen-v5e-v004-6ef489/artifacts/selector_input_provenance.json), [frozen selector artifact](../runs/20260919T065405Z-N7-screen-v5e-v004-6ef489/artifacts/frozen_selector_input.json), and [original fit artifact](../runs/20260919T064008Z-N6a-selector-fit-v002-6ef3f3/artifacts/selector.json).
