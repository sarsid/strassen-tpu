# N7: frozen selector on reserved matrix geometries

The frozen rule used a custom multiplication on three of the 16 reserved geometries, and each of those three was faster than native XLA in complete-call timing under its individual paired 95% confidence interval. It chose native on the other 13. Across all shapes, the selected route's median latency was 1.0041 times the fastest independently measured alternative; its largest observed shortfall was 7.60%. These are useful, limited results for the registered Gaussian-input scope on one v5e allocation. They do not establish a generally optimal selector or numerical safety for arbitrary inputs.

Run: `20260919T073017Z-N7-evaluate-v5e-v004-8036b5`.

Archive commit: `26f0646f5834b8e39b5e62907097e890b9d1d688`.

Journal SHA256: `757c77c09ad1016415ee4930305d6d0229af3f128ef802db9433f113cbe88153`.

Frozen selector SHA256: `18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77`.

Independent comparator selection SHA256: `5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37`.

## What was held fixed and what was tested

The rule was fit using the N5 training results and frozen before any N7 screening observations. The 16 held-out matrix tuples were reserved before N5 timing and are disjoint from the 16 training tuples and the N1–N4 shape set. N7 then screened independent cubic and Strassen alternatives using the registered equal budgets of 20 attempted candidates per family and shape. It did not add those choices to the rule. The separate evaluation used seed 20260924, fresh from the screen's seed 20260923, and timed native XLA, selected cubic, selected Strassen and the frozen rule's chosen implementation together.

All 16 groups completed: **128 successful scoped results**, comprising four arms, two timing scopes and 16 shapes. There were no numerical failures, errors or skips. Complete-call timing includes the callable's preparation and finishing operations; prepared-kernel timing reuses prepared operands. Neither scope includes compilation. The rule decision was made before these timings, as discussed below.

The selector bytes, selection link, canonical and outer archive seals, committed outer manifest, recorded environment identity and source/config provenance were checked. The selector hash remains identical to the original N6a artifact and the N7 screen. The evaluation identity equals the original rule/selection cohort identity on allocation `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, using JAX/JAXlib 0.7.2 and libtpu 0.0.21.1. These results are from the same allocation; they are not a fresh-machine or v6e replication.

## The rule's choices

The frozen rule chose **13 native, two Strassen and one cubic** implementations. Eight native choices followed a nearest training prototype without a confident custom advantage. Two used the small-dimension guard, two the padding guard, and one the outside-training-neighborhood guard.

Its registered guards remain unchanged: minimum dimensions M≥256, K≥512 and N≥512; maximum log2 nearest-prototype distance 1.75; maximum padded-volume ratio 1.5. Training labels required a lower confidence bound above 1.01. The caller must declare the validated input scope, `gaussian_synthetic`; the rule does not inspect data to establish numerical safety. Unknown inputs, cancellation cases and unvalidated real activations fall back to native.

| Held-out shape ID | Matrix (M,K,N) | Frozen choice | Training prototype |
|---|---|---|---|
| held_rect_deep_k | 3072,6144,1024 | Cubic plain; tile 1024,1024,1024 | vary_k_8192 |
| held_rect_tall | 6144,3072,1024 | Strassen interleaved_output_accumulator; tile 1024,1024,1024 | vary_m_8192 |
| held_square_6144 | 6144,6144,6144 | Strassen interleaved_output_accumulator; tile 2048,2048,512 | square_8192 |

Tiles are listed as (BM,BN,BK). All three custom choices have padded-volume ratio 1. Their decisions were independently reconstructed from the frozen prototypes and guards; all 16 reconstructed labels, choices and reasons match the artifact. No rule update follows from the missed opportunities below.

## Fresh timing comparisons

Speedup is the reference arm's arithmetic mean divided by the selector arm's arithmetic mean. A win requires the entire paired 95% interval to exceed 1; a loss requires the entire interval below 1. An interval containing 1 is inconclusive, not a demonstration of equivalence. Intervals are the registered 2,000-resample paired bootstrap of 30 rounds, with no multiplicity correction.

| Scope | Reference | Wins | Inconclusive | Losses |
|---|---|---:|---:|---:|
| Complete call | Native XLA | 3 | 12 | 1 |
| Complete call | Selected cubic | 11 | 2 | 3 |
| Complete call | Selected Strassen | 8 | 5 | 3 |
| Prepared kernel | Native XLA | 3 | 13 | 0 |
| Prepared kernel | Selected cubic | 6 | 6 | 4 |
| Prepared kernel | Selected Strassen | 6 | 4 | 6 |

The three custom choices account for all three complete-call wins over native:

| Shape | Native mean, ms | Selector mean, ms | Native / selector | Paired 95% CI |
|---|---:|---:|---:|---|
| held_rect_deep_k | 0.522628 | 0.477520 | 1.0945× | [1.0765, 1.1195] |
| held_rect_tall | 0.503111 | 0.465874 | 1.0799× | [1.0585, 1.0988] |
| held_square_6144 | 2.792191 | 2.560692 | 1.0904× | [1.0854, 1.0956] |

These are speedup ratios, approximately 8.6%, 7.4% and 8.3% latency reductions respectively. All three also have prepared-kernel intervals above 1 versus native. The rule's deep-K cubic choice is not the independently selected cubic configuration; its tall Strassen choice also differs from the independently selected Strassen configuration.

The only nominal complete-call loss versus native is `held_small_m16`: selector 0.339574 ms versus native 0.321087 ms, speedup 0.9456×, CI [0.8944, 0.9938]. Both arms execute the same native implementation. This observation must remain in the results, but it is not evidence that the rule selected inferior arithmetic. Independent measurement, order/noise effects and multiple comparisons are material cautions.

For context, independently selected Strassen itself has six wins, three inconclusive results and seven losses versus native in complete-call scope. Thus the rule avoided several poor Strassen choices while also missing some beneficial ones. This is a description of these 16 observations, not a claim of an optimal policy.

## Descriptive regret and missed opportunities

For each shape and scope, the reported ratio is selector latency divided by the minimum mean latency among the three independently measured alternatives: native, selected cubic and selected Strassen. The selector arm is excluded from that minimum. This is a comparison with a bounded set of observed alternatives, not an exhaustive oracle or a generalization bound.

| Statistic | Complete call | Prepared kernel |
|---|---:|---:|
| Median latency ratio | 1.0041 | 1.0148 |
| Arithmetic mean ratio | 1.0159 | 1.0269 |
| 90th percentile ratio | 1.0632 | 1.0814 |
| Maximum ratio | 1.0760 | 1.0965 |
| At most 1% slower | 10 / 16 | 7 / 16 |
| At most 5% slower | 12 / 16 | 13 / 16 |
| At most 10% slower | 16 / 16 | 16 / 16 |
| Family matches lowest observed mean | 11 / 16 | 9 / 16 |

Percentiles use linear interpolation of the sorted 16 ratios. Family agreement is descriptive; a lowest measured mean is not a noise-free target label.

| Shape | Frozen family | Fastest observed alternative family | Complete-call latency ratio |
|---|---|---|---:|
| held_qwen_down | Native | Strassen | 1.07600 |
| held_rect_wide | Native | Strassen | 1.06501 |
| held_rect_short_k | Native | Strassen | 1.06143 |
| held_small_m16 | Native | Native | 1.05758 |
| held_tail_b | Native | Native | 1.01949 |
| held_square_768 | Native | Native | 1.01036 |
| held_square_3072 | Native | Strassen | 1.00708 |
| held_rect_deep_k | Cubic | Strassen | 1.00640 |
| held_gemma_gateup | Native | Native | 1.00188 |
| held_qwen_gateup | Native | Native | 0.99721 |
| held_mistral_gateup | Native | Native | 0.99701 |
| held_small_m128 | Native | Native | 0.99692 |
| held_square_6144 | Strassen | Strassen | 0.99456 |
| held_tail_a | Native | Native | 0.98950 |
| held_square_1536 | Native | Native | 0.98903 |
| held_rect_tall | Strassen | Strassen | 0.98487 |

The largest complete-call shortfall is `held_qwen_down` (1024,25600,5120). Its distance 2.3345 exceeds the frozen neighborhood limit, so it falls back to native: 1.718159 ms versus selected Strassen 1.596807 ms, 7.60% slower. The wide and short-K rectangles also retain native labels and are 6.50% and 6.14% slower than their fastest measured alternatives. The largest prepared-kernel shortfall is the wide rectangle: 0.512402 ms versus selected Strassen 0.467292 ms, 9.65% slower.

Ratios below 1 are retained. A rule can choose a different member of the same algorithm family from the independently screened winner, and identical routes can receive different independent timing means. Such ratios do not demonstrate that the selector beats an ideal oracle.

## Duplicate routes and unmeasured routing cost

Fourteen of the 16 selector routes duplicate an alternative's exact algorithm, variant and tile: all 13 native fallbacks, plus the 6144-square Strassen choice. Their measured differences are not algorithmic improvements or estimates of policy overhead. Only the deep-K cubic and tall Strassen routes differ from the corresponding independently selected implementations.

The selector decision is computed before compilation and the timed rounds. The timing tables therefore measure execution of the chosen matrix-multiplication implementation; they do not include a Python policy decision on every call. A separate single host decision observation per shape ranges from 1.08 to 52.89 microseconds, with median 23.545 microseconds. That is not a steady-state routing benchmark and is not added to the reported latencies. A deployment claim must specify whether decisions are cached per geometry or paid on each call, and measure that scope separately.

## Numerical scope and independent audit

All recorded results pass the unchanged finite-output and error gates: relative L2 ≤0.02 and maximum absolute error ≤`0.001 + 0.05 × max_abs_reference`. The largest recorded relative L2 is 0.00450246. Inputs are the registered synthetic Gaussian BF16 operands, with the common FP32 accumulation/output contract. The error checks retain the registered sampled-reference scope and whole-output finite check. These passes do not remove N4's known cancellation failures, certify unknown input values, or establish model-quality preservation. Qwen, Mistral and Gemma labels in this experiment identify matrix geometries, not real model activations.

Read-only inspection verified all 3,840 positive finite raw timing samples, exactly rounds 0–29 once per scoped result, and their recorded means across the 4,315 journal records. Every reported fastest alternative and regret ratio was independently recomputed from those final means. Paired round coverage and each speedup ratio of means were verified; this review used the archived bootstrap interval endpoints rather than independently reproducing the random bootstrap resamples. Recorded numerical-gate calculations were checked, without regenerating the matrices or numerical references.

The supported claim is that this frozen, interpretable rule transferred three useful custom choices to reserved geometries and stayed within 10% of the fastest observed alternative in both timing scopes across this bounded sample. Its conservatism leaves measured gains unused. Sixteen geometries on one allocation, one Gaussian evaluation seed, per-comparison intervals, a limited candidate search and excluded per-call policy lookup do not establish broad deployment performance. Native XLA is an essential practical comparator; the elementary custom cubic/Strassen families and their 20-candidate screens are not an exhaustive strongest-possible baseline search. Fresh-cohort replication and v6e remain separate later experiments.

No benchmarks were rerun, no frozen files were edited and no commits were made for this review.

Primary evidence: [evaluation summary](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/selector_evaluation.json), [frozen decisions](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/selector_decisions.json), [raw journal](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/results.jsonl), [provenance](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/input_provenance.json), [run summary](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/summary.json), and [screen review](N7_SCREEN_REVIEW_v001.md).
