# N7 fixed-policy replication on a fresh v5e allocation

**The three custom selector routes retained their complete-call gains over native XLA on the new logical v5e allocation.** All 128 scoped results passed, with 3,840 recorded timing samples. The selector stayed within 10% of the fastest measured eligible alternative on every held-out shape in both timing scopes. This supports repeatability for these fixed Gaussian inputs and geometries; it does not establish universal numerical safety or v6e portability.

The [pending v001 scaffold](N7_REPLICATION_FINDINGS_v001.md) is preserved. This report uses the final sealed replica and compares the two cohorts separately, without retuning or pooling their raw timings.

## Preserved experiment and qualified cohort

The original [N7 evaluation](../runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts/selector_evaluation.json) and [replica](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/artifacts/selector_evaluation.json) use the same 16 reserved shapes, Gaussian seed `20260924`, numerical gates, four arms and 30 timed rounds per arm and scope. The arms are native XLA, independently selected cubic, independently selected Strassen and the frozen selector. Its choices remain **13 native, two Strassen and one cubic**. The two timings are complete call, including device preparation/cropping, and prepared kernel, reusing prepared device operands. Compilation, initial host transfer and reference calculation are excluded.

BF16 inputs/pre-adds, FP32 accumulation/output and DEFAULT dot precision are unchanged. [Compatibility evidence](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/artifacts/cohort_compatibility.json) records matching TPU v5 lite device metadata, JAX/jaxlib `0.7.2`, libtpu `0.0.21.1`, NumPy `2.1.3`, complete package versions, runtime flags and image `release-colab-external-images_20260917-060051_RC00`. Only allocation/endpoint, hostname/host ID and boot ID changed.

| Cohort | Allocation suffix | Hostname | Boot ID |
|---|---|---|---|
| Original | `3iv9vtqwssy8q` | `6f3b9b8cc92c` | `64fe01c7-4c57-43e9-9f49-6805fb8a887d` |
| Replica | `s89akviwl7d7` | `ec3069c714ba` | `823b0acd-effe-4415-b35c-89465d02ed5f` |

Both full endpoints begin `tpu-v5e1-s-kkb-usw1c1-`. This proves distinct logical allocations; physical chip serials are unavailable. New setup and smoke identities matched, smoke passed 24/24, and the launcher's expected identity was byte-identical to the canonical smoke environment. The actual replica runtime matched that new identity before measurements.

The [replication inputs](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/artifacts/replication_inputs.json) preserve the original evaluation seal/journal, rule, selection and source links. Independently reread hashes matched the original selector (`18f0cee2…`), selections (`5376adc4…`), original journal (`757c77c0…`) and new smoke identity (`d82eeab9…`). All 16 decisions retained their choice, tile, variant, prototype, distance, reason and input scope. The runner verified five directly reused module hashes; the final frozen-source audit found no changed frozen working files, including imported dependencies.

## Completeness and numerical checks

The [summary](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/artifacts/summary.json) and [raw journal](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/artifacts/results.jsonl) reconcile to **16/16 groups, 128/128 unique successful scoped results and 3,840 finite positive samples**. Each scoped result has rounds 0–29 exactly once. There are no failed, skipped or uncompleted cases. Recomputed means matched recorded means within `6.7e-16` ms; all selector ratio-of-means values matched exactly. All recorded paired comparisons use the same 30 rounds and are numerically eligible.

Every result passed finite-output, relative-L2 ≤0.02 and maximum-error ≤`0.001 + 0.05 × max_abs_reference` gates. Maximum observed relative L2 was **0.00450246**, matching the original study. The 128 scoped results comprise 16 full-reference and 112 sampled-cross-product checks; sampled checks still use all K terms. Passing this Gaussian scope does not remove the previously observed cancellation failures.

## The three unchanged custom routes

Shapes are `(M,K,N)`; tiles are `(BM,BN,BK)`. Speedup is native mean divided by selector mean. Brackets show the recorded paired 95% bootstrap interval.

| Shape and fixed route | Original complete-call speedup | Replica native / selector, ms | Replica complete-call speedup |
|---|---:|---:|---:|
| Deep K `(3072,6144,1024)`: cubic plain, tile `(1024,1024,1024)` | 1.0945× [1.0765,1.1195] | 0.505164 / 0.467890 | **1.0797× [1.0590,1.1004]** |
| Tall `(6144,3072,1024)`: Strassen interleaved/output accumulator, tile `(1024,1024,1024)` | 1.0799× [1.0585,1.0988] | 0.520030 / 0.466297 | **1.1152× [1.0915,1.1431]** |
| Square `(6144,6144,6144)`: Strassen interleaved/output accumulator, tile `(2048,2048,512)` | 1.0904× [1.0854,1.0956] | 2.768972 / 2.547567 | **1.0869× [1.0811,1.0918]** |

All three also retain prepared-kernel gains: deep K **1.0698× [1.0583,1.0816]**, tall **1.0773× [1.0574,1.0991]**, square **1.0863× [1.0810,1.0907]**. Their original prepared ratios were 1.0757×, 1.0766× and 1.0861× respectively. The complete-call improvements correspond to about 7.4%, 10.3% and 8.0% less time than native in the replica.

## All held-out comparisons, including regressions

Entries are **wins / inconclusive / losses**, classifying each recorded interval relative to 1. These are per-comparison intervals from 2,000 paired bootstrap resamples, without multiplicity correction. This review reconciled rounds, means and ratios and inspected archived interval endpoints; it did not rerun bootstrap resampling.

| Scope | Selector compared with | Original | Replica |
|---|---|---:|---:|
| Complete call | Native | 3 / 12 / 1 | 4 / 12 / 0 |
| Complete call | Selected cubic | 11 / 2 / 3 | 9 / 5 / 2 |
| Complete call | Selected Strassen | 8 / 5 / 3 | 10 / 2 / 4 |
| Prepared kernel | Native | 3 / 13 / 0 | 3 / 13 / 0 |
| Prepared kernel | Selected cubic | 6 / 6 / 4 | 9 / 3 / 4 |
| Prepared kernel | Selected Strassen | 6 / 4 / 6 | 7 / 4 / 5 |

The fourth replica complete-call native “win” is `held_mistral_gateup`: **1.0046× [1.0002,1.0108]**, with native and selector means 1.512956 and 1.505991 ms. Both routes execute the same native implementation. It is a nominal timing difference, not evidence of a faster algorithm. The original native/native loss at `held_small_m16` is inconclusive in the replica. Fourteen of the 16 selector routes duplicate an independently timed alternative exactly, so apparent differences between those duplicate routes require the same caution. Selector decisions are computed before timing; a policy lookup on every invocation is not included.

Using selected Strassen for every shape remains unattractive: its replica complete-call comparison against native has **6 wins, 1 inconclusive and 9 losses**, versus the original 6/3/7. The preserved rule avoids that blanket choice but does not always select the fastest measured alternative.

| Descriptive regret: selector / fastest eligible alternative | Original call | Replica call | Original prepared | Replica prepared |
|---|---:|---:|---:|---:|
| Median | 1.0041 | 1.0068 | 1.0148 | 1.0079 |
| Mean | 1.0159 | 1.0141 | 1.0269 | 1.0247 |
| 90th percentile | 1.0632 | 1.0562 | 1.0814 | 1.0769 |
| Maximum | 1.0760 | 1.0834 | 1.0965 | 1.0975 |
| Within 1% / 5% / 10%, out of 16 | 10 / 12 / 16 | 11 / 14 / 16 | 7 / 13 / 16 | 9 / 12 / 16 |

The largest replica regret is `held_qwen_down`: 8.34% complete-call and 9.75% prepared. These denominators use the same three eligible alternatives in each cohort and exclude the selector itself; ratios below 1 are retained. This is a bounded observed comparison, not an exhaustive tuning oracle. Family agreement with the lowest observed mean is 12/16 call and 9/16 prepared in the replica, versus 11/16 and 9/16 originally.

## Canonical evidence and limits

Replica `20260919T084449Z-N7-replicate-v5e-v004-955768` executed source commit `cb4fcc17`, archived in `74d0108`, with remote archive SHA256 `a5762516ee9fc7667d2f9f707ba0e0633d1852e4706b16b9410c67aa1ef9c880`. [Completion](../runs/20260919T084449Z-N7-replicate-v5e-v004-955768/completion.json) records exit 0 and no remote worker left running. The [canonical evidence audit](../runs/20260919T084816Z-n7-replica-evidence-audit-v002-8400c2/artifacts/audit.json), commit `92d0772`, passed with no issues for replica or smoke and no frozen working-source changes. The [release record](../runs/20260919T084858Z-release-replica-v5e-after-audit-v001-eb7929/execution.log), commit `99be0179`, confirms the replica allocation was released and absent after audit.

The evidence supports persistence of the three fixed custom gains and bounded observed regret on one additional compatible logical v5e allocation. The same evaluation seed was reused, so this adds a runtime cohort, not new data-distribution coverage. Thirty rounds on one allocation are not independent machine replications. Neither cohorts nor scopes are pooled. Real-activation quality, production serving behavior, other runtime versions and v6e require their own evidence; the separate N9 study addresses only its stated application scope.

This report was produced by read-only source/evidence reconciliation. No new measurements, code changes, policy fitting or commits were performed for it.
