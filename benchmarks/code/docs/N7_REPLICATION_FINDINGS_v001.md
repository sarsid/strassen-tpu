# N7 fresh-v5e replication — findings pending

**Pending scaffold, not a replication result.** A new logical v5e allocation has been created. Its setup, smoke qualification, replica measurements and final evidence audit have not been reviewed for this document. No replica latency, speedup, accuracy or generalization finding is asserted here. A finalized report should be published as a new version after canonical evidence arrives, preserving this pending snapshot.

## Question and unchanged experimental scope

Does the already frozen N7 selector preserve its observed usefulness on another logical v5e allocation, without refitting its rule or retuning either comparator?

The planned study reuses the original 16 reserved matrix geometries, Gaussian evaluation seed 20260924, numerical gates, chosen tiles/variants and four arms: native XLA, independently selected cubic, independently selected Strassen and the frozen selector route. Both complete-call and prepared-kernel timing scopes remain separate. The plan specifies 30 timed rounds per successful scoped result: **128 planned scoped results and 3,840 planned timing samples if all cases succeed**. These are expected counts, not observed replica counts.

BF16 inputs and Strassen pre-adds, FP32 accumulation/output and DEFAULT dot precision remain unchanged. Complete-call timing includes preparation/finishing inside the callable; prepared-kernel timing reuses prepared operands. Compilation, transfer of initial host inputs and reference calculation are outside these MM timing scopes. This experiment tests synthetic matrix products, not real LLM activations or model quality.

## Evidence presently established

| Item | Recorded evidence | Status in this document |
|---|---|---|
| Original N7 evaluation | `20260919T073017Z-N7-evaluate-v5e-v004-8036b5` | Existing completed study; see original findings |
| Original allocation | `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q` | Original identity preserved |
| Independent comparator screen | `20260919T065405Z-N7-screen-v5e-v004-6ef489` | Existing choices reused unchanged |
| Portable replica inputs | `data/n7_replication_v001/` | Original full evaluation artifacts and exact rule/selection bytes staged |
| New provisioning execution | `20260919T084229Z-n7-replica-provision-v5e-v001-ff8204` | Archived completion says completed, exit code 0 |
| New assignment | `tpu-v5e1-s-kkb-usw1c1-s89akviwl7d7` | Create output says `created: true`, `accelerator: V5E1` |
| New named session | `strassen-mm-focus-v5e-n7replica-20260919-v001` | Recorded by provisioning output |
| New setup identity | Pending canonical setup evidence | Not yet qualified here |
| New smoke run | Pending canonical smoke evidence | Not yet qualified here |
| Replica execution/source/result commits | Pending | No timings reported |
| Replica canonical evidence audit | Pending | No audit pass claimed |

The provisioned endpoint differs from the original. That establishes a new logical allocation, not a different physical chip: Colab does not expose the TPU chip serial needed for that claim. `V5E1` in the allocation API is not a substitute for successful device/software qualification.

Primary current sources: [new allocation output](../runs/20260919T084229Z-n7-replica-provision-v5e-v001-ff8204/execution.log), [provisioning completion](../runs/20260919T084229Z-n7-replica-provision-v5e-v001-ff8204/completion.json), [portable-input provenance](../data/n7_replication_v001/provenance.json), [original N7 findings](N7_FINDINGS_v001.md), and [replication runbook](N7_REPLICATION_RUNBOOK_v001.md).

## Identity and preservation evidence required for the final report

The replica runner is frozen `benchmark_n7_replica_v002.py`. Its added `--smoke-environment` argument identifies the canonical smoke artifacts directory, while the launcher still passes a copied single-file expected identity. The runner requires byte-identical identities, then checks the canonical smoke seal, completion and 24 passing results. It verifies new setup versus new smoke and actual replica runtime versus the new expected identity before allowing cross-cohort comparisons.

The final report must inspect these recorded artifacts:

| Artifact | What it must establish |
|---|---|
| `replication_inputs.json` | Original evaluation seal/journal links; exact selector/selection hashes; new setup/smoke links; `smoke_expected_identity_link.status == byte_identical`; unchanged reused-source hashes |
| `cohort_compatibility.json` | Distinct original/new allocation IDs; only disclosed allocation/host/boot fields differ; unchanged software, runtime image/flags, device metadata and arithmetic contract |
| `original_environment.json`, `new_setup_identity.json`, `new_smoke_environment.json`, `environment.json` | Each identity is preserved rather than rewritten to make validation pass |
| `original_selector.json`, `original_selections.json` | Exact original policy/alternative bytes, with original identities left intact |
| `source_manifest.json` and source/config snapshots | Frozen campaign/reservation and actual executed implementation; all imported frozen code remains unchanged under the repository audit |
| `planned_cases.json`, `selector_decisions.json` | Same 16 shapes, inputs, four arms, scopes and algorithm/variant/tile decisions |
| Outer/canonical seals and final archive audit | Complete local evidence belongs to the recorded execution and source commit |

Expected selector SHA256: `18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77`.

Expected comparator selection SHA256: `5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37`.

Expected original evaluation journal SHA256: `757c77c09ad1016415ee4930305d6d0229af3f128ef802db9433f113cbe88153`.

The runner directly verifies original hashes for `kernels_v002.py`, `benchmark_v001.py`, `benchmark_n5_v001.py`, `benchmark_n6_n7_v002.py` and `selector_v002.py`. The complete frozen-source audit remains relevant to their other imported dependencies. A compatibility failure is an explicit unsuccessful replication attempt; it must not be converted into a positive result by changing the original identity or silently relaxing a gate.

## Measurements to fill from canonical replica evidence

Read `summary.json` and `results.jsonl` first. Report completed groups, exact scoped-result counts, numerical failures, compilation/VMEM errors, skips and interruptions. Check unique `(group, distribution, seed, arm, scope)` keys. Each successful scoped result needs finite positive samples for rounds 0–29 exactly once, with its recorded mean reproduced from those samples. Completion alone does not imply numerical eligibility or speedup.

| Evidence quantity | Replica observation |
|---|---|
| Completed/planned groups | Pending / 16 planned |
| Successful scoped results | Pending |
| Numerical failures / compile errors / OOM / skips | Pending |
| Raw timing coverage and mean checks | Pending |
| Maximum observed relative L2 and error-gate verification | Pending |
| Exact rule-decision preservation | Pending |

For each scope and reference arm, summarize selector wins, inconclusive comparisons and losses using the recorded paired 95% intervals. Speedup is reference mean divided by selector mean. An interval wholly above 1 indicates a nominal win; wholly below 1 indicates a nominal loss; an interval containing 1 is inconclusive. These remain per-comparison intervals without multiplicity correction. State whether the bootstrap was independently reproduced or only its archived endpoints were inspected.

| Scope | Reference | Original cohort | New cohort |
|---|---|---|---|
| Complete call | Native XLA | See original report | Pending |
| Complete call | Selected cubic | See original report | Pending |
| Complete call | Selected Strassen | See original report | Pending |
| Prepared kernel | Native XLA | See original report | Pending |
| Prepared kernel | Selected cubic | See original report | Pending |
| Prepared kernel | Selected Strassen | See original report | Pending |

Inspect the original three custom selector decisions individually: `held_rect_deep_k`, `held_rect_tall` and `held_square_6144`. Preserve and report any reversal; do not select only successful cases. Also inspect the 13 native fallback routes and distinguish duplicate-implementation timing differences from changes in the chosen arithmetic.

## Regret and cross-cohort interpretation

`selector_evaluation.json` reports selector mean divided by the lowest mean among numerically eligible native, selected-cubic and selected-Strassen alternatives in the same shape, scope and cohort. The selector is excluded from that denominator. Retain ratios below 1: a rule may choose a different same-family configuration, and duplicated routes are still independently timed. Missing or failing alternatives do not become zero-time baselines; report their eligibility and coverage explicitly.

Fill median, mean, 90th percentile, maximum and counts within 1%, 5% and 10% for each scope from the valid replica cases. This is descriptive regret over a bounded comparator set, not an exhaustive oracle or a generalization bound. Explain any denominator/coverage change before comparing the two cohort summaries.

Compare original and new within-cohort paired ratios side by side. Do not pool raw samples, average original and replica latencies into one observation, or imply that 30 rounds across 16 shapes represent 480 independent machines. The frozen selector decision is computed before compilation/timing; MM latency excludes a Python policy lookup per invocation. Small differences between identical native routes do not establish routing overhead.

## Conclusion placeholder

**Pending canonical replica results and audit.** The final report will state whether the specific original custom gains and bounded-regret behavior persisted, weakened, reversed, or could not be assessed. A second logical v5e allocation cannot establish universal numerical safety, real-activation/model quality, production serving performance, or v6e portability. Known cancellation failures remain outside the selector's validated Gaussian scope.

This pending scaffold was prepared by read-only source/evidence inspection. No code, policy, runbook or existing result was modified, and no tests, remote commands, benchmarks or commits were executed.
