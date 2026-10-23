# Strassen MM Focus — morning review

**N5–N9 and the separate fresh-v5e N7 replication are complete. Their evidence audits passed.** One-level Strassen helps some matrix shapes, and the frozen rule's three custom choices won again on a fresh allocation. Those MM gains do not consistently become faster full-model execution.

Start with **N5** below. Each linked report includes the positive results, negative results and supporting artifacts.

| Experiment | Short takeaway | Read more |
|---|---|---|
| **N5 — fair tuning and fresh confirmation** | Across 16 training shapes, Strassen had 6 wins, 4 inconclusive results and 6 losses against native XLA. The large square and Qwen-shaped products beat both native and separately tuned cubic. | [N5 findings](docs/N5_CONFIRMATION_REVIEW_v001.md) |
| **N6 — TPU device traces** | Ordinary call timings and device timings sometimes rank implementations differently. A separately labeled, post-hoc check confirms device advantages for two large winning shapes. The original reversals remain part of the results. | [Original N6](docs/N6_FINDINGS_v001.md) · [Large-shape supplement](docs/N6_SUPPLEMENT_FINDINGS_v001.md) |
| **N7 — choosing an implementation for unseen shapes** | The frozen rule chose custom code on 3 of 16 held-out shapes; all three beat native. It chose native on the other 13 and missed some opportunities. | [N7 findings](docs/N7_FINDINGS_v001.md) |
| **N8 — combining multiplication with application operations** | Standard fusion helps some paths. Fused Strassen beats native on 6/8 residual cases but loses all 8 complete-call SwiGLU comparisons. Early finalization mostly makes performance worse. | [N8 findings](docs/N8_FINDINGS_v001.md) |
| **N9 — actual Qwen and Mistral models** | Both custom policies pass the defined corpus-quality checks. Strassen's resident layers are slower on Qwen and faster on Mistral; transfer-inclusive full-forward timings are near parity. Gemma remains access-blocked. | [N9 findings](docs/N9_FINDINGS_v001.md) |

**Keep these limits in view.** N4 found 24 Strassen failures on cancellation-heavy inputs; later passes do not remove that risk. A shape-based rule cannot establish numerical safety for arbitrary values. N7 timings exclude policy lookup. N9 covers teacher-forced prefill, with only three descriptive full-forward repeats. Confidence intervals are per-comparison. The traces do not establish MXU/vector utilization or overlap. The [earlier N1–N4 handoff](START_HERE_v001.md) preserves that study; its “not yet run” statements reflect its earlier date.

**Fresh-v5e replication: complete, 128/128 checks passed.** With no retuning, the same cubic choice and two Strassen choices again beat native in complete-call timing: 1.080×, 1.115× and 1.087× respectively. The other 13 choices use native; a nominal win between duplicate native routes is not an algorithmic gain. Cohorts are reported separately, with no pooling of raw timings. [Replication findings](docs/N7_REPLICATION_FINDINGS_v002.md) · [Passing replication audit](runs/20260919T084816Z-n7-replica-evidence-audit-v002-8400c2/artifacts/audit.md).

**Later work:** v6e replication. Two-level Strassen remains an optional future experiment. Gemma still requires authorized model access before evaluation. Both study allocations have been released; the [final release record](runs/20260919T084858Z-release-replica-v5e-after-audit-v001-eb7929/execution.log) confirms the replica allocation is absent. No TPU experiment remains running.

For provenance, see the [decision log](decisions/N5_N9_v001.md) and passing audits for [N5–N6](runs/20260919T065232Z-N5-N6-evidence-audit-v002-e8ba5a/artifacts/audit.md), [N7 and the N6 supplement](runs/20260919T073531Z-N7-supplement-evidence-audit-v002-9b8d91/artifacts/audit.md), [N8](runs/20260919T074606Z-N8-evidence-audit-v002-a6cbd3/artifacts/audit.md), and [N9](runs/20260919T083449Z-N9-evidence-audit-v002-358ac4/artifacts/audit.md). Audit success means the evidence checks passed; it does not mean every experiment produced a speedup. Executed code, failures and results remain in versioned archives and local commits.

[Live local status and agents](http://127.0.0.1:8765)
