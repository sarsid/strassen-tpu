# Strassen MM Focus — morning review

**N5–N9 are complete on one v5e allocation, and their evidence audits passed.** The main finding is that one-level Strassen helps some matrix shapes. Those gains do not consistently become faster full-model execution.

Start with **N5** below. Each linked report includes the positive results, negative results and supporting artifacts.

| Experiment | Short takeaway | Read more |
|---|---|---|
| **N5 — fair tuning and fresh confirmation** | Across 16 training shapes, Strassen had 6 wins, 4 inconclusive results and 6 losses against native XLA. The large square and Qwen-shaped products beat both native and separately tuned cubic. | [N5 findings](docs/N5_CONFIRMATION_REVIEW_v001.md) |
| **N6 — TPU device traces** | Ordinary call timings and device timings sometimes rank implementations differently. A separately labeled, post-hoc check confirms device advantages for two large winning shapes. The original reversals remain part of the results. | [Original N6](docs/N6_FINDINGS_v001.md) · [Large-shape supplement](docs/N6_SUPPLEMENT_FINDINGS_v001.md) |
| **N7 — choosing an implementation for unseen shapes** | The frozen rule chose custom code on 3 of 16 held-out shapes; all three beat native. It chose native on the other 13 and missed some opportunities. | [N7 findings](docs/N7_FINDINGS_v001.md) |
| **N8 — combining multiplication with application operations** | Standard fusion helps some paths. Fused Strassen beats native on 6/8 residual cases but loses all 8 complete-call SwiGLU comparisons. Early finalization mostly makes performance worse. | [N8 findings](docs/N8_FINDINGS_v001.md) |
| **N9 — actual Qwen and Mistral models** | Both custom policies pass the defined corpus-quality checks. Strassen's resident layers are slower on Qwen and faster on Mistral; transfer-inclusive full-forward timings are near parity. Gemma remains access-blocked. | [N9 findings](docs/N9_FINDINGS_v001.md) |

**Keep these limits in view.** N4 found 24 Strassen failures on cancellation-heavy inputs; later passes do not remove that risk. A shape-based rule cannot establish numerical safety for arbitrary values. N7 timings exclude policy lookup. N9 covers teacher-forced prefill, with only three descriptive full-forward repeats. Confidence intervals are per-comparison. The traces do not establish MXU/vector utilization or overlap. The [earlier N1–N4 handoff](START_HERE_v001.md) preserves that study; its “not yet run” statements reflect its earlier date.

**Fresh-v5e N7 replication: PENDING.** Its result is not included here. It will reuse the frozen rule and alternatives without retuning, with the new allocation reported separately. [Replication plan](docs/N7_REPLICATION_RUNBOOK_v001.md). **v6e replication and two-level Strassen remain future work.**

For provenance, see the [decision log](decisions/N5_N9_v001.md) and passing audits for [N5–N6](runs/20260919T065232Z-N5-N6-evidence-audit-v002-e8ba5a/artifacts/audit.md), [N7 and the N6 supplement](runs/20260919T073531Z-N7-supplement-evidence-audit-v002-9b8d91/artifacts/audit.md), [N8](runs/20260919T074606Z-N8-evidence-audit-v002-a6cbd3/artifacts/audit.md), and [N9](runs/20260919T083449Z-N9-evidence-audit-v002-358ac4/artifacts/audit.md). Audit success means the evidence checks passed; it does not mean every experiment produced a speedup. Executed code, failures and results remain in versioned archives and local commits.

[Live local status and agents](http://127.0.0.1:8765)
