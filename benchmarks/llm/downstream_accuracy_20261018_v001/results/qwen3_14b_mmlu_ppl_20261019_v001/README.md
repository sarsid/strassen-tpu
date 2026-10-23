# Qwen3-14B: completed MMLU and perplexity subsets

This snapshot preserves two complete tasks from the October 19 accuracy study.
GSM8K is unfinished; this is not a completed three-model study. The previous
first-80-question snapshot is unchanged.

| Task | Native | Strassen S1 | Difference | Paired 95% interval | Preservation criterion |
| --- | ---: | ---: | ---: | --- | --- |
| MMLU, 114 questions | 94/114 (82.4561%) | 94/114 (82.4561%) | 0 pp | [-25.4396, +25.4396] pp | Inconclusive at 0.5 pp margin |
| WikiText-2, 32 windows / 16,384 scored tokens | 7.83395212 | 7.83426697 | +0.00031485 (+0.004019%) | Relative change [-0.085136%, +0.088274%] | Meets 1% relative margin |

All 114 MMLU predictions are identical: 94 both correct, 20 both incorrect,
zero Native-correct/Strassen-incorrect and zero reverse cases. Subject-level
results and all paired rows are in summary.json and the two JSONL files. The
registered stratified Hoeffding interval is conservative; identical predictions
in this small subset do not establish the narrow accuracy-preservation margin.
The pooled binomial interval is descriptive only, not the registered verdict.

Perplexity uses the registered paired whole-window bootstrap (20,000 draws,
seed 20261018). No metric definitions, margins or inference settings changed
after seeing results. Conclusions apply only to this fixed checkpoint, subset,
BF16 precision and plain S1 configuration. Do not interpret the raw-completion
MMLU score as a vendor-reported quality score or perplexity as accuracy.

The full original manifest and execution contract are retained. The separate
analysis-manifest.json explicitly filters to the two complete tasks; it does
not substitute a smaller set for any complete-task result. All input row hashes,
batch order/contracts and frozen scientific source hashes were checked by
accuracy_checkpoint_report_v001.py before unchanged accuracy_analysis_v003.py
performed the analysis. Runtime admission passed before scoring.

The Colab allocation disappeared while the first GSM8K batch was decoding.
No completed GSM8K batch was recovered, and the cause is unknown. Completed
MMLU/perplexity pairs can be restored on a replacement only under exact
execution-contract equality and a fresh train-only admission check.
