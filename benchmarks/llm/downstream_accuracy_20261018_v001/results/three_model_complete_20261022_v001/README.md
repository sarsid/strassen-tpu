# Three-model downstream accuracy study: complete results

All nine registered model–task evaluations are complete. On these small subsets,
Qwen3-14B and Qwen3-32B have unchanged aggregate MMLU/GSM8K accuracy.
Gemma4-31B has unchanged GSM8K accuracy and one fewer correct MMLU answer
with Strassen. All three perplexity comparisons meet the predefined preservation
criterion. None of the small accuracy subsets rules out degradation at the
registered margins; their preservation assessments remain inconclusive.

| Model | Benchmark | Native | Strassen S1 | Difference | Paired 95% confidence interval | Meets preservation criterion |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3-14B | MMLU | 82.4561% | 82.4561% | +0.0000 pp | [-25.4396, +25.4396] pp | Inconclusive; 0.5 pp margin |
| Qwen3-14B | GSM8K | 59.3750% | 59.3750% | +0.0000 pp | [-22.5416, +22.5416] pp | Inconclusive; 1 pp margin |
| Qwen3-14B | WikiText-2 PPL | 7.83395212 | 7.83426697 | +0.00031485 (+0.004019%) | Abs [-0.00651963, +0.00707489]; relative [-0.085136%, +0.088274%] | Yes; 1% relative margin |
| Qwen3-32B | MMLU | 84.2105% | 84.2105% | +0.0000 pp | [-25.4396, +25.4396] pp | Inconclusive; 0.5 pp margin |
| Qwen3-32B | GSM8K | 90.6250% | 90.6250% | +0.0000 pp | [-12.7976, +12.7976] pp | Inconclusive; 1 pp margin |
| Qwen3-32B | WikiText-2 PPL | 6.90390076 | 6.90596510 | +0.00206434 (+0.029901%) | Abs [-0.00586117, +0.01003638]; relative [-0.085144%, +0.144705%] | Yes; 1% relative margin |
| Gemma4-31B | MMLU | 83.3333% | 82.4561% | -0.8772 pp | [-26.3168, +24.5624] pp | Inconclusive; 0.5 pp margin |
| Gemma4-31B | GSM8K | 90.6250% | 90.6250% | +0.0000 pp | [-12.7976, +12.7976] pp | Inconclusive; 1 pp margin |
| Gemma4-31B | WikiText-2 PPL | 4.94330423 | 4.94008936 | -0.00321487 (-0.065035%) | Abs [-0.00733526, +0.00051372]; relative [-0.145940%, +0.010416%] | Yes; 1% relative margin |

Differences are Strassen minus Native. Perplexity is not an accuracy percentage.
Accuracy: higher is better; perplexity: lower is better. PPL intervals include
zero in all three models, so the small changes do not establish improvement
or regression. No multiple-comparison-adjusted global equivalence claim is made.

## Paired predictions

| Model | Benchmark | Different scored predictions | Native correct / S1 wrong | Native wrong / S1 correct |
| --- | --- | ---: | ---: | ---: |
| Qwen3-14B | MMLU | 0/114 | 0 | 0 |
| Qwen3-14B | GSM8K | 5/32 | 2 | 2 |
| Qwen3-32B | MMLU | 2/114 | 1 | 1 |
| Qwen3-32B | GSM8K | 0/32 | 0 | 0 |
| Gemma4-31B | MMLU | 1/114 | 1 | 0 |
| Gemma4-31B | GSM8K | 0/32 | 0 | 0 |

Qwen3-32B and Gemma4-31B have identical extracted answers on all 32 GSM8K
problems, with no generation-cap failures. Qwen3-14B has 31/32 matching
extracted answers, but five scored-prediction disagreements. Its cap failures
are 11 Native and 10 S1; all four changed correctness outcomes involve a
cap versus a normal stop. The frozen cap rule scores capped outputs as failures,
even if an answer was extracted earlier. The strict string matcher also keeps
`9.00` distinct from `9`; no post-hoc numeric normalization was applied.

Gemma loses one MMLU answer (`moral_scenarios`, test:8721, Native C / S1 D,
reference C), giving an observed -0.8772 pp change. This exceeds the target
0.5 pp margin as a point estimate, while the interval remains inconclusive.
Qwen3-32B has one MMLU gain and one loss. Qwen3-14B has none.

## Scope and reproducibility

Each model uses 114 MMLU test questions (two in each of 57 subjects), 32 seeded
GSM8K test problems, and 32 nonoverlapping WikiText-2 windows containing 16,384
scored tokens. These are small subsets, not full benchmark evaluations.
Subject-level MMLU results, exact example IDs and paired predictions are in
the linked model reports and JSON summaries.

Within each model, Native and plain Strassen S1 share checkpoint weights,
BF16/DEFAULT precision, tokenizer, frozen prompts/examples, five-shot settings,
context, batch 8, greedy decoding, 1,024-token cap, stopping/extraction rules
and PPL masks/stride. No chat-template or reasoning-mode change occurs between
arms. S1 has one recursion level and tile 128/256/256 at supported q/k/v/o/gate/up/down
projections, with unchanged non-MM operations and no fused epilogues.
One v6e was used with weight streaming. These are accuracy measurements;
generation wall times and token counts do not establish inference speedups.

| Model | Exact checkpoint | Revision | Complete result |
| --- | --- | --- | --- |
| Qwen3-14B | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | [Report](../qwen3_14b_complete_20261019_v001/README.md) |
| Qwen3-32B | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | [Report](../qwen3_32b_complete_20261021_v001/README.md) |
| Gemma4-31B | `google/gemma-4-31B` | `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` | [Report](../gemma4_31b_complete_20261022_v001/README.md) |

Exact dataset revisions, task variants, preprocessing, templates, seeds, software
versions, commands and input/source hashes are retained in the model manifests,
execution contracts, frozen source bundles and verification records. The custom
JAX harness uses the documented task definitions and scoring; these are not
unqualified model-card or standard-harness leaderboard scores.

The registered preservation margins are 0.5 pp MMLU, 1 pp GSM8K and 1% relative
PPL increase. They express small practical degradation tolerances, not a claim
that these small sample sizes can certify them. Accuracy uses paired conservative
bounds (stratified Hoeffding for MMLU and simultaneous discordant-cell
Clopper-Pearson for GSM8K). PPL uses 20,000 paired whole-window NLL bootstrap
draws, block size 1, seed 20261018. The frozen decision rules were applied
unchanged. Nonsignificance alone is not treated as preserved performance.

## Provenance and limitations

All models have complete, hash-verified result archives and reproduced analyses.
Recovery preserved completed paired batches unchanged rather than rerunning them.
Each model report discloses its runtime origins. Gemma has an explicit remaining
provenance gap: detailed numerical admission/preflight receipts for the first
104 MMLU pairs from v010 were not saved before runtime loss. The original
manifest records admission passed; later fresh v011/v014 receipts do not
retroactively fill this gap. Gemma GSM8K and PPL have actual admission evidence.

The final archive and all three model archives were verified locally before
TPU release, and an independent allocation listing confirmed release.
No further experiments are queued. Previous partial results, failures and
source snapshots remain unchanged. No model weights or recovery seed bundles
are included in this new publication.

The evidence supports stable observed downstream scores in most comparisons
and perplexity preservation for these checkpoints and subsets. It does not
establish full-benchmark accuracy equivalence, validate S2 or every fused kernel,
or justify claims about future architectures without separate measurements.
