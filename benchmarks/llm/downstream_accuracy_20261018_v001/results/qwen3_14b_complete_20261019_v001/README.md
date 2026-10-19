# Qwen3-14B: complete paired downstream subset evaluation

All three registered Qwen3-14B subsets are complete. Native and Strassen have
equal aggregate accuracy on MMLU and GSM8K; perplexity differs by +0.004019%.
The small accuracy subsets do **not** establish equivalence within the
predefined degradation margins. Perplexity meets its predefined criterion.
Qwen3-32B and Gemma4-31B are separate pending models, not included in these results.

| Model | Benchmark | Native | Strassen S1 | Difference (S1 minus Native) | Paired 95% confidence interval | Meets preservation criterion |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3-14B | MMLU, 114 questions | 94/114 (82.4561%) | 94/114 (82.4561%) | 0 pp | [-25.4396, +25.4396] pp | Inconclusive at 0.5 pp margin |
| Qwen3-14B | GSM8K, 32 problems | 19/32 (59.375%) | 19/32 (59.375%) | 0 pp | [-22.5416, +22.5416] pp | Inconclusive at 1 pp margin |
| Qwen3-14B | WikiText-2 perplexity, 32 windows | 7.83395212 | 7.83426697 | +0.00031485 (+0.004019%) | Absolute [-0.00651963, +0.00707489]; relative [-0.085136%, +0.088274%] | Yes, within 1% relative increase |

Perplexity is a language-modeling metric, not an accuracy percentage. It covers
16,384 scored tokens in 32 nonoverlapping windows. MMLU samples two questions
from each of 57 subjects. Subject-level scores are retained in `summary.json`.
These are the full **registered small subsets**, not the full benchmark splits.

## Paired disagreements and stopping behavior

MMLU predictions are identical on all 114 questions: 94 both correct, 20 both
incorrect. GSM8K has 17 both correct, 11 both incorrect, two
Native-correct/Strassen-incorrect cases, and two
Native-incorrect/Strassen-correct cases. Five scored predictions differ
(15.625%), including one case where both predictions are scored incorrect.

The first extracted numeric answers match on **31/32** GSM8K examples. This
descriptive agreement is distinct from the registered final-answer score:

| Example | Native | Strassen | Registered outcome |
| --- | --- | --- | --- |
| test:100 | Extracted 175; token cap | Extracted 175; normal stop | Strassen correct only |
| test:128 | Extracted 880; token cap | Extracted 880; normal stop | Strassen correct only |
| test:1026 | Extracted 43200; normal stop | Extracted 43200; token cap | Native correct only |
| test:1205 | Extracted 4; normal stop | Extracted 4; token cap | Native correct only |
| test:380 | Extracted reference answer 803; token cap | Extracted 760.42; normal stop | Both incorrect |

Native hit the 1,024-token cap on 11/32 examples; Strassen on 10/32. The
predefined rule counts budget exhaustion incorrect even when an answer was
already extracted. Strict string comparison also counts `9.00` versus `9`
and `6.00` versus `6` incorrect for both algorithms. No post-hoc numeric
normalization or token-budget exception is used. The genuine numeric-answer
disagreement on `test:380` is retained even though both final scores are wrong.

`gsm8k-stopping-audit.json` contains all paired stopping/extraction details.
The unmodified generated text and token IDs are in `raw/qwen3_14b/` and in
the original `model.tar.gz` archive. Equal aggregate scores do not imply
identical generation trajectories or individual predictions.

## Frozen settings and interpretation

Checkpoint: `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Both arms use identical real
weights and BF16/DEFAULT precision. Plain tile-local Strassen S1, tile
128/256/256, replaces supported q/k/v/o/gate/up/down matrix products without
fused epilogues. This is an accuracy study, not an inference-speed measurement.

The tokenizer, input rows, five-shot raw-completion prompts, context limits,
batch 8, greedy decoding, 1,024-token generation cap, stopping/extraction
rules, PPL stride and masks are matched. Exact software versions, seeds,
checkpoint, source/input hashes and execution settings are retained in the
manifest, contract, source bundle and
[frozen execution protocol](../../LLM_ACCURACY_EXECUTION_20261018_v003.md).
Execution used one v6e; layer weights were streamed for memory capacity.

The registered confidence methods are a conservative stratified Hoeffding
bound for MMLU, paired discordant-cell simultaneous Clopper-Pearson bounds
for GSM8K, and a paired whole-window NLL bootstrap for perplexity (20,000
draws, seed 20261018). Neither accuracy interval excludes degradation beyond
its margin. The observations support similar aggregate scores on these inputs;
they do not prove downstream accuracy preservation at the specified margins.
The PPL upper relative bound is below 1%, supporting preservation for this
checkpoint, text subset, precision and S1 configuration.

## Verification and reproduction

The remote model archive is 1,428,152 bytes with SHA-256
`e62cb2ff94ba447e13bb44a6fd802285ac556fcb9a61e8104c0cdd54c516c940`.
The local hash matches the remote receipt. Offline verification checks all 23
batch files, all 178 paired rows per arm, exact source/input/contract hashes,
fresh train-only admission and equality between aggregated rows and batches.
The original remote summary is retained under `raw/qwen3_14b/summary.json`.

The frozen analysis reproduces every score, count and preservation verdict.
One PPL confidence endpoint differs by 2.26e-17 between the remote and local
platforms. `verification.json` records this difference and the verifier's
1e-12 absolute/relative float-comparison tolerance. Discrete results remain
exact. The first verifier rejected that last-bit difference; its source,
failed command and receipt are retained in `prior-verifier-roundoff-failure/`.
The successful version is archived in `report-execution/`. The running
scientific code, inputs, scoring, statistical methods and margins were not changed.

With Python, NumPy and SciPy installed, reproduce into a new directory:

```sh
scratch=$(mktemp -d)
mkdir "$scratch/source"
tar -xzf ../../source.tar.gz -C "$scratch/source"
python code/accuracy_model_report_v002.py \
  --model-archive model.tar.gz --archive-receipt archive-receipt.json \
  --source-archive ../../source.tar.gz \
  --inputs "$scratch/source/plans/llm_accuracy_20261018_v001/subset_qwen3_14b_v002" \
  --output "$scratch/reproduced"
```

Earlier partial snapshots and runtime failures remain intact. Completion here
means Qwen3-14B's three subsets are verified; the three-model study is not done.
