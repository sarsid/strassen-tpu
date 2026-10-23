# Gemma4-31B: complete paired downstream subset evaluation

All three registered small subsets are complete. Native and Strassen S1 each
score 29/32 on GSM8K, with all 32 extracted/scored answers identical and no
generation-cap failures. Strassen loses one MMLU question relative to Native.
Perplexity changes by -0.065035% and meets the predefined preservation criterion.
The small downstream accuracy subsets remain inconclusive at their registered
margins; equal observed scores are not evidence of statistical equivalence.

| Model | Benchmark | Native | Strassen S1 | Difference (S1 minus Native) | Paired 95% confidence interval | Meets preservation criterion |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Gemma4-31B | MMLU, 114 questions | 95/114 (83.3333%) | 94/114 (82.4561%) | -0.8772 pp | [-26.3168, +24.5624] pp | Inconclusive at 0.5 pp margin |
| Gemma4-31B | GSM8K, 32 problems | 29/32 (90.625%) | 29/32 (90.625%) | 0 pp | [-12.7976, +12.7976] pp | Inconclusive at 1 pp margin |
| Gemma4-31B | WikiText-2 perplexity, 32 windows | 4.94330423 | 4.94008936 | -0.00321487 (-0.065035%) | Absolute [-0.00733526, +0.00051372]; relative [-0.145940%, +0.010416%] | Yes, within 1% relative increase |

MMLU contains two questions from each of 57 subjects; subject-level results are
in `summary.json`. Perplexity covers 16,384 scored tokens in 32 nonoverlapping
WikiText-2 windows. Perplexity is a language-modeling metric, not an accuracy
percentage. These are the complete registered subsets, not full test splits.

## Disagreements and generation

MMLU has 94 questions correct under both arms, 19 incorrect under both, one
Native-correct/Strassen-incorrect case, and no gains. The single changed
prediction is `test:8721` in `moral_scenarios`: reference C, Native C, Strassen D.
The observed -0.8772 pp difference exceeds the 0.5 pp target margin; its wide
interval leaves preservation inconclusive and does not establish a regression
beyond that margin either. We report the observed loss rather than hide it.

GSM8K has 29 problems correct under both arms and three incorrect under both.
Shared wrong answers are `test:128` (1000 versus reference 880), `test:236`
(30 versus 31), and `test:380` (1095 versus 803). All 32 extracted/scored answers
match. Both arms finish all examples before the 1,024-token generation cap.
Equal final answers do not imply identical generated text or length: for
example `test:1026` is correct in both arms, using 203 Native versus 176
Strassen tokens. These lengths are not kernel speed measurements.

## Matched configuration and interpretation

Checkpoint: `google/gemma-4-31B` at
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`, dense text model. Both arms use the
same weights and BF16/DEFAULT precision. Plain tile-local Strassen S1 replaces
supported q/k/v/o/gate/up/down projections, tile 128/256/256, one recursion
level, with no fused epilogues. Non-MM operations are unchanged.

Tokenizer, frozen examples, five-shot raw-completion prompts, context limit,
batch 8, greedy decoding, 1,024-token cap, stop rules, extraction and perplexity
masks/stride match between arms. Exact revisions, versions, seeds, commands and
hashes are recorded in the contracts and manifests. One v6e was used, with
weights streamed for memory capacity. This study assesses accuracy, not speed.

The registered MMLU interval is a conservative stratified Hoeffding bound.
GSM8K uses paired discordant-cell simultaneous Clopper-Pearson bounds.
Perplexity uses a paired whole-window NLL bootstrap with 20,000 draws,
block size 1 and seed 20261018. The margins are 0.5 pp MMLU, 1 pp GSM8K and
1% relative perplexity increase, fixed before inspecting results. Small
samples cannot rule out accuracy degradation at these tight margins.
Perplexity's upper relative confidence limit is below 1%, supporting
preservation for this checkpoint, subset, precision and S1 configuration.
Its interval includes zero, so the lower point estimate is not evidence of
improvement. No conclusions extend to other tasks, S2 or fused kernels.

## Recovery provenance and verification

All 23 paired batch files and 178 rows per arm pass the committed offline
verifier. It checks the remote archive hash/size, frozen source/input/contract
hashes, current admission, row aggregation, extraction, generation lengths,
and reproduction of the registered analysis. Discrete results and criteria
match exactly. Two GSM8K interval endpoints differ by 3.55e-15 across platforms,
within the recorded 1e-12 absolute/relative tolerance.

- Run v010 produced MMLU batches 000–012 (104 pairs). Its saved manifest says
  admission passed, but detailed original numerical preflight/admission
  receipts were not downloaded before runtime loss. This evidence gap remains.
- Run v011 produced MMLU batches 013–014, all four perplexity batches, and
  GSM8K batches 000–001 after fresh validation. Its real receipts are retained.
- Run v014 produced GSM8K batches 002–003 after new train-only preflight and
  admission, with exactly the same regenerated execution contract. All 21
  inherited batches are unchanged; the earlier MMLU/perplexity report is unchanged.

`execution-provenance.json` records every batch hash and origin. The root
current-admission receipt applies to v014, not retroactively to v010. The
verifier's `runtime_admission_passed` flag means the archived current admission
passed; it does not erase the original MMLU receipt gap.

The result-only archive is 420,384 bytes, SHA-256
`e0b2c5d34a24f1a89d86b479e0ec46eab84721aa988b3b197ccdb4629b8f4d6c`.
It contains JSON measurements, predictions and provenance, not model weights.
The final three-model archive was also locally hash-verified before TPU release.
Release was independently confirmed. See the
[three-model report](../three_model_complete_20261022_v001/README.md).

## Reproduce the analysis

With Python, NumPy and SciPy installed, run from this directory. The archived
scientific source is the already-published exact bundle; this command extracts
only the frozen evaluation input directory.

```sh
scratch=$(mktemp -d)
mkdir "$scratch/source"
tar -xzf ../../recovery_gemma_20261021_v001/source.tar.gz -C "$scratch/source"   plans/llm_accuracy_20261018_v001/subset_gemma4_31b_v002
python code/accuracy_model_report_v002.py   --model-archive model.tar.gz --archive-receipt archive-receipt.json   --source-archive ../../recovery_gemma_20261021_v001/source.tar.gz   --inputs "$scratch/source/plans/llm_accuracy_20261018_v001/subset_gemma4_31b_v002"   --output "$scratch/reproduced"
```

Earlier snapshots and failures remain unchanged. No further experiments are queued.
