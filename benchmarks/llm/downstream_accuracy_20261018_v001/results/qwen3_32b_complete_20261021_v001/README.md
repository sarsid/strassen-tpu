# Qwen3-32B: complete paired downstream subset evaluation

All three registered Qwen3-32B subsets are complete. Native and Strassen S1
have equal aggregate MMLU and GSM8K accuracy. All 32 GSM8K extracted answers
match, and neither arm exhausted its generation budget. Strassen perplexity
is 0.029901% higher. The accuracy subsets remain inconclusive at the predefined
degradation margins; perplexity meets its preservation criterion.

| Model | Benchmark | Native | Strassen S1 | Difference (S1 minus Native) | Paired 95% confidence interval | Meets preservation criterion |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3-32B | MMLU, 114 questions | 96/114 (84.2105%) | 96/114 (84.2105%) | 0 pp | [-25.4396, +25.4396] pp | Inconclusive at 0.5 pp margin |
| Qwen3-32B | GSM8K, 32 problems | 29/32 (90.625%) | 29/32 (90.625%) | 0 pp | [-12.7976, +12.7976] pp | Inconclusive at 1 pp margin |
| Qwen3-32B | WikiText-2 perplexity, 32 windows | 6.90390076 | 6.90596510 | +0.00206434 (+0.029901%) | Absolute [-0.00586117, +0.01003638]; relative [-0.085144%, +0.144705%] | Yes, within 1% relative increase |

These are complete registered **small subsets**, not full benchmark splits.
MMLU uses two questions from each of 57 subjects. Subject-level results are in
`summary.json`. Perplexity covers 16,384 scored tokens in 32 nonoverlapping
WikiText-2 windows; it is a language-modeling metric, not an accuracy percentage.

## Paired predictions and generation

MMLU has 95 questions correct under both arms, 17 incorrect under both, one
Native-correct/Strassen-incorrect case, and one Native-incorrect/Strassen-correct
case. Predictions differ on 2/114 questions (1.7544%):

| Example | Subject | Reference | Native | Strassen | Outcome |
| --- | --- | --- | --- | --- | --- |
| test:2530 | formal_logic | D | D | A | Strassen loss |
| test:6067 | high_school_world_history | C | B | C | Strassen gain |

GSM8K has 29 problems correct under both arms and three incorrect under both.
There are no gains, losses, scored-prediction disagreements or extracted-answer
disagreements. Both arms stop normally on all 32 problems before the
1,024-token cap. The three shared errors include `9.00` versus reference `9`,
`6.00` versus `6`, and `760.42` versus `803`. The frozen strict string-matching
rule is preserved; no post-hoc numeric normalization changes these scores.

Equal extracted answers do not imply identical generated text or generation
length. For example, `test:1028` yields the correct answer 200 with 840 Native
tokens and 1,015 Strassen tokens; `test:1205` yields 4 with 554 Native tokens
and 435 Strassen tokens. These lengths are not matrix-multiplication speed
measurements. Full generated text, token IDs, stopping reasons and per-example
predictions are preserved in `raw/qwen3_32b/` and `gsm8k-stopping-audit.json`.

## Matched configuration and interpretation

Checkpoint: `Qwen/Qwen3-32B` at
`9216db5781bf21249d130ec9da846c4624c16137`. Native and plain tile-local
Strassen S1 use identical weights, BF16/DEFAULT precision and non-MM
operations. S1 uses tile 128/256/256 and one recursion level at the supported
q/k/v/o/gate/up/down projections, without fused epilogues.

Both arms use the same tokenizer, frozen examples, five-shot raw-completion
prompts, context limit, batch 8, greedy decoding, 1,024-token generation cap,
stopping rules, answer extraction and perplexity masks/stride. Exact versions,
seeds, commands, checkpoint and source/input hashes are in the execution
contract, manifests, archived sources and execution records. Execution uses
one v6e with weights streamed for memory capacity. This is an accuracy study,
not an inference-speed experiment.

The frozen methods use a conservative stratified Hoeffding bound for MMLU,
paired discordant-cell simultaneous Clopper-Pearson bounds for GSM8K, and a
paired whole-window NLL bootstrap for perplexity (20,000 draws, seed 20261018).
The accuracy intervals do not rule out degradation beyond the registered
0.5 pp MMLU or 1 pp GSM8K margins. Equal observed accuracy therefore does not
establish equivalence. Perplexity's upper relative bound is below the registered
1% margin, supporting preservation for this checkpoint, text subset, precision
and S1 configuration. Conclusions do not extend to other tasks or kernels.

## Recovery provenance and verification

The complete model archive is 1,131,419 bytes, SHA-256
`e58dc6e6dbb2a068b67c8f06ce7d8dae3df2bceb7042fee917154c56eef39ce8`.
Its local hash matches the remote receipt. The offline verifier checks all 23
batch files and 178 rows per arm, the frozen source/input/contract hashes,
runtime admission, generation lengths, extraction, and exact equality between
aggregated rows and paired batches. It reproduces all discrete scores and
preservation verdicts. Two GSM8K confidence endpoints differ by 3.55e-15
between platforms, within the recorded 1e-12 absolute/relative tolerance.

This result spans a provider runtime loss and a checkpointed resumption:

- Run `20261019-accuracy-study-v008` supplied all 15 MMLU batches, four
  perplexity batches and the first two GSM8K batches (16 paired problems).
- Run `20261020-accuracy-study-v010` supplied the last two GSM8K batches after
  fresh train-only preflight and runtime admission. It completed on October 21
  at 00:56:55 UTC. All 21 inherited batch files match the original recovery seed
  byte-for-byte; the regenerated execution contract matches exactly.
- Previously published MMLU/perplexity scores are unchanged. The original
  admission is retained under `raw/qwen3_32b/provenance/`, alongside the fresh
  root admission and resumed event log. The resumed event log does not claim
  to contain the original runtime's events.

`execution-provenance.json` records every batch's origin and hash. The exact
scientific source and seed are preserved in
[the recovery snapshot](../../recovery_20261020_query_checkpoints_v001/README.md);
the fresh admission and contract checks are in
[the admission snapshot](../../recovery_20261020_admission_v001/README.md).
The model verifier, exact command, successful execution log and its committed
source snapshot are included in `code/` and `report-execution/`.

With Python, NumPy and SciPy installed, reproduce from this directory into a
new output directory:

```sh
scratch=$(mktemp -d)
mkdir "$scratch/source"
tar -xzf ../../recovery_20261020_query_checkpoints_v001/source.tar.gz -C "$scratch/source"
python code/accuracy_model_report_v002.py \
  --model-archive model.tar.gz --archive-receipt archive-receipt.json \
  --source-archive ../../recovery_20261020_query_checkpoints_v001/source.tar.gz \
  --inputs "$scratch/source/plans/llm_accuracy_20261018_v001/subset_qwen3_32b_v002" \
  --output "$scratch/reproduced"
```

This is an additive publication. Earlier partial results, runtime failures and
sources remain intact. Qwen3-14B and Qwen3-32B are complete; Gemma4-31B remains
in progress at publication, so the three-model study is not yet complete.
