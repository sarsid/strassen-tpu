# Qwen3-14B: first sixteen paired GSM8K examples

This immutable interim snapshot contains two completed batches of the registered
**32-example** GSM8K subset. Batch 3 of 4 was running at 15:05 UTC on October 19.
The GSM8K task and the three-model study remain incomplete. This ordered prefix
does not receive an interim confidence interval or preservation verdict.

| Fixed scoring rule | Native | Plain Strassen S1 |
| --- | ---: | ---: |
| Correct final predictions | 9/16 (56.25%) | 11/16 (68.75%) |
| Reached the 1,024-token generation limit | 5/16 | 2/16 |
| Correct in newly completed batch 2 | 5/8 | 5/8 |

There are two Native-incorrect/Strassen-correct cases and zero
Native-correct/Strassen-incorrect cases under the frozen scoring rule. The two
correctness differences are the previously reported `test:100` and `test:128`:
both algorithms extracted the correct answer, but Native exhausted the token
budget and was counted incorrect. This explains the score advantage; it does
not establish better reasoning or preserved accuracy.

**First extracted answers match on 15/16 examples.** The new disagreement is
`test:380`: Native extracted `803` (the reference answer) but reached the token
limit, while Strassen extracted `760.42` and stopped normally. Both are scored
incorrect under the registered rules. Thus there are three different scored
predictions, although only two differ in correctness. The numeric-answer
disagreement is retained explicitly rather than hidden by the equal scores on
that example.

Both algorithms produced `9.00` versus reference `9` on `test:227`, and `6.00`
versus `6` on `test:258`; the frozen strict string comparison counts these
incorrect. No post-hoc numeric normalization or budget exception is applied.
Raw generated text, token IDs, stop reasons, extracted answers and scored
predictions are preserved in `batches/`.

Checkpoint: `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Both arms use identical weights,
BF16, DEFAULT precision, batch 8 and greedy generation. S1 uses tile
128/256/256 at q/k/v/o/gate/up/down without fused epilogues. The original full
manifest and execution contract are retained.

## Recovery and verification

Batch 1 was recovered from the previous runtime; batch 2 completed on recovery
v006. The regenerated execution contract and restored batch were verified
byte-identical, and fresh train-only admission passed on v006. Scientific
source, inputs, settings and scoring were unchanged. The included earlier
runtime log contains that admission; it is not a log of batch 2 completion.
The observed-progress receipt records the subsequent transition to batch 3.

`verification.json` records source/input hashes, paired-row checks and fresh
admission. `report-execution/` archives successful offline verification using
the committed report implementation. No TPU computation was launched by this
report. Prior first-eight and complete MMLU/perplexity snapshots remain intact.

With Python, NumPy and SciPy installed, reconstruct into a new directory:

```sh
scratch=$(mktemp -d)
mkdir "$scratch/source"
tar -xzf ../../source.tar.gz -C "$scratch/source"
python code/accuracy_gsm_prefix_report_v001.py \
  --checkpoints . \
  --inputs "$scratch/source/plans/llm_accuracy_20261018_v001/subset_qwen3_14b_v002" \
  --source-archive ../../source.tar.gz \
  --runtime-log runtime-log-20261019T1312Z-v001.log \
  --batches 2 --output "$scratch/reconstructed"
```

Later results must be published in another snapshot; this folder is not a
replacement for the final registered 32-example analysis.
