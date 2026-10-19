# Qwen3-14B: first eight paired GSM8K examples

This immutable interim snapshot contains the first completed batch of the
registered **32-example** GSM8K subset. The second batch was running at
10:01 UTC on October 19. The GSM8K task and the three-model study are incomplete.
No preservation verdict or general accuracy advantage is claimed from this
ordered prefix.

| Fixed scoring rule | Native | Plain Strassen S1 |
| --- | ---: | ---: |
| Correct final predictions | 4/8 (50%) | 6/8 (75%) |
| Reached the 1,024-token generation limit | 3/8 | 1/8 |

Two scored predictions differ: `test:100` and `test:128` are counted incorrect
for Native because it exhausted the generation budget, while Strassen reached
a registered stopping condition and received credit. There are no
Native-correct/Strassen-incorrect cases in this batch.

**All eight first extracted answers are identical between algorithms.**
Generated reasoning and stopping behavior can nevertheless differ. Both arms
exhausted the budget on `test:155`. Both produced `9.00` for the reference answer
`9` on `test:227`, which the frozen strict string comparison counts incorrect.
These details explain the observed scores; they do not replace them with a
post-hoc numerical-match metric. Extraction, stopping and budget rules remain
unchanged for the continuing study.

Checkpoint: `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Both arms use the same weights,
BF16, DEFAULT precision, batch 8 and greedy generation. S1 uses tile
128/256/256 at q/k/v/o/gate/up/down without fused epilogues. The complete
execution contract and original full manifest are retained here.

## Verification and reproduction

`batches/gsm8k-000.json` preserves raw paired generated text, token IDs, stop
reasons and extracted/scored predictions. `summary.json` contains per-example
descriptive results. `verification.json` records matching input/source hashes,
the unchanged execution contract and fresh train-only admission. The included
runtime log is an earlier immutable snapshot containing that admission; the
later completed batch is preserved separately. It is not a completed-batch log.
`report-execution/` records the successful offline verification command and
committed analysis sources. No TPU computation was launched by this report.

From this directory, with Python, NumPy and SciPy available, reconstruct into
a new temporary directory:

```sh
scratch=$(mktemp -d)
mkdir "$scratch/source"
tar -xzf ../../source.tar.gz -C "$scratch/source"
python code/accuracy_gsm_prefix_report_v001.py \
  --checkpoints . \
  --inputs "$scratch/source/plans/llm_accuracy_20261018_v001/subset_qwen3_14b_v002" \
  --source-archive ../../source.tar.gz \
  --runtime-log runtime-log-20261019T0911Z-v001.log \
  --batches 1 --output "$scratch/reconstructed"
```

The prior complete MMLU/perplexity snapshot and all failed runtime evidence
remain separate. Later GSM8K results must be added in a new snapshot.
