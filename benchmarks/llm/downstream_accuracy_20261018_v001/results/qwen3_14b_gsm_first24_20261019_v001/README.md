# Qwen3-14B: first twenty-four paired GSM8K examples

This immutable interim snapshot contains three completed batches of the
registered **32-example** GSM8K subset. The final batch was running at 16:56 UTC
on October 19. The GSM8K task and the three-model study remain incomplete.
Confidence intervals and the preservation assessment are deferred until the
complete registered subset; this ordered prefix does not establish equivalence.

| Fixed scoring rule | Native | Plain Strassen S1 |
| --- | ---: | ---: |
| Correct final predictions | 14/24 (58.33%) | 15/24 (62.50%) |
| Reached the 1,024-token generation limit | 8/24 | 6/24 |
| Correct in newly completed batch 3 | 5/8 | 4/8 |

There are two Native-incorrect/Strassen-correct cases and one
Native-correct/Strassen-incorrect case. All three correctness differences
involve the same extracted answer but different stopping behavior:

- `test:100` and `test:128`: Native exhausted the generation budget, while
  Strassen stopped normally and received credit.
- `test:1026`, new in batch 3: both extracted the correct answer `43200`.
  Native stopped normally after 629 tokens; Strassen reached the 1,024-token
  cap and was counted incorrect.

**First extracted answers match on 23/24 examples.** The exception remains
`test:380`: Native extracted the reference answer `803` but reached the cap;
Strassen extracted `760.42` and stopped normally. Both are scored incorrect.
This numeric-answer disagreement is retained even though it does not change
their paired correctness score. There are four different scored predictions
in total, three with different correctness.

Both arms also produced `9.00` versus reference `9` on `test:227` and `6.00`
versus `6` on `test:258`; the frozen strict string rule counts these incorrect.
No post-hoc numeric normalization or budget exceptions are applied. Raw text,
token IDs, extracted answers, stop reasons and scored predictions are saved
in `batches/`. These results do not establish a reasoning or accuracy advantage
for either algorithm.

Checkpoint: `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Both arms use identical weights,
BF16, DEFAULT precision, batch 8 and greedy generation. S1 uses tile
128/256/256 at q/k/v/o/gate/up/down without fused epilogues. The original full
manifest and execution contract are retained.

## Recovery and verification

Batch 1 was recovered from the previous runtime; batches 2 and 3 completed on
v006. The regenerated execution contract and recovered batch were verified
byte-identical, with fresh train-only admission on v006. Scientific source,
inputs, settings and scoring were unchanged. The included earlier runtime log
contains admission, not the later batch completions. The observed-progress
receipt records the transition to batch 4.

`verification.json` records source/input hashes, paired-row verification and
fresh admission. `report-execution/` archives the successful offline verification
and committed report sources. No TPU computation was launched by this report.
The first-eight, first-sixteen and complete MMLU/perplexity snapshots remain
separate and unchanged.

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
  --batches 3 --output "$scratch/reconstructed"
```

Later results must be added in another snapshot. This folder does not replace
the final registered 32-example analysis.
