# Gemma4-31B: completed MMLU and perplexity subsets

Immutable interim results from October 21, 2026. MMLU and perplexity are complete;
GSM8K is still running and is excluded. Eight of nine model/task evaluations are
complete across the three-model study. These are accuracy measurements, not timings.

| Metric | Native | Strassen S1 | Difference | Registered 95% interval | Preservation criterion |
| --- | ---: | ---: | ---: | --- | --- |
| MMLU, 114 examples | 95/114 (83.3333%) | 94/114 (82.4561%) | -0.8772 pp | [-26.3168, +24.5624] pp | Inconclusive at 0.5 pp margin |
| WikiText-2 perplexity, 32 windows / 16,384 scored tokens | 4.94330423 | 4.94008936 | -0.00321487 (-0.0650349%) | Relative change [-0.1459400%, +0.0104159%] | Preserved at 1% relative margin |

MMLU predictions agree on 113 of 114 questions. The single disagreement is
`moral_scenarios`, `test:8721`, gold C: Native predicts C and S1 predicts D.
There is one Native-correct/Strassen-incorrect case and no reverse gains;
94 questions are jointly correct and 19 jointly incorrect. The observed accuracy
loss is one question. This subset does not establish preservation within the
registered 0.5 percentage-point margin or a statistically established regression.
The registered equally stratified Hoeffding interval is conservative. The pooled
interval in `summary.json` is descriptive only and is not the decision rule.
All subject-level scores are in `summary.json`.

Perplexity uses the frozen paired whole-window bootstrap: 20,000 draws,
seed 20261018, block size one, token-weighted NLL. The absolute difference interval
is [-0.00733526, +0.00051372]. The relative interval rules out more than 1%
degradation on these windows. Lower observed S1 perplexity is not evidence of a
significant improvement; the interval includes zero. Perplexity is not an accuracy
percentage. Conclusions apply only to the evaluated checkpoint, tasks, precision,
and Strassen configuration.

## Configuration and recovery provenance

Checkpoint: `google/gemma-4-31B` at
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`, dense text path.
Both arms use identical weights, BF16/DEFAULT, prompts, tokenizer, examples,
masking, batch eight, and compiled 16-query decoder blocks. S1 uses one recursion
level, tile 128/256/256, at q/k/v/o/gate/up/down projections, with no fused epilogues.
Hardware/software and all settings are in `execution-contract.json`.

The first 13 MMLU batches (104 paired questions) were produced on the earlier
v010 runtime, then copied byte-for-byte from the frozen recovery seed. The final
two MMLU batches (10 questions) and all four perplexity batches (32 windows) were
produced on the replacement v011 runtime. `checkpoint-source/provenance/batch-origins.json`
records the origin and hash of every batch. The execution contract is unchanged.

**Original-runtime evidence limitation:** the saved original Gemma manifest records
`admission_status=passed`, but detailed original numerical admission and preflight
receipts were not downloaded before the runtime disappeared. This gap remains
explicit in `checkpoint-source/provenance/original-runtime-evidence.json`.
Fresh passed preflight and admission receipts establish the replacement runtime's
validation only; they must not be presented as the original runtime's receipts.

Before continuing, the replacement passed unchanged training-only preflight at
64, 129 and 1,024 tokens and fresh runtime admission. Native cached/full outputs
match exactly; each arm's cached/full outputs also match exactly at 129 and 1,024
tokens. All tested logits are finite. This internal-consistency check does not
assert that Native and S1 outputs agree with each other. Both complete Qwen
studies were restored unchanged and were not recomputed.

Scientific bundle SHA-256:
`ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`.
Execution-contract digest:
`b0501f542a0d85b6fb05901579a45d33a413d192f3cdf25a2b2deb8517536e13`.
The unchanged scientific bundle and separate actual operational worker are
preserved in [Gemma recovery](../../recovery_gemma_20261021_v001/README.md).

## Reproduction and files

`native.jsonl` and `strassen.jsonl` contain 146 scored rows per arm. `batches/`
and `checkpoint-source/` preserve the 19 complete paired batches. `inputs/`
contains the frozen full planned input manifest and examples; GSM8K has not been
included in this result report. `runtime-admission.json` is the fresh replacement
receipt. `verification.json` records passed source/input/contract checks.
`analysis-source/` contains the exact committed offline verifier and frozen
analysis; `report-completion.json` records its successful execution.

From this directory, with the study's NumPy/SciPy analysis environment:

```sh
python analysis-source/accuracy_checkpoint_report_v001.py \
  --checkpoints checkpoint-source \
  --inputs inputs \
  --source-archive ../../recovery_gemma_20261021_v001/source.tar.gz \
  --admission checkpoint-source/provenance/admission-20261021T065510Z-20a85c.json \
  --output reproduced-report
```

Use a new output directory. `artifact-manifest.json` covers the original verified
report; `publication-manifest.json` covers this entire publication. Previous
snapshots remain unchanged. GSM8K and complete-model results will be separate
publications.
