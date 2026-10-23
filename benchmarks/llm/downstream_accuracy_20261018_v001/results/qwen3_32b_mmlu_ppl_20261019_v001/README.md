# Qwen3-32B: completed MMLU and perplexity subsets

Immutable interim snapshot from October 19, 2026. MMLU and perplexity are complete;
GSM8K remains in progress and is excluded here. The overall study has completed
five of nine model/task evaluations. These are accuracy measurements, not timings.

| Metric | Native | Strassen S1 | Difference | Registered 95% interval | Preservation criterion |
| --- | ---: | ---: | ---: | --- | --- |
| MMLU, 114 examples | 96/114 (84.2105%) | 96/114 (84.2105%) | 0 pp | [-25.4396, +25.4396] pp | Inconclusive at 0.5 pp margin |
| WikiText-2 perplexity, 32 windows / 16,384 scored tokens | 6.90390076 | 6.90596510 | +0.00206434 (+0.0299011%) | Relative change [-0.0851445%, +0.1447055%] | Preserved at 1% relative margin |

MMLU has two prediction disagreements: one Native-correct/Strassen-incorrect
(formal_logic, test:2530, gold D; Native D, S1 A) and one
Native-incorrect/Strassen-correct (high_school_world_history, test:6067, gold C;
Native B, S1 C). There are 95 jointly correct and 17 jointly incorrect examples.
Equal aggregate accuracy does not establish equivalence on this small subset.
Subject-level results and both disagreements are in `summary.json`. The
registered equally stratified Hoeffding interval is conservative; the pooled
binomial interval in that JSON is descriptive only, not the decision criterion.

Perplexity uses the frozen paired whole-window bootstrap (20,000 draws, seed
20261018, block size one), with token-weighted NLL. Its absolute-difference
interval is [-0.00586117, +0.01003638]. The registered relative interval rules out
more than 1% degradation on these windows. Conclusions are limited to this
checkpoint, subset, precision and Strassen configuration.

## Matched configuration and provenance

Checkpoint: `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137`.
Both arms use unchanged weights, BF16/DEFAULT, identical examples, tokenizer,
prompts and masks, batch eight, and the same compiled 16-query decoder blocks.
S1 uses one recursion level, tile 128/256/256, for q/k/v/o/gate/up/down projections;
there are no fused epilogues. Hardware is one Colab v6e; software and all settings
are recorded in `execution-contract.json` and the frozen manifest.

The query-block execution repair was admitted on training inputs before testing;
it did not change scoring, benchmark selection, precision, or preservation margins.
See [repair and validation](../../repair_20261019_query_blocks_v002/README.md).
The matching scientific bundle is
[archived here](../../repair_20261019_query_blocks_v002/source.tar.gz), SHA-256
`ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495`.
The execution-contract digest is
`e416613ce19ecc7c55d4b5f1ed8367a534716f5dd40c88d4f33f82042467d1b9`.

`verification.json` records successful input/source/contract/admission checks.
`checkpoint-source/` preserves the 19 original paired batch files and download
provenance. `native.jsonl` and `strassen.jsonl` hold all 146 rows per arm.
`report-execution/` preserves the exact offline reconstruction command, source
and successful receipt, using unchanged `accuracy_analysis_v003.py`.
`artifact-manifest.json` hashes the verified report; `publication-manifest.json`
hashes this complete publication. Earlier snapshots and failed attempts remain
unchanged. GSM8K and full-model completion must be published separately.
