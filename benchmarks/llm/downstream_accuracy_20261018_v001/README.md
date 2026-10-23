# Downstream accuracy study — preparation and train-only timing pilot

This immutable initial snapshot contains the registered small-subset accuracy
study and its executable sources. **No downstream test scores are included.**
The three completed LLM timing studies remain unchanged.

Compare pinned Qwen3-14B, Qwen3-32B and dense Gemma4-31B checkpoints using Native
and plain tile-local Strassen S1. Both arms use BF16 inputs/outputs, DEFAULT
precision, identical non-MM operations, and the same checkpoint weights.
This isolates the MM substitution; it does not validate every fused epilogue
from the earlier timing study. S1 tile: 128/256/256, one recursion level.

Per model: 114 MMLU test questions (two per subject), 32 seeded GSM8K test
problems, and 32 nonoverlapping WikiText-2 windows with 16,384 scored tokens.
Batch size 8. GSM8K uses independent greedy generation, five fixed train
examples, and a 1,024-token budget. Complete checkpoint/dataset revisions,
matching rules, preservation margins and uncertainty methods are in the three
protocol files; later amendments take precedence. Small-subset intervals may
be inconclusive even when observed scores agree.

## Measured train-only pilot

Qwen3-14B, eight separate training prompts, first 16 generated tokens:

- Native and S1 selected the same 128 tokens per arm.
- Median paired decode step: **7.1496 seconds**, advancing all eight sequences
  once under each arm. This is elapsed offloaded evaluation time, not kernel speed.
- Weight preparation: approximately 604.7 seconds.
- This is **not MMLU/GSM8K accuracy or an accuracy-preservation result**.
- Four batches whose longest generated sequence is 256 tokens would require
  about 2.03 hours for Qwen3-14B generation alone. Output lengths and throughput
  for the larger models have not yet been measured. The prior provisional
  aggregate 3–8-hour range is not established by this pilot.

The raw pilot archive was downloaded and SHA-256 verified. Runtime retention
was requested, but Colab subsequently listed no assignment; the available logs
do not establish the disappearance's cause. The complete evaluation driver and
three-model queue were prepared after this pilot; the pilot did not automatically
launch the downstream study.

## Reproduction and preservation

`source.tar.gz` is the exact committed study source, inputs and parent kernels;
its hash and source commit are in `source-manifest.json`. Extract it and inspect
`runtime/accuracy_study_remote_v001.py` and `tools/accuracy_study_v001.py`.
The worker pins its software, records execution contracts, runs train-only cache
admission, saves paired batches, and attempts bounded recovery without deleting
failed evidence. Model checkpoints are fetched by immutable revision and kept
outside public experiment archives. Local admission/recovery evidence is under
`validation/`; the timing pilot is preserved under `train_timing_pilot/`.

Future test results will be added separately, without replacing this snapshot
or any earlier benchmark results.
