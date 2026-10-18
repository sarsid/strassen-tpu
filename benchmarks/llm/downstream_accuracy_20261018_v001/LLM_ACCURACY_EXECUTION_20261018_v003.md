# Small paired accuracy study: executable admission and analysis

Registered before any test-set scoring. The successful Qwen3-14B train-only
timing pilot is disclosed background: eight prompts, sixteen output tokens,
all 128 selected tokens agreed. This is not a downstream accuracy result.

The original protocol and subset amendment remain in force. This execution
addendum makes the following details explicit before examining test scores:

* The admitted projection implementation is plain parent S1, tile 128/256/256,
  with shared non-MM operations and explicit BF16 boundaries. This tests an
  algorithm-only substitution, not the fused timing study's complete kernels.
* Run Qwen3-14B, Qwen3-32B, then Gemma4-31B in independent processes. Keep all
  114/32/32 registered examples and their original order. Batches have eight
  rows; inactive rows are masked independently and never scored. Fixed padded
  widths per task avoid repeated compilations without truncating real tokens.
* Before each model's test examples, use the first 64 tokens of the eight
  separately saved training prompts to compare Native cached final-token
  logits against full-prefix recomputation. Require finite outputs and relative
  L2 at most 0.005, and record greedy prediction agreement and maximum absolute
  difference. This threshold accommodates BF16 backend reduction-order changes;
  it is not an accuracy-preservation threshold. A failed check stops that model.
  Do not gate admission on Native versus Strassen quality: regressions in the
  actual algorithm comparison must be reported, not filtered away.
* Local evidence covers official Qwen BF16 layer semantics, previously qualified
  Gemma local/global formulas, cache/padding, independent greedy histories,
  EOS/text stops, budget exhaustion and scoring. Archive the source versions
  and exact local admission receipts alongside each runtime contract.
* A maximum-length GSM8K output counts incorrect even if it contains an answer;
  retain both the extracted answer and the ineligible prediction=null. The
  analysis verifies this pre-existing rule rather than extracting anew and
  accidentally granting credit to a truncated completion.
* MMLU samples exactly two questions per subject. Its pooled binomial interval
  is only descriptive because the sample is stratified. Use a conservative
  distribution-free two-sided 95% Hoeffding bound on the equal-subject mean
  paired correctness difference, with half-width sqrt(2 log(40)/114). Uniform
  sampling without replacement within subjects is conservative for this bound.
  Subject counts are descriptive. This intentionally broad interval will often
  be inconclusive; keep the original 0.5pp margin and report observed scores.
  GSM8K retains the paired discordant-cell interval and 1pp margin. PPL uses
  paired whole-window bootstrap with 32 nonoverlapping windows and a 1% margin.

Checkpoint every completed batch into an exclusive file. On process failure,
resume only uncommitted batches using the same source/configuration. A paired
batch is atomic: never mix half of an old pair with a new one. Do not overwrite
failed attempts or historical study data. Preserve per-model artifacts before
removing that run's disposable private weight cache to free checkpoint disk.
Retain the TPU on errors. Release only after all requested model artifacts have
been independently downloaded and hash-verified, or on explicit user request.

The pilot's median paired decode step is 7.1496 seconds for eight examples.
Four groups of 256-token outputs would take about 2.03 hours for Qwen3-14B
generation alone. This extrapolation assumes the longest output in each paired
group is 256 tokens; output lengths and larger-model throughput are unmeasured.
The previous aggregate 3–8 hour range is not established by this pilot.
