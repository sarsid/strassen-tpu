# Larger real LLM speed and error study

Requested 2026-09-22: determine whether Strassen provides useful real-model
speedups and quantify the accuracy/numerical cost. This is a new experiment;
prior v5e measurements remain separate.

Models: official Qwen3-8B, Mistral-7B-v0.3, Gemma 3-12B text backbone and
Qwen3-14B, with immutable revisions in prepare_large_models_v002.py. All exact
BF16 checkpoint matrices and tokenizers. No quantized or smaller substitutes.
Use one Colab v6e when available. Any v5e fallback is a separately named cohort;
full-residency memory failures are reported explicitly, never streamed under
a resident label. No new paid subscription or compute-unit purchase is implied.

Policies: default Native, independently tuned Native, cubic, Strassen1,
Strassen2, plus an unfactored full-JIT Native control. Keep BF16 inputs and
Strassen pre-additions, FP32 MM accumulation/output. Same attention, nonlinear
functions, normalization and residual operations for every policy.

Independently qualify Native against official Transformers/PyTorch CPU BF16
execution on 64 identical tokens. Preserve prior qualification gates. Tune
the two MLP projections at M=2048 using the same 52 registered candidates
(four Native settings and sixteen tiles per custom family). Freeze choices
before fresh-input confirmation; retain all failures and ineligible policies.

Tuning uses WikiText2 test windows0..3. Model quality uses windows16..31,
16 disjoint 2048-token contexts, 32752 next-token positions. It is held out
from kernel tuning, but remains one text corpus, not a downstream task suite.

Report ground-truth top1 and top5 accuracy, NLL/perplexity, correct-to-wrong and
wrong-to-correct transitions, agreement with Native, forward/reverse KL,
Jensen-Shannon divergence, total variation, multiclass Brier score, 15-bin
expected calibration error, and probability confidence/margins. Report final
logit relative L2, RMSE, MAE, max absolute error and cosine similarity. Retain
per-token statistics and quantiles, with paired whole-window bootstrap
intervals for NLL and accuracy differences. Record full-output finiteness.
Kernel error uses the existing FP64 sampled cross-product reference with all K.
Layer diagnostics separately record local error on identical Native input and
accumulated error on each algorithm's own propagated state.

The existing acceptance gates remain |delta NLL|<=0.01, KL<=0.02 and
top1 agreement>=97%. Added diagnostics are not new pass thresholds. Never hide
a fast but inaccurate algorithm; explain why an ineligible policy is unavailable.

Timing runs in a fresh process after quality. Load and lay out every layer,
embedding table and vocabulary head once outside timing. Require the complete
weight working set plus 2 GiB workspace to fit the reported device memory limit.
Record host reading/layout and host-to-device transfer setup separately.

Measure two real operations at batch1/context2048: prompt forward through the
last-token vocabulary logits, and teacher-forced next-token scoring through
all vocabulary logits and loss evaluation. Device embedding lookup, all layers,
normalization/head, Python dispatch and final synchronization are timed. No
checkpoint access, layout conversion, weight transfer or compilation inside
the timer. Two warmups per policy/scope, nine randomized paired rounds over
three held-out windows. Compare against every Native control using paired
round bootstrap intervals. This is an unfused research implementation, not a
production serving engine; persistent KV caching, decode latency, free-running
generation quality and downstream task accuracy are not measured by this version.

Use independent tune, quality and resident processes/stages to limit memory
and retain completed evidence if an allocation disappears. Archive source,
configurations, exact input hashes, raw samples and errors per stage. Do not
silently pool hardware generations or allocations. Release only the owned
allocation after retrieval. Existing Gemma checkpoint credential authorization
persists; credential contents stay in the private runtime root outside archives.
