# Paired downstream accuracy study — preregistration v001

Registered 2026-10-18 before any new MMLU, GSM8K or full-corpus scores. This is
a separate accuracy experiment, not a reinterpretation of the timing study's
four-segment perplexities. No accuracy-preservation conclusion is assumed.

## Checkpoints and scope

| Model | Official checkpoint | Immutable revision |
|---|---|---|
| Qwen3-14B | Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 |
| Qwen3-32B | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 |
| Gemma4-31B | google/gemma-4-31B | 5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89 |

The public Hub API confirmed all three identities/revisions; receipts and
config/tokenizer/generation files are in plans/llm_accuracy_20261018_v001/metadata.
Gemma is the dense checkpoint's text component, not Gemma3, an instruction-tuned
substitute, a quantization or an MoE. Its pinned tokenizer supplies no chat
template. Both Qwen checkpoints are the exact previously studied checkpoints.

Compare Native and tile-local classical Strassen S1. Use identical unmodified
checkpoint tensors, BF16 parameters/activations/outputs and FP32 dot accumulation,
DEFAULT dot precision. Attention score/value products, embeddings and LM head
remain native in both arms. The intended substitutions are decoder linear
projections, including Q/K/V/O and FFN gate/up/down where supported; actual
per-site coverage and native fallbacks must be recorded. Strassen must operate
during both prompt processing and generated-token processing. No gold-answer
teacher forcing may substitute for GSM8K generation.

Non-matmul operations, their order and BF16 rounding boundaries must be identical
between arms and match the checkpoint's reference semantics. In particular,
the timing study's Strassen-only QK/RoPE, activation and residual epilogues are
not automatically admitted to this algorithm-only comparison. The parent source
is immutable commit 95be1fb088656a89813b04492e1d77c66b36ccf9. Its plain S1 GEMM
can be reused; its full fused block cannot be assumed to isolate MM.

## Preservation criteria fixed before scores

Define accuracy difference as Strassen minus Native, in percentage points.
Preserved means the lower endpoint of the paired two-sided 95% confidence
interval is strictly greater than -0.5 pp for MMLU, or -1.0 pp for GSM8K.
These practical tolerances correspond to no more than five additional mistakes
per 1,000 knowledge questions and ten per 1,000 math problems, respectively.
The looser math tolerance reflects the smaller benchmark and sensitivity of
multistep generated answers. They are study choices, not universal standards.

For perplexity, preserved means the upper endpoint of the paired two-sided 95%
interval for relative increase is strictly less than 1%. PPL ratio is
exp(delta mean NLL); the margin is log(1.01) nats/token. It is small enough to
exclude a material systematic language-modeling loss without treating PPL as
accuracy. Previous exploratory short-text results are disclosed background,
not samples in this confirmatory evaluation.

If the entire interval lies beyond the degradation margin, report a regression
beyond tolerance. If the interval overlaps the margin, report inconclusive.
A statistically detectable small loss can still satisfy the practical margin;
report that loss. Failure to reject zero difference does not establish preservation.
Overall claims cover only the tested settings. Subject results are descriptive,
with clearly labeled unadjusted intervals; do not assert simultaneous guarantees.

## Matched benchmark definitions

Reference harness: EleutherAI/lm-evaluation-harness commit
d6de81643928d653435c431bae19945d41d32520. Its task definitions are archived.
The custom TPU runner must identify its own source commit and disclose any
deviation; it must not claim to be the unmodified harness.

* MMLU: cais/mmlu revision c30699e8356da336a370243923dbaf21066bb9fe,
  all 57 subjects, full test split, five dev examples per subject in stored
  order. Standard question/A–D/Answer prompt and subject description. Select
  the largest summed continuation log likelihood among A/B/C/D; no generated
  explanation. Report question-weighted overall accuracy, subject scores/counts,
  macro subject accuracy as secondary, and paired prediction disagreement.
* GSM8K: openai/gsm8k revision 740312add88f781978c0658806c59bc2815b9866,
  main configuration, all 1,319 test examples. Five train demonstrations chosen
  once using seed 20261018; save their dataset row IDs and exact rendered prompts.
  Standard Question/Answer format with reference train solutions ending ####.
  Greedy decoding, no sampling/beam search, max 1,024 new tokens. Stop on the
  pinned checkpoint EOS token(s), Question:, </s>, or <|im_end|>. Strict-match
  extraction (first `#### (-?[0-9.,]+)`) is primary. Normalize commas, dollar
  signs and a final period as in the pinned task. Missing/invalid answer or
  exhausted budget counts as incorrect, never omitted. Record stop reason and
  generated-token count; flexible extraction is optional labeled secondary.
* PPL: Salesforce/wikitext revision b08601e04326c79dfdd32d625aee71d232d685c3,
  wikitext-2-raw-v1 full test. Join stored rows with newline, tokenize once with
  add_special_tokens=False. Fixed context 2,048, stride 512. Predict every corpus
  token after the first exactly once; earlier overlapping context is masked.
  Gemma uses a BOS prefix in each independent window, consuming one context
  position, and does not score BOS; Qwen has no added BOS. Save token and mask
  hashes. Report token-weighted exp(total NLL/scored tokens), not a mean of window
  perplexities. Both arms have exactly the same windows, positions and masks.

Use raw completion prompts for all three models, no chat template, no hidden
reasoning parser, no extra system message or mode switches. This deliberately
matches traditional benchmark completion protocols, not each model vendor's
maximum-quality chat/thinking evaluation. Qwen enable_thinking is not invoked
because apply_chat_template is not used. Prompt text and task mode stay fixed
within each pair; do not compare absolute accuracy across models as a ranking.

Fix context limit 8,192 for downstream tasks. If any prompt plus generation
budget exceeds it, stop admission before scoring and register a versioned
length-only amendment; no silent truncation or discarded examples. Target
inference batch size 8, fixed row ordering and right-padding for scoring, with
explicit padding masks/position IDs. Generation requires a validated common
padding/cache policy. If memory forces a smaller batch, amend and freeze it for
both arms before scoring any test examples; never adapt it according to scores.

## Pairing and uncertainty

Store example ID, subject, gold answer, prompt/token hash, prediction, correctness,
generation/likelihood diagnostics and all per-example quality records for both
arms. Reject duplicate IDs, missing pairs, changed prompts/golds/token counts or
incomplete splits before producing an overall preservation verdict.

For accuracies, construct a conservative paired 95% interval for
p(Native-wrong/Strassen-right) minus p(Native-right/Strassen-wrong).
Use simultaneous Clopper–Pearson binomial bounds on the two discordant-cell
probabilities (four tails, alpha/4), then difference their bounds. This retains
nonzero uncertainty when no disagreements are observed, unlike a degenerate
ordinary paired bootstrap. Report all four correctness cells and total changed
predictions, including both-wrong prediction changes. Subject CIs use the same
method without a preservation claim.

For PPL, preserve each window's NLL sum and scored count. Paired moving-block
bootstrap contiguous groups of eight windows, 20,000 resamples, seed 20261018,
recomputing token-weighted losses and exponentiating each draw. Both arms use
the same resampled groups. This accounts partially for local text dependence;
it is an empirical corpus interval, not a guarantee for arbitrary future text.
Report paired absolute and relative PPL intervals and delta NLL. Require at
least 20 groups for a preservation decision; otherwise report inconclusive.

## Admission, reproducibility and operation

Before test scoring: validate paired routing and dtype/rounding contracts;
validate Native formulas against official Transformers; validate cached greedy
generation against full-prefix recomputation; validate padding independence,
EOS/budget/stop/extraction and PPL masking; run only dev/train smoke examples.
Freeze protocol, implementation commit, checkpoint and dataset receipts, software
lock, site/tile routing and device topology. Save JAX/jaxlib/libtpu/Transformers/
tokenizer versions, compiler flags, seeds and exact launch commands. No numerical
gate can be silently relaxed. New failures and amendments get new immutable files.

Existing Colab supports one v6e chip. Neither larger BF16 checkpoint is fully
resident on that chip, and the existing timing harness has no KV-cache decoder.
Layer-offloaded autoregressive generation is possible but may require days.
Do not launch the old prefill worker as though it executes GSM8K. No new TPU
is allocated until a real executable path passes admission. User's pending
runtime information can affect deployment, not preservation margins or scoring.

Maintain a stage/model/benchmark/example-count live log; preparation is 0/9
evaluations complete. Save restartable per-example outputs and cumulative work,
retain an owned TPU on error for diagnosis, and release after verified archival.
Preserve and commit new results separately; never replace the three timing studies.

Required final table: Model | Benchmark | Native | Strassen | Difference | paired
95% CI | Meets preservation criterion. Missing results are "not run", not zero.
