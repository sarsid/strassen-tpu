# Small paired accuracy pilot — amendment v002

Registered 2026-10-18 after the user narrowed the request to a small subset,
before any model scores. Supersedes only dataset scope and PPL resampling in
LLM_ACCURACY_20261018_v001.md; checkpoint revisions, precision, prompts,
generation settings, matched rounding and accuracy margins remain unchanged.

* MMLU: two uniformly selected test questions from each of 57 subjects (114 total),
  random.Random(20261018), subjects in sorted order, selected row IDs sorted.
  Retain five dev demonstrations per subject. Overall pilot accuracy is equal
  subject-weighted by construction; it does not estimate the original test split's
  question-weighted average. Subject rows with n=2 are descriptive only.
* GSM8K: 32 test problems sampled without replacement with the same fixed seed,
  selected dataset row IDs sorted. Keep the same five train demonstrations.
* Perplexity: 32 seeded randomly selected WikiText-2 windows with nonoverlapping
  context spans, drawn from the previously registered context-2048/stride-512
  window construction. Use exactly the same masks/positions within each pair.
  Bootstrap paired whole windows (20,000 draws, seed 20261018), recomputing
  token-weighted corpus NLL and PPL. Report this as subset PPL, not full test PPL.
  Selection is before model scoring and never depends on prediction or loss.

Accuracy margins remain MMLU 0.5 pp, GSM8K 1.0 pp; PPL relative margin 1%.
The conservative paired intervals may not establish these margins with so few
examples, even with zero prediction disagreements. Report inconclusive rather
than widening margins, omitting intervals, or claiming preservation from equality.
CIs describe this pilot sampling design; conclusions do not cover untested data.

Run a short, labeled train/dev generation pilot first. Measure weight loading,
prefill, per-token decode and output length separately before estimating run time.
The earlier days estimate was an unmeasured full-split extrapolation and is withdrawn.
Use the existing Colab deployment for initial qualification; no larger-runtime
decision is needed to prepare or measure this small pilot. No whole-benchmark
scoring or full-split queue is authorized by this amended scope.
