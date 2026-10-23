# Matched Qwen3-32B v6e tuning decisions

The reproduction is frozen at parent public-preview commit
95be1fb088656a89813b04492e1d77c66b36ccf9. Its original code and timing loops run
unchanged; the wrapper replaces public HTTP transport with validated cached byte
ranges. The original and current JAX/libtpu environments are separate processes
and separate result groups. None of their timing differences are attributed to
our kernel changes.

Our adapter uses the parent's attention implementation, RMSNorm, RoPE, model
dimensions, checkpoint revision and evaluation token tensors. Its Native path
must match the parent before tuning proceeds. It deliberately does not reuse
the different attention and strict BF16 rounding contract in model_fused_v003.
Custom epilogues operate on FP32 accumulators, matching the parent's fused
experiment; their differences from the parent's BF16 Native boundaries are
measured, not described as algebraically identical floating-point computations.

An additional precision difference is deliberately retained in the qualified
kernel control: our fused Q/K segmented norm dots explicitly use HIGHEST,
whereas the parent's fused norm dots omit precision and use DEFAULT. On TPU,
DEFAULT float32 dots use BF16 internal arithmetic; HIGHEST uses additional
passes for greater precision ([JAX documentation](https://docs.jax.dev/en/latest/201/precision.html)).
Thus the custom comparison has the same parent attention and unfused model
primitives, but is not identical to the parent's fused normalization arithmetic.
The metadata records this difference. Neither its runtime nor its error effect
may be attributed solely to Strassen recursion or tile selection. The exact
parent reproduction retains the original lower-precision fused norm unchanged.

## Search and selection

* Tune batch 8, sequence 1024 on v6e, using actual layer-0 weights and activations
  produced from the parent's repetitive-text embedding input. Do not infer fused
  winners from random-matrix measurements or transplant v5e settings.
* Search q, k, O, gate/up and down separately. V stays Native, matching the parent
  scope. A family can also retain Native at any other site. Thus a selected S1 or
  S2 model profile is a documented mixture of projections, not a claim that every
  multiplication uses Strassen.
* Offer full contraction alongside 1024-wide K panels. For S2, 1024 leaves K=256
  after two recursion levels, avoiding the parent's 512-panel example whose
  K=128 leaves underfill a v6e MXU. Full-K candidates also have K leaves divisible
  by 256. M/N geometries include the parent winners and smaller alternatives;
  actual timing decides whether narrow output leaves or special 2560-wide tiles
  are worthwhile.
* Compare blocked cubic and full-tile cubic, both product/output accumulator
  strategies for S1/S2, and one/two input buffers. Offer early finalization only
  for S1 gate/up with output accumulators. Depths 3/4 are excluded.
* The pilot retains our 112 MiB per-kernel budget as a control and also offers
  the parent's 120 MiB Strassen / 124 MiB cubic allowances. The bounded complete
  screen uses 120/124 MiB for full contraction and 112 MiB for short K. These
  kernel budgets are independent of Native's whole-function scoped-VMEM flag.
  The fused scratch-memory estimate is a pruning heuristic, not proof of
  feasibility; compilation failures remain in the candidate ledger.
* Tune Native's whole-layer scoped-VMEM setting over unset, 32, 48, 64, 96 and
  112 MiB. All arms retain the parent's enhanced-launch-barrier flag. Therefore
  “Native default” here means the default scoped-VMEM setting, not an environment
  with every TPU flag unset. Use the selected whole-layer setting in fused
  screening, confirmation and quality evaluation. This is a bounded search;
  it does not exhaustively optimize every compiler flag for every hybrid policy.
* Screen each candidate after two warmups with eight synchronized samples in
  v004. Its Native-normalized scores only propose candidates: the final v005
  selection uses a separate 32-round interleaved shortlist measurement, as
  detailed below. Keep Native when it wins. Require finite outputs and
  projection relative L2 error below 0.10 as a development eligibility check.
  This loose screening limit is not the final prediction-quality gate.
* Freeze site choices before testing the complete block on the parent's separate
  deterministic hidden-state input. Confirm with three warmups and 30 rotating,
  reversed-order samples. Include default and tuned Native, Native at the
  parent's 48 MiB setting, and the unmodified parent streamed S1 policy as fresh
  controls. Do not alter the profile using these confirmation measurements.

## Timing and accuracy contracts

The v002 search was preserved as a pilot after measured compilation cost showed
that its 728 candidates would exceed its 70-minute tuning deadline. The Q/BF16
group was completed before the planned stop; partial subsequent work is not a
completed model result. The final v003 menu removes repeated full-K memory-cap
trials, uses 120/124 MiB for full K, retains one/two-buffer choices for full K,
and uses two buffers for short-K pipelining. It has 364 candidates across the
five sites and two output dtypes. Its tuning timeout is 90 minutes, within the
controller's 150-minute overall bound. All families, output dtypes, raw samples,
error gates and independent block/64-layer/WikiText confirmations are retained.
The smaller menu is a declared budget choice, not a claim that omitted choices
cannot win. Pilot and final measurements remain separate.

The pilot also exposed alternating fast/slow Native projection timings. A
three-sample median control could favor whichever candidate happened to follow
a slow control batch. Final tuner v004 retains the 364-candidate menu and uses
eight-sample arithmetic means for candidates, Native controls and Native option
selection. Eight observations balance the observed two-call pattern. Raw
samples remain available, and the independent whole-block confirmation still
uses 30 rotating/reversed-order samples. The v003 runtime lost its connection
during bootstrap, before measurements, and was released; v004's controller
retries this idempotent bootstrap up to three times before releasing on failure.

BF16 and FP32 custom projection stores are separate profiles. Both feed the
parent's BF16 model consumers, and the conversion is inside the timed call.
Native fallback sites retain the parent's BF16 projection behavior. This tests
two implementation choices for a BF16 LLM; it is not a comparison of an entirely
FP32 LLM against an entirely BF16 LLM.

Weight padding and packing are explicit preparation outside resident timing.
Activation padding, output cropping, Q/K layout restoration, epilogues and
consumer casts are timed. Identical prepared layouts share storage within one
layer; no packed weight is reused across different checkpoint layers. Cubic
receives the same fusion opportunities as Strassen, including Q/K and O residual
fusion that the parent's cubic control did not use.

For each frozen profile, reuse the parent's 64-layer timing loop and its exact
repetitive-text and WikiText scoring positions. Record raw samples, per-layer
hidden-state errors, logit errors, NLL, perplexity, KL, top-token agreement and
finite checks. The task gates are declared in the protocol, not tuned using the
evaluation corpus. Failed or slower candidates are evidence, not missing rows.

Summed resident compute excludes downloads, compilation, weight preparation and
transfer, embedding lookup and final scoring. Preserve those costs separately.
A positive result establishes this fixed prefill workload, not arbitrary decode
batches, serving latency, all LLMs, or independently verified official-model
accuracy. No v5e follow-up is queued by this experiment.

## Interleaved shortlist refinement (v005)

The v004 screen exposed a further failure mode of short-control normalization.
For K/FP32, one S2 candidate took 0.51092 ms on average; its eight Native control
calls included a 2.90966 ms outlier and averaged 0.81301125 ms. Division by that
slow control produced an artificially favorable 0.32964 ms selection score.
The raw samples remain preserved. That score is not an achieved runtime.

A separate v005 refinement retains, per site/dtype/family, the two lowest raw
mean candidates, the two lowest symmetrically trimmed candidate/control ratios,
and the v004 winner (deduplicated). Native remains eligible. These rules define
only a shortlist, not a performance result. All shortlisted executables are
compiled first and then measured together for 32 rounds with rotating/reversed
order. Selection uses the median of eight four-round means, without division by
a Native control. Four-round means average the observed alternating phases;
the median limits isolated stalls. All 32 samples and arithmetic means are
saved, including outliers. Independent whole-block confirmation still reports
30-sample arithmetic means and cannot change the selected profile. The Native
compiler cap stays frozen from the original screen. No quality-evaluation
outcome is used to tune tiles.

This is a declared correction to timing methodology, not permission to keep
retuning until a desired speedup appears. The complete original screen and any
original profile evaluation remain separate evidence.

To avoid discarding the completed screen or downloading the checkpoint again,
an explicitly archived supervisor handoff waits for the isolated v004 tuning
child to finish successfully, preserves its profile and independent block
confirmation, then uses a separately frozen v005 source archive on the same
runtime. The old supervisor launches no model evaluation. The new supervisor
refines the profile before the eight model stages. The existing local controller
still downloads the complete evidence and releases the one owned TPU. Source
hashes, process identities and the handoff receipt are included in the archive.

## Bounded cap audit and one timing repeat

Review of the complete screen found down-tile candidates requiring 113 or
118 MiB that failed the chosen 112 MiB short-K cap. The parent permits 120 MiB
for Strassen and 124 MiB for cubic on v6e. A separate down-only audit retries
exactly tile (2048, 2560, 1024), two buffers, both output stores and each family's
two accumulator modes at those parent allowances. It retains the previous down
choice as a measured control and changes a policy only for a score improvement
greater than 1%. All other sites, Native options and norm precision stay frozen.
Changed policies receive new full-model speed and quality measurements; unchanged
profiles reuse their already completed evaluation. The main study is preserved.

The first main S1/BF16 streamed pass contains one 197.153625 ms synchronized
sample at layer 51, surrounded by four samples of 13.824629–13.853529 ms.
With the unchanged parent five-sample mean, this makes that layer's mean
50.5029582 ms and makes the S1-versus-cubic performance interval inconclusive.
The original raw result and failed performance gate are retained. Exactly one
additional unchanged-profile streamed pass is declared, after the main study,
to assess timing stability with the checkpoint cache already populated. It is
reported separately, not substituted for the original pass or used to tune tiles.
The cause of the isolated stall is not established by these timing records.

The audit completed: nine arms compiled, including the old S1 control, and four
failed at the larger allowances. Only S1/BF16 down changed, to the product
accumulator at 120 MiB. Its interleaved selection score was 2.78659 ms versus
2.81873 ms for the old output accumulator. The successfully compiled S1/FP32
output candidate took 2.98921 ms against Native's 2.92505 ms, so FP32 retained
Native. Larger allowances did not make every candidate feasible: spilling pushed
some product variants beyond physical VMEM, and other S2 variants exceeded the
120 MiB cap. No claim is made that all possible larger-cap/tile combinations
were exhausted.

The unchanged S1/BF16 streamed repeat measured 887.782 ms and the revised policy
884.971 ms; the latter passed the independent quality gate. The approximately
0.32% time reduction is small and did not appear in the separate whole-block
confirmation. Both remain behind the parent. The final
[combined report](../results/v6e/parent_llm_study_20261002_v001/README.md) preserves
all passes, precision differences, numerical errors and the remaining scope
limitations. The runtime has been released; this record queues no further work.
