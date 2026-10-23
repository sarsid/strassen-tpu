# Joint accumulator, tiling and buffering study

User-authorized on 2026-09-24, before measurement. Single v6e, existing pinned
JAX 0.11.2 / libtpu 0.0.48 environment. Six contrasting shapes: parent wide
projection, 8192 cube, swapped wide/tall pair, K=12288 and 16384 cube. Each
has separate FP32 and BF16 output comparisons, with BF16 inputs/pre-adds and
FP32 accumulation/reconstruction. Timing includes final output conversion.

Search both Strassen depths 1 and 2, seven-product deferred versus immediate
outer-quadrant accumulation, a bounded expanded output-tile menu, K-panel
length and one/two pipeline buffers. Immediate accumulation writes directly
to FP32 output, or four FP32 scratch quadrants before final BF16 store. Inner
S1 reconstruction for S2 stays inside the K panel. Native default and five
per-executable VMEM settings remain independent baselines.

Retain every registered previous short/full or pilot control as a candidate.
Screening uses at most twelve custom candidates per randomized batch with a
repeated Native_default anchor, three warmups and seven rounds. Rank by the
ratio of candidate latency to that same-batch anchor, excluding incomplete,
failed or numerically ineligible measurements. This anchor normalization is
a drift mitigation, not a guarantee of removing all variation.

The bounded K menu includes full K, the largest aligned divisors no greater
than K/2, K/4 and 1024, plus all control panels. The complete divisor list and
omissions are recorded. A rough footprint above 160 MiB prunes candidates
against the common 112 MiB kernel allowance; historical controls bypass
this heuristic. Estimated footprint is not proven compiler feasibility.

Freeze best Native, overall, S1/S2 and each depth/accumulator/contraction
stratum before confirmation. Rerun on three fresh seeds with five warmups
and thirty paired rotated rounds, including historical controls, selected
S1/S2 at the other accumulator/buffer setting, and matched blocked cubic.
No confirmation-driven reselection. Recommend the frozen custom winner only
if all error gates pass, its mean speedup is at least 1.01, and the paired
pointwise 95% interval's lower bound exceeds one. Otherwise use frozen Native.

The 1% margin is a declared practical policy, not a hardware constant. Error
eligibility retains the previous finite / relative L2 <= 2% / scaled maximum
absolute gate and sampled 128x128 all-K FP64 reference. These synthetic-data
gates do not certify arbitrary inputs or model quality. This is development
shape tuning, not unseen-shape validation. Pointwise intervals have no
multiple-comparison correction.

Deliver raw data, per-candidate pruning/exclusion/selection reasons, all
counterfactuals, a shape-specific research recommendation table, and a
TUNER_DESIGN.md explaining these decisions and measured outcomes. Distinguish
observations from memory/scheduling hypotheses. No LLM fusion, padding,
depths 3/4, alternative formula or transposed-input work in this experiment.

Source is committed before validation/execution. Every phase has its own
immutable archive; failures are retained. One allocation, serial timings,
verified retrieval and automatic runtime release. No new general application
dispatcher is installed by this measurement campaign.
