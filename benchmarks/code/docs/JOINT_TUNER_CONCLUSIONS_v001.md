# Joint v6e tuner: results and design rationale

This is a **partial study: three of six planned shapes**, stopped at the user's
request to preserve token quota for cleanup and documentation. The TPU was
released and verified absent before the report was generated. Tall
32768 × 8192 × 2048, K=12288, and the 16384 cube remain unrun.

The completed subset supports S1 for every tested shape/output-precision pair.
The expanded tuner found a modest new FP32 improvement on the 8192 cube; the
other S1 choices retained previously known configurations. This is useful
evidence for selective tuning, not a claim that the new strategy improves
every shape.

## Confirmed running times

Matrix dimensions below are **M × K × N**. Each entry uses three fresh inputs
and 30 paired timing rounds per input. Speedup is Native latency / S1 latency.
Native means the frozen winner among the registered Native compiler-memory
settings, rather than an assertion of the best possible cubic implementation.

| Shape | Output | Native ms | S1 ms | S2 ms | S1 speedup |
|---|---|---:|---:|---:|---:|
| 8192 × 5120 × 51200 | FP32 | 5.46982 | 4.72724 | 5.11263 | 1.1571× |
| 8192 × 5120 × 51200 | BF16 | 5.20188 | 4.66160 | 5.01409 | 1.1159× |
| 8192 × 8192 × 8192 | FP32 | 1.58899 | 1.48591 | 1.60297 | 1.0694× |
| 8192 × 8192 × 8192 | BF16 | 1.58277 | 1.46274 | 1.48996 | 1.0821× |
| 2048 × 8192 × 32768 | FP32 | 1.52527 | 1.40269 | 1.48346 | 1.0874× |
| 2048 × 8192 × 32768 | BF16 | 1.54822 | 1.40359 | 1.47573 | 1.1030× |

S2 beat Native in five of these six comparisons, but never beat S1. On the
FP32 8192 cube, S2 was about 0.9% slower than Native. These are warmed resident
MM timings, including final output conversion and excluding compilation and
transfers; they are not end-to-end LLM results.

## What the tuner chose, and why

Kernel tile dimensions are **BM × BN × BK**: output rows, output columns,
then contraction-panel length. All selected S1 configurations use two buffers.

| Shape | FP32 S1 choice | BF16 S1 choice |
|---|---|---|
| Parent projection | Products; 2048 × 1024 × 5120; full K | Same |
| 8192 cube | Direct output; 2048 × 2048 × 4096; two K panels | Products; 2048 × 2048 × 2048; four K panels |
| Wide projection | Products; 2048 × 1024 × 8192; full K | Same |

**Search the interacting choices together.** Larger output tiles and longer K
panels can improve useful work per tile but consume more working memory.
Two input buffers also consume memory. Accumulator layout can determine whether
the desired tile and buffering combination fits. Optimizing these settings
independently would miss that interaction.

**Keep both accumulator layouts.** The products implementation retains seven
outer products in FP32 and reconstructs at the end. The outputs implementation
updates four result quadrants during each K panel: directly in FP32 output, or
in four FP32 scratch quadrants before the final BF16 store. Outputs saves live
scratch storage; products postpones repeated outer reconstruction. The code
establishes this tradeoff, while timing determines which implementation wins.

The FP32 cube gives the strongest concrete example. Its selected direct-output
tile compiled successfully. The seven-product implementation at exactly the
same tile and two-buffer setting required 128.87 MiB against a reported
127.94 MiB total VMEM capacity and failed compilation. Direct output therefore
made a useful configuration feasible. Its confirmed speedup over the replayed
previous S1 pilot was **1.01225×**, with paired 95% interval
**[1.00588, 1.01952]**. The same-tile cubic control also failed to fit; there is
no valid timing comparison against that particular control.

**Do not assume smaller storage is faster.** On the parent FP32 projection,
seven-product S1 was about 1.26% faster than direct-output S1 at the same tile.
The corresponding BF16 S1 comparison was unresolved. On the wide BF16 shape,
the two S1 layouts were also statistically unresolved. Four-quadrant storage
improved parent BF16 S2 by about 1%, but S1 remained faster overall.

**Keep buffering in the search.** At the selected S1 tile, two buffers gave
1.33–1.60× the speed of one buffer across these comparisons. That is roughly
25–38% less latency, not 33–60% less latency. This is measured evidence for the
configuration choice; attributing the entire gain to a specific overlap or
stall mechanism would require profiling.

**Choose contraction by the complete shape and output contract.** Full K won
for both projections, whereas shorter panels won for the cube. The cube and
wide projection have the same conventional FLOP count and output size, yet
select different configurations. Total FLOPs or matrix volume alone is an
insufficient dispatch rule. FP32 and BF16 output can also select different
layouts and panel lengths on the same shape.

## Why the selection procedure is designed this way

The tuner uses a bounded registered tile menu and selected aligned K divisors,
including full K and every historical control. It records omitted divisors and
every pruning decision. A rough footprint above 160 MiB prunes candidates
against the common 112 MiB custom-kernel allowance; historical controls bypass
that heuristic. The estimate is not a proof of compiler feasibility or
infeasibility, and the search is not exhaustive.

Screening uses seven timed rounds and ranks latency relative to Native_default
in the same randomized batch. This mitigates drift. Only complete measurements
passing the numerical gate may win. The tuner freezes Native, overall, each
depth, and each depth/layout/contraction-stratum winner before confirmation.
It then measures historical controls and matched storage/buffering alternatives
on three fresh inputs. Confirmation does not reselect the winner.

The saved recommendation requires the frozen custom winner to pass every
confirmation error gate, beat tuned Native by at least a 1.01 mean speedup
ratio, and have a paired 95% lower bound above one. Otherwise it falls back to
Native. The 1% margin is an engineering policy, not a hardware constant.
All six completed shape/precision pairs passed this rule with S1.

This is an **offline tuner**. For the first two screening phases, successful
compilations alone took about 30 minutes per shape; all timed calls together
took under 30 seconds per shape. Save and reuse the resulting choices rather
than repeat this broad search for each call. The emitted policy is a research
artifact; it has not been installed as a general application dispatcher.

## Numerical cost and limits

All algorithms use the same BF16 operands. Strassen pre-adds are BF16;
products, accumulation and reconstruction are FP32. Output conversion is
included in timing and error. Errors are relative to an FP64 product of those
same quantized BF16 operands, using 128 sampled rows × 128 sampled columns and
every K term. Full-output finiteness is also checked.

Across the completed shapes, worst-per-seed relative L2 errors were:

| Algorithm | FP32 output | BF16 output |
|---|---:|---:|
| Native | approximately 0.00001% | 0.167–0.169% |
| S1 | 0.440–0.443% | 0.469–0.471% |
| S2 | 0.975–0.987% | 0.989–1.002% |

Every successfully executed confirmation case passed the declared numerical
gate. The report preserves relative L2, maximum absolute error, RMSE, mean
absolute error, p50/p99 absolute error and normwise error. Passing the gate
does not mean exact arithmetic or certification for arbitrary inputs.

There were 1,333 screening outcomes (1,250 successful and 83 memory failures)
and 285 confirmation outcomes (279 successful and six memory failures).
The six confirmation failures were two matched FP32 cube alternatives,
repeated across three inputs. Compiled smoke validation was separate: 34/34
passed. Failed candidates remain in the archives.

These are three development shapes with Gaussian inputs. Confidence intervals
are pointwise and conditional on screening, with no multiplicity adjustment.
The study does not establish cancellation robustness, accuracy on real model
activations, unseen-shape performance, or LLM prediction quality. The remaining
three shapes must be measured before drawing conclusions about the full plan.

## Saved evidence

- Results and per-depth matched comparisons: `runs/20260924T231451Z-joint-partial-results-v001-cc6665/artifacts/RESULTS.md`
- Protocol decisions: the adjacent `TUNER_DESIGN.md`.
- Every candidate's registry, screen observations and exclusion reasons: the adjacent `candidate_decisions.json`.
- Full confirmation statistics and errors: the adjacent `results.json`.
- Six validated research recommendations: the adjacent `tuner_policy.json`.
- Verified TPU release: `runs/20260924-joint-v001/operations/quota-closeout-v001/release.log`.
- Shutdown and deferred shapes: `runs/20260924-joint-v001/user-closeout-v001.json`.

The first shutdown helper refused to signal the already-finished phase process
and exited before release. That control failure is preserved in its log.
The allocation was then released directly and verified absent; all six
completed screen/confirmation phase archives were hash-verified for this report.
