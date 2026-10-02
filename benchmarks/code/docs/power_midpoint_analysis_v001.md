# Power-and-midpoint study: proposed analysis and plot specification

Status: proposal only. This document contains no new TPU measurements and does not authorize or claim a completed run. The initial campaign samples 48–64 interesting geometries from the full lattice. All timing, feasibility and numerical fields remain unknown until the corresponding checks run. Preserve earlier N1–N9 evidence unchanged; commit the final registered plan and each sealed campaign artifact set as separate immutable revisions.

## Coordinates, lattice and sampling

Use the mathematical convention A∈R^(M×K), B∈R^(K×N), C∈R^(M×N). Display all shape plots in axis order **M, N, K**. Existing runner arrays are ordered `(M,K,N)` and tile arrays `(BM,BN,BK)`; conversions must be named explicitly in data and plotting code.

For i=0,…,16, the intended midpoint is `2^i + (2^(i+1)−2^i)/2 = 3×2^(i−1)`. The i=0 midpoint, 1.5, is not an integer matrix extent and is omitted without rounding. Deduplicating the interval endpoints yields

`D = {2^j : j=0,…,17} ∪ {3×2^(i−1) : i=1,…,16}`.

There are 34 valid extents and 34³=39,304 mathematical shapes. The largest endpoint is 131,072 and largest midpoint 98,304. Full lattice membership does not imply that a matrix can fit in a device allocation or finish within the run budget.

For the first 48–64 sampled geometries, use a frozen, deterministic design with explicit strata: a square size ladder; power/midpoint pairs; fixed-volume or approximately fixed-volume aspect-ratio contrasts; thin-M, thin-N and thin-K cases; axis permutations; large aligned shapes; and resource-boundary cases. Include some unfavorable cases and tiny sizes to locate the overhead/padding regime. Select before new timings. Store a per-shape reason and selection seed. Do not call the sample uniformly random unless that is actually its design, and do not interpret its win fraction as a lattice-wide prevalence estimate.

Reserve a specified subset of new shapes for a final frozen tile-selection rule. Reserve fresh input seeds and fresh timing rounds even for shapes used during tuning. Old N5/N7 shapes are useful anchors but are not new held-out geometries. If follow-up points are chosen after observing a boundary, label them adaptive exploration and keep the original confirmation set separate.

## Baselines, precision and timing scopes

Compare native XLA, tuned full cubic Pallas and tuned one-level Strassen Pallas. Native's internal tile choice is compiler-controlled in the present interface; record it as `compiler_managed`, never assign a fictitious user tile. Any native tuning must enumerate real exposed compiler/configuration choices while preserving the same numerical contract and record each attempted choice. Compiler diagnostics may expose lowering details without establishing a user-controllable tile.

Record exact kernel and dependency hashes, algorithm, variant, tile, original/padded shape, dot precision, input/pre-add/accumulation/output dtypes, compiler flags, TPU/runtime identity and process/allocation identity. The historical contract is BF16 inputs, BF16 Strassen pre-additions, FP32 accumulation/output and DEFAULT dot precision. BF16 pre-add rounding is an additional numerical operation and FP32 output does not remove it. Alternative precision is a separately labeled experiment.

Use identical quantized operands for the three arms within each shape/seed. Keep numerical eligibility fixed before tuning. Report whole-output finiteness and relative-L2/max-absolute error against a reference using the exact quantized inputs. Full references are preferred where practical; sampled output cross-products must retain every K term, save row/column indices and identify their limited coverage. Add several Gaussian seeds and a small prespecified cancellation/dynamic-range stress suite as numerical diagnostics. Failed numerical arms remain visible in performance plots but are ineligible for a winning choice.

Primary latency is a synchronized complete device call, including required device padding/layout work and cropping. Report prepared-kernel latency separately. Compilation time, tuning time, initial host transfer and optional cold-call latency are separate costs. Fresh timing rounds follow tile/variant freezing. Randomize or balance arm order within each round, retain order and raw durations, and use device synchronization appropriate to the runner. Treat tiny operations as an overhead-sensitive regime; if repeated-invocation batching is added, label its scope and repetition count separately.

Device-module profiles are a diagnostic subset, separate from host-observed latency. Select profiles by frozen criteria spanning a win, tie and loss plus at least one large case. Do not subtract separately collected instrumented device timings from ordinary host-call timings to claim measured dispatch cost. Never label compiler cost-analysis fields as measured hardware utilization.

## Tile ranges as feasible tuple sets

For shape s, algorithm a and a finite set of tested valid configurations C_a(s), let `T_hat(c,s)` be its confirmation mean latency. Each configuration includes its variant and the full `(BM,BN,BK)` tuple. Define the descriptive near-optimal set

`E_epsilon(a,s) = {c∈C_a(s): numerically eligible and T_hat(c,s) ≤ (1+epsilon) min_d T_hat(d,s)}`.

Choose epsilon before inspecting new timing; 1%, 3% and 5% can be reported as prespecified sensitivity levels with 3% as a proposed primary level. These are near the best *tested* configuration, not globally optimal. Freeze the shortlisted configurations after screening, then measure them again; do not select and confirm the minimum from the same noisy samples.

Publish the actual tuple/variant members, attempted count, feasible count and confirmation count. Per-axis minima/maxima are projections only: a BM range and a BK range do not imply all their Cartesian combinations are good or even feasible. Display holes, multiple disconnected clusters, and variants explicitly. A candidate with an interval overlapping the best is not proven equivalent. If a confidence-qualified tolerance claim is needed, prespecify a one-sided comparison or equivalence procedure with multiplicity handling; leave insufficiently resolved candidates marked uncertain.

Equal attempted tuning budgets do not guarantee equal feasible search space or equal optimization effort. Record compile failures, numerical failures, OOMs, preflight exclusions and timeouts. Report both separately tuned family winners and a small same-tile/same-accumulator ablation on feasible shared tuples. The latter helps separate arithmetic changes from tile/variant selection; it does not replace the practical full-cubic baseline.

## Analytical features and hypotheses

For a candidate tile b=(b_m,b_n,b_k), define

`M_p = ceil(M/b_m)b_m`, `N_p = ceil(N/b_n)b_n`, `K_p = ceil(K/b_k)b_k`,

`q = (M_p/b_m)(N_p/b_n)(K_p/b_k)`, `rho = M_p N_p K_p/(MNK)`.

Retain directional padding ratios as well as rho, the count of output tiles `(M_p/b_m)(N_p/b_n)`, and sequential K panels `K_p/b_k`. Small output grids and long K sequences are different from many parallel output tiles even at equal MNK. M, N and K permutations are not assumed hardware-equivalent.

The current source applies one Strassen level **inside every tile-panel product**, not one global recursive split. Ignoring compiler transformations and using multiply-plus-add as two FLOPs, full cubic performs source-estimated dot work `F_C=2M_pN_pK_p`; Strassen performs `F_S=(7/4)M_pN_pK_p`. At equal aligned tile, its ideal dot-work reduction is 12.5%, equivalent to 8/7≈1.143× speedup only if all other costs and effective dot throughput match. This is a conditional arithmetic reference, not a universal observed-speedup ceiling: tiling, compiler schedules and baseline efficiency can differ.

The current full cubic performs one source accumulator update per output element per panel. The Strassen source has five A pre-additions, five B pre-additions, and twelve quadrant accumulator updates per panel. Source-level element counts are approximately

`V_C = q b_m b_n`,

`V_S = q [(5/4)b_k(b_m+b_n) + 3b_m b_n]`.

Thus at identical tile and padding the incremental source vector work relative to saved dot FLOPs is

`(V_S−V_C)/(F_C−F_S) = 5/b_m + 5/b_n + 8/b_k`.

This exposes a useful **tile-size** tradeoff. It is a work-count ratio, not a runtime ratio: vector and MXU throughput, fusion, liveness and overlap differ. For unequal selected tiles use each arm's own padded counts rather than this cancellation. A necessary condition for the padded Strassen dot estimate to be smaller than the padded cubic estimate is `rho_S/rho_C < 8/7`; meeting it does not establish a speedup. For example, doubling one otherwise equal padded extent produces a dot-work ratio `(7/8)×2=1.75`, outweighing the arithmetic saving before vector work.

At fixed tile size both F and V grow proportionally to MNK for aligned expanding shapes. Both present implementations therefore remain Θ(MNK), and increasing the global matrix alone does not asymptotically eliminate the repeated pre-add cost. Observed improvement at larger sizes can come from launch amortization, less relative padding, different selected tiles, parallel occupancy, reuse or compiler effects. Larger tiles reduce the source overhead ratio but can increase live storage, restrict occupancy or fail compilation. Monotonic global-size improvement is a hypothesis to test, not a mathematical consequence of using seven products.

A proposed interpretable model is `T ≈ L + Phi(F/P_eff, V/R_eff, Q/B_eff) + T_prepare`, where L represents fixed costs, Q estimated traffic and Phi describes schedule/overlap. Use additive and maximum-style models only as separately labeled approximations; their parameters cannot be uniquely identified from aggregate latency alone. Start with features `log2 M,N,K`, directional aspect ratios, padding ratios, tile dimensions, output-grid size and K-panel count. Fit on training observations and assess predictions on held-out shapes, retaining residuals. A fitted latency model is not evidence of measured bandwidth, MXU occupancy or vector/MXU overlap.

The original-operand byte count `2MK+2KN+4MN` for BF16 inputs and FP32 output is an allocation/I/O reference, not the measured traffic of a warm call. Runtime liveness may include original and padded buffers, references and multiple executables. Use actual memory preflight for admission, retain its assumptions and distinguish capacity failure from slowness. Do not extrapolate the previous warm-trace HBM lower bound: the existing N6 study explicitly found an observation below its proposed bound.

## Plot inventory

Every plot identifies cohort, seed scope, timing scope, hardware/runtime, denominator and whether values are measured, analytical, predicted or still planned. Publish static PNG/PDF figures plus machine-readable source tables; an interactive 3D companion can provide hover/filtering. Use common scales across comparable panels and colorblind-safe palettes. Before measurements, plot only the lattice, sampling plan and analytical resource/work estimates; never fill latency or winner maps with synthetic values.

| Figure family | Required views and interpretation |
|---|---|
| 3D study map | Plot the sampled points at `(log2 M,log2 N,log2 K)` against a faint full-lattice context. Color by design stratum before runs; after runs use eligibility/failure or measured winner. Distinguish reserved, explored, measured and excluded points. Tooltips contain exact dimensions, reason, tile and status. Do not connect missing cells into an observed surface. |
| Axis slices | Provide all 34 fixed-K MN slices, all 34 fixed-N MK slices and all 34 fixed-M NK slices in paginated small multiples or an indexed atlas. The sampled 48–64 shapes make most cells unmeasured; show those cells explicitly as blank/gray. Include occupancy counts. Do not interpolate a 34³ winner cube from a sparse sample. |
| Slice metrics | At each relevant slice offer Strassen/native, Strassen/cubic and cubic/native speedups; fastest eligible observed family; latency; uncertainty; and failure/padding. Use one global diverging `log2(reference/candidate)` color scale centered at zero. A compact report can show occupied slices while the atlas retains every slice. |
| Pairwise speedups | Sorted forest plots for all sampled shapes and all three pairwise comparisons, with paired confidence intervals, a 1× reference and a prespecified practical margin. Include regressions and same-route controls if measured. Use separate panels for complete and prepared timing. |
| Size and aspect ratios | Square-ladder latency/speedup curves, fixed-axis ladders, power-versus-midpoint pairs, and aspect-ratio/permutation panels. Join only prespecified comparable paths; use gaps for missing/failed measurements. Plot geometric volume and each axis separately to expose confounding. |
| Throughput | Plot useful classical work `2MNK/T` for all families, labeled effective classical-equivalent FLOP/s for Strassen. If plotting estimated executed dot work/T, use separate panels and its padded 7/8 factor. Do not compare these different numerators as the same hardware utilization. |
| Tile landscape | For each shape/family show latency versus complete tuple, candidate rank/ECDF, and 2D BM×BN heatmaps faceted by BK and variant. Retain unsuccessful attempted candidates as distinct symbols. Provide a machine-readable tuple table as the definitive near-optimal set. |
| Tile tolerance | Plot 1%, 3%, 5% near-optimal sets; cardinality; selected tuples on log2 axes; and their stability across seeds/confirmation blocks. Pairwise coordinate projections carry a warning that unseen Cartesian combinations are not established. |
| Mechanism diagnostics | Speedup versus padding ratio, output-grid size, sequential K panels and `5/BM+5/BN+8/BK`. Include equal-tile ablation and selected-tile comparisons as separate series. Overlay analytical thresholds only where their assumptions hold. |
| Numerical accuracy | Relative L2 and max error by shape/seed/algorithm, error versus speedup, and an accuracy–latency Pareto view. Show thresholds and sampled/full-reference coverage; failures remain plotted. Cancellation/dynamic-range diagnostics use separate symbols and captions. |
| Feasibility | Attempted/feasible/confirmed counts, compile/OOM/numerical/timeout categories on the same shape lattice, estimated memory versus limits, compiler temporary estimates and compile time. Distinguish unattempted from failed. No failure receives a fabricated latency or speedup. |
| Timing diagnostics | Raw sample distributions and time-order drift for every confirmed case, paired difference/ratio diagnostics, balanced arm order and variability versus mean latency. Separate within-run intervals from between-cohort variability. |
| Model assessment | Observed versus held-out predicted latency/speedup, residuals versus size/aspect/padding, predicted winner and calibration/error by stratum. Display measured markers over model surfaces. Report how far a prediction is from sampled shapes; do not imply data coverage beyond the sample. |
| Campaign accounting | Coverage by exponent/stratum/axis, screen/confirm/reserved membership, actual tuning budget per family, compile and measurement wall time, and completed/excluded counts. Keep immutable run/source identifiers in export metadata. |

For 3D winner plots, encode the interval/practical-margin decision separately from the lowest observed mean: a small noisy difference is unresolved. An optional point-size channel can show speedup magnitude or latency but must have an explicit legend. Avoid hidden points by allowing rotation, projections and filtering; the static slice atlas is the authoritative accessible alternative.

## Statistical interpretation and delivery checks

Define every speedup as the ratio of reference mean latency to candidate mean latency; greater than one favors the candidate. Use aligned rounds for paired bootstrap comparisons where pairing is real. Publish raw samples and all registered results. A different random matrix seed addresses input variability; another process/day/allocation addresses runtime variability; neither is created by resampling the same 30 rounds.

Prespecify the primary comparison family, practical margin and multiplicity policy. Individual 95% intervals can be useful exploratory graphics but cannot establish simultaneous 95% coverage across dozens of shapes, three comparisons, two scopes and many tiles. Report unadjusted effect-size intervals transparently; use a prespecified simultaneous or multiplicity-adjusted procedure for any confirmatory family-wide winner claims. Do not silently discard inconclusive points or advertise the most extreme selected examples as independent confirmation.

Select tiles from screening, confirm frozen choices with fresh rounds/seeds, and use reserved shapes only after fitting any dispatch/tile-range model. If a model is revised after seeing the reserved results, that set becomes development evidence and a new held-out evaluation is needed. State the bounded candidate family and sample domain with every generalization claim.

Before release, reconcile the plan manifest with every attempted shape/configuration, numerical gate, raw-round count and failure record. Verify conversion between `(M,K,N)` source order and `(M,N,K)` plot order, denominator orientation and unit consistency. Check plotted values against the source tables, inspect all exported pages for clipping/legibility, and retain versions/hashes of plot code and data. Report nothing as a measured frontier until timing exists.

## Existing evidence motivating this proposal

- [N5 confirmation](N5_CONFIRMATION_REVIEW_v001.md): Strassen complete-call comparisons against native gave six wins, four inconclusive results and six losses on 16 training shapes. The 8192 cube showed 1.1242× native/Strassen; three wide M=512 model-shaped products were near 0.90×. Thus larger M can matter, but volume alone has not established the mechanism.
- [N6 original profiles](N6_FINDINGS_v001.md): native had the shortest traced module for all three representatives, while some ordinary complete-call rankings favored custom kernels. It also documents why the warm-trace HBM estimate is not a validated lower bound.
- [N6 large-shape supplement](N6_SUPPLEMENT_FINDINGS_v001.md): the two deliberately favorable large examples had actual Strassen device-module wins, including 1.1291× versus native for the 8192 cube. These are post-hoc diagnostics, not a prevalence estimate.
- [N7 replication](N7_REPLICATION_FINDINGS_v002.md): three frozen custom routes retained gains on a fresh logical v5e allocation, while applying selected Strassen everywhere produced six wins, one inconclusive result and nine losses versus native. The same data seed was reused; this replicates a runtime cohort, not a new input distribution.
- [Kernel v001](../src/strassen_mm/kernels_v001.py) specifies tile-local Strassen, BF16 pre-add rounding, asymmetric alignment requirements and explicit prepared/complete scopes. [Kernel v002](../src/strassen_mm/kernels_v002.py) adds the full-cubic output-accumulator variant. These implementations justify the source-level counts above; they do not establish hardware instruction counts.

This document was prepared from existing source and findings. No new benchmark, profile, hardware allocation or timing was performed for it.
