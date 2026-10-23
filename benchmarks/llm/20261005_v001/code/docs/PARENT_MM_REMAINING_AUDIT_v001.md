# Remaining parent MM differences after the full-contraction study

Read-only comparison on 2026-09-24. No kernel changes or TPU experiments.
`git ls-remote https://github.com/sarsid/strassen-tpu.git refs/heads/public-preview HEAD`
returned `95be1fb088656a89813b04492e1d77c66b36ccf9` for both refs. This matches
the previously archived parent audit. The cached GitHub landing page is older;
the commit-pinned source is the basis of this review.

The local pinned parent kernel SHA-256 is
`0d3f227e8eb899804c0295b8d3cbbb7c29fa96822cd17a64b5092dd2e8526b76`, verified
against `_parent_95be1fb/provenance.json`. Its Mosaic compatibility file also
matches that provenance. Parent code was inspected, not executed this turn.

## Highest-priority remaining standalone MM coverage

1. **Tune accumulator storage jointly with tiling and buffering.** Parent S1
   uses four FP32 quadrant accumulators for BF16 output, or the output tile as
   accumulator for ordinary FP32 output. Our latest `kernels_fullk_v002.kernel`
   retains seven outer products and reconstructs after K. The four-quadrant
   representation has lower persistent mathematical storage; the seven-product
   representation avoids repeated reconstruction across K panels. Compiler
   allocation and measured speed determine the actual tradeoff. Our older
   `kernels_v001._split_kernel` already implements four-quadrant and FP32
   output-accumulator variants, and the first full-K probe ran the pinned
   parent S1. Thus the mechanism is not wholly missing. The latest twelve-shape
   two-precision tuner does not offer it as an independently tuned S1 family.
   BF16 parent S1 was measured on the parent geometry in the earlier probe;
   those timings must not be pooled with the newer allocation.

2. **Widen dimension-divisor candidate coverage.** The parent's site tuner
   enumerates a predefined divisor menu for BN and BK alongside BM choices,
   prunes using a rough VMEM estimate, and actually compiles survivors. Our
   latest full-K study includes full K but uses a small output-tile menu and
   two shorter-panel choices. The replay recovered omitted winning layouts.
   An expanded joint search is justified; this is a coverage improvement,
   not proof that the parent has a globally optimal tuner. A VMEM estimate
   should remain a screening aid rather than proof of feasibility.

Sources: parent `strassen_pallas.py:230-286, 1044-1077`;
`experiments/qwen3/benchmark_qwen3_site_tile_tune.py:76-107`;
ours `src/strassen_mm/kernels_fullk_v002.py:18-32`,
`src/strassen_mm/benchmark_fullk_v002.py:15-63`,
`src/strassen_mm/kernels_v001.py:135-225`.

## Additional features, with narrower relevance

- **Transposed LHS without a full HBM transpose.** Parent
  `strassen_matmul_lhs_transposed` reads physical A[K,M] tiles and transposes
  each tile locally to compute A.T @ B. Our current standalone factories
  require the ordinary A[M,K], B[K,N] contract and do not expose that path.
  This is relevant to transpose-heavy workloads and weight gradients, not
  a speed improvement to already laid-out A @ B inputs. No new performance
  claim is established here. Source: parent `strassen_pallas.py:1224-1312`.

- **Mixed cubic/Strassen K panels.** The parent can run selected panels with
  eight ordinary products and others with seven Strassen products. Static
  panel sets or a prefetched scalar schedule are supported; the latter can
  reuse one executable across panel schedules. This can explore numerical
  error versus speed without a separate HBM intermediate. The parent calls
  these panels exact, but they still perform floating-point multiplication
  and accumulation. This is different from our boundary core/tail handling
  and S2's deferred outer reconstruction. It is not in the current MM tuner.
  Source: parent `strassen_pallas.py:575-595, 713-736, 1079-1130`.

- **Alternative rank-seven formulas and block permutations.** Parent exposes
  classical, dual and powers formulas plus M/K/N half-block permutations.
  Current ordinary S1/S2 search uses classical Strassen. These are research
  options, not established missing v6e speedups: the parent dispatcher selects
  classical and the initial-snapshot archive explicitly retains rejected
  numerical-stability investigations. Lower algebraic growth is not by itself
  a guarantee of smaller measured BF16 errors. Source: parent
  `strassen_pallas.py:381-573, 882-940`; `archive/initial-snapshot/README.md`.

- **Broader API precision and producer-fusion options.** Parent accepts FP32
  inputs as well as BF16 and exposes dot precision. Our current comparison
  fixes BF16 inputs/pre-adds; FP32 output is not FP32 input. Parent also
  forwards `allow_input_fusion` to the compiler. This is permission for
  eligible producers to fuse, not a guaranteed transformation or demonstrated
  gain for standalone resident input arrays. LLM epilogues/weight packing
  remain outside this review's requested scope.

## Already covered, and deployment distinction

Both implementations keep intermediate products local, use dependency-spaced
product ordering and Pallas input pipelining. Our prior v6e work tested one,
two and three input buffers and M/N traversal; the current full-K search uses
one/two buffers for full K and two for shorter panels. BF16 final stores and
full-K candidates are now present. Native has separate per-executable tuning,
including default and 48 MiB; custom kernel VMEM is independent. Parent
rejects nondivisible shapes and supplies no additional boundary-padding fix.

Parent `tuned_matmul` has a fixed v5e-era profile dispatcher for supported
squares and listed projections; it is not an adaptive v6e contraction tuner.
We also have historical v5e selector work (`selector_v001.py` and later
versions). The missing deployment integration discussed in the preceding
turn is specifically the latest v6e full/short-K, S1/S2, two-precision study.

The evidence supports prioritizing broader candidate coverage and joint
accumulator selection. Transpose handling and mixed-panel numerical control
are distinct extensions. No additional demonstrated universal standalone-MM
optimization was identified that explains a large remaining speed gap.

## Existing experiment evidence

- `runs/20260924-fullk-v001/operations/report/artifacts/RESULTS.md`: first
  fixed-configuration parent/current comparison, four FP32 geometries and
  one parent BF16 geometry. Some parent configurations were competitive;
  others were slower or infeasible. This does not establish a universal
  accumulator winner.
- `runs/20260924-fullk-precision-v001/operations/report/artifacts/RESULTS.md`:
  latest twelve-shape separate FP32/BF16 study.
- `runs/20260924-fullk-replay-v001/operations/report/artifacts/RESULTS.md`:
  omitted-layout replay with baselines rerun on the same allocation.

This note supplements `PARENT_KERNEL_AUDIT_v001.md`; it does not modify that
earlier audit or any executed source or result archive.
