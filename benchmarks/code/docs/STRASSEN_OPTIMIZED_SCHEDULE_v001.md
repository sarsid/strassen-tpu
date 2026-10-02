# Product-order hypotheses for strassen_optimized, version 001

This ledger records a source-level search, not a new performance experiment. It enumerates all 5,040 permutations of the seven standard Strassen products, freezes six distinct candidate schedules, and makes no new TPU timing or hardware-utilization claim. It does not change any frozen kernel, experiment configuration, or previous measurement. The existing order is already optimal under every proxy used here, so this analysis provides **no predicted speedup over the existing order**.

The implementation is [strassen_schedule_v001.py](../src/strassen_mm/strassen_schedule_v001.py). It imports only the Python standard library. Product IDs are one-based and match `p1` through `p7` in `kernels_v001.py`; quadrant indices are zero-based row-major. An immutable schedule ID encodes the complete order, such as `strassen_schedule_v001_p4652731`. Aliases are conveniences within this frozen version, not identifiers to reinterpret later.

## Source dependencies and invariant work

For A quadrants A0,A1,A2,A3 and B quadrants B0,B1,B2,B3, the products and signed accumulator updates are:

| Product | A operand | B operand | Output updates | BF16 input combinations |
|---|---|---|---|---:|
| p1 | A0+A3 | B0+B3 | C0 += p1; C3 += p1 | 2 |
| p2 | A2+A3 | B0 | C2 += p2; C3 -= p2 | 1 |
| p3 | A0 | B1−B3 | C1 += p3; C3 += p3 | 1 |
| p4 | A3 | B2−B0 | C0 += p4; C2 += p4 | 1 |
| p5 | A0+A1 | B3 | C0 -= p5; C1 += p5 | 1 |
| p6 | A2−A0 | B0+B1 | C3 += p6 | 2 |
| p7 | A1−A3 | B2+B3 | C0 += p7 | 2 |

Every order therefore contains seven dots, five A combinations, five B combinations, and twelve quadrant accumulator updates per tile-panel. Every two-term input expression retains the existing explicit BF16 rounding boundary. The order does not reduce the dot FLOPs or total source additions. For a tile (BM,BN,BK), the counts remain

\[
F_S=\frac74 B_MB_NB_K,\qquad
V_S=\frac54 B_K(B_M+B_N)+3B_MB_N.
\]

Relative to one full cubic dot and its full accumulator update, the extra vector work per saved dot FLOP remains

\[
\frac{V_S-V_C}{F_C-F_S}=\frac5{B_M}+\frac5{B_N}+\frac8{B_K}.
\]

These are counts of source element operations. They exclude initialization, copies, stores, compiler-generated instructions, and runtime scheduling. The ten explicit casts describe BF16 rounding semantics, not ten guaranteed standalone hardware instructions. A combination touches BM×BK/4 elements; a B combination touches BK×BN/4 elements. Consequently expression counts are not equally weighted time costs when BM differs from BN.

## What the schedule scores measure

Let Q(p) be the output quadrants updated by product p. The within-panel adjacent-overlap count is the sum of `|Q(p_i) ∩ Q(p_{i+1})|` over the six neighboring pairs. The cyclic count adds the last-to-first pair. Cyclic scoring matters because output accumulators persist across sequential K panels; for a one-panel multiplication, the within-panel score is the relevant alternative. Neither counts CPU launches, physical concurrent programs, or pipeline stalls.

For each quadrant, collect the product positions at which it is updated. The cyclic update gaps are differences between consecutive positions, including the wrap into the next panel. A gap of one means adjacent products update that quadrant. The inverse-gap penalty is the sum of 1/g over all twelve cyclic gaps, stored as an exact rational number. It favors separating repeated updates, without assigning a clock-cycle cost to them or modeling instruction ordering inside a product.

Two additional scores are the maximum number of pre-add/sub expressions in any cyclic window of two or three products. These describe source bursts, not temporary-buffer liveness or measured vector-unit contention. Products are consumed immediately in the source; compiler lowering may combine, schedule, or buffer them differently.

The hypothesis ranking is the lexicographic tuple `(cyclic_overlap, inverse_gap_penalty, two_product_preadd_burst, three_product_preadd_burst)`. It is a deterministic search recipe, **not a latency model**. There is no fitted coefficient, device timing, or speedup estimate in this module.

## Enumeration result and frozen registry

The existing order `(4,6,5,2,7,3,1)` reaches the global minimum of all four proxy components. Of all 5,040 orders, 112 achieve the minimum cyclic overlap of two, and 28 tie the complete proxy tuple `(2,37/6,3,5)`.

These minima also have simple combinatorial lower bounds. C0 and C3 each receive four updates in a seven-position cycle, so each must have at least one adjacent pair; two total overlaps is a lower bound. For each four-update quadrant, the most even positive integer gaps are (1,2,2,2), yielding inverse-gap sum 5/2. For each two-update quadrant they are (3,4), yielding 7/12. Together the bound is `2×5/2 + 2×7/12 = 37/6`. The ten input combinations imply maximum cyclic window sums of at least `ceil(20/7)=3` and `ceil(30/7)=5`. The existing order attains all these bounds simultaneously.

| Alias | Exact product order | Status and purpose |
|---|---|---|
| canonical | 1,2,3,4,5,6,7 | Existing source control; dependency score intentionally retained as a comparison |
| current | 4,6,5,2,7,3,1 | Existing measured interleaving; default registry choice, not a newly selected winner |
| proxy_lexicographic | 1,2,7,3,4,6,5 | Untimed alternative; lexicographically first complete-proxy minimizer |
| greedy_from_p1 | 1,2,5,6,4,3,7 | Untimed local greedy schedule; least overlap with previous and penultimate products, then expression count and product ID |
| reverse_current | 1,3,7,2,5,6,4 | Untimed reversal control; identical cyclic proxies, different source accumulation order |
| diverse_proxy_tie | 6,4,1,3,7,2,5 | Untimed complete-proxy minimizer maximizing its minimum Kendall distance from the first five registry orders, with lexicographic tie breaking |

Canonical has three within-panel overlaps plus one boundary overlap, inverse-gap penalty 106/15, and pre-add bursts four and six. Current has one within-panel overlap plus one boundary overlap, penalty 37/6, and bursts three and five. Greedy reaches the same overlap and inverse-gap minima but has a two-product pre-add burst of four. The diverse candidate is at least ten pair-order inversions from each preceding registry candidate. Reversals and cyclic rotations share cyclic scores, but can differ in startup behavior and FP32 accumulation order; they are not asserted to compile into distinct executables.

This negative result belongs in the paper's optimization history: a source search found multiple tied candidates but no additional reduction in these proxies beyond the existing interleaving. Any future improvement would require evidence about compiler scheduling or another mechanism that these scores do not resolve. A proxy reduction from canonical to current must not be translated into a percentage runtime improvement.

## Integration, validation, and evidence to retain

The runtime can obtain a tuple using:

```python
from strassen_mm.strassen_schedule_v001 import get_schedule

schedule = get_schedule("current")
order = schedule.order
identity = schedule.schedule_id
```

The core should apply each product exactly once and retain the existing input-combination BF16 casts, FP32 accumulation/output, and tile-panel scope. Floating-point accumulation is order-dependent. Exact symbolic algebra confirms the seven products implement classical block multiplication, but does not prove identical BF16/FP32 answers or accuracy for arbitrary data. Each compiled candidate still needs its own registered numerical gate before performance eligibility. No new holdout geometry or numerical-stress result is claimed here.

The focused test file verifies all 5,040 formal block-polynomial expansions, exhaustive proxy minima and tie counts, deterministic registry search recipes, exact asymmetric tile counts, repeated-panel boundary scoring, strict order validation, immutable IDs, and exclusive JSON export. Eleven standard-library tests passed locally. These are algebra/combinatorics checks, not device performance evidence.

The enumeration can be archived without a TPU or JAX. The output must be a new path; the tool refuses to overwrite it:

```sh
PYTHONPATH=src python -m strassen_mm.strassen_schedule_v001 \
  --all-orders --output NEW_schedule_source_scores.json

PYTHONPATH=src python -m unittest discover -s tests \
  -p test_strassen_schedule_v001.py -v
```

A later empirical comparison should preserve schedule ID, exact order, tile, compiler/source identity, numerical outcomes, compilation outcomes, executable identity when available, and both timing scopes. Compare at fixed tiles to isolate scheduling before allowing a jointly tuned tile comparison. Use fresh paired confirmation for any selected winner and account for the expanded candidate search. A failure or an indistinguishable executable is a result to record. Until those measurements exist, expected incremental benefit from new orders remains **unknown**, and the default remains the established current order.
