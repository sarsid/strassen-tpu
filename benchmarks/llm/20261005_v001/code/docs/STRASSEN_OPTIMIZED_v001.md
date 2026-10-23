# strassen_optimized: implementation and optimization record v001

`strassen_optimized` is a new experimental program alongside the frozen kernels.
It implements standard one-level Strassen inside each tile-panel, BF16 operands
and pre-additions, FP32 accumulation and output, and DEFAULT dot precision.
Public dimensions are **M,N,K**; tiles are **BM,BN,BK**. The historical kernel
factory uses M,K,N and is not changed.

The name describes the optimization work, not a verified runtime improvement.
This first implementation is correctness-tested in JAX 0.7.2's CPU Pallas
interpreter, and local TPU-targeted export/lowering checks pass. Device-side TPU
compilation, performance and memory qualification require the
separate ablation runner. No CPU latency is reported as TPU performance.

## Implemented changes and their controls

| ID | Change | Intended benefit | Control and important limit |
|---|---|---|---|
| SOPT-BASE | Independent implementation of the existing one-level algebra and product order | Preserve a comparable starting point | New `baseline` versus frozen `kernels_v002` Strassen; require numerical parity and measure compiler differences |
| SOPT-001 | Implicit partial blocks plus explicit BF16 zero masks inside the kernel | Avoid explicit full-array padding and output cropping | `masked_edges` versus `baseline`; same full-tile dot work, additional masks can cost time |
| SOPT-002 | Four functional FP32 quadrant accumulators, one write per quadrant per panel; seed first contributions on panel zero | Reduce mutable accumulator operations and first-panel initialization/addition work | `local_accumulators` versus `baseline`; 4 versus 12 source quadrant writes, not an asserted HBM-traffic reduction; greater live state/branching may regress |
| SOPT-003 | Enumerate all 5,040 product orders; expose six reproducible representative schedules | Test compiler sensitivity to dependency spacing and pre-add bursts | Current order remains default; it already ties for best under the declared proxy; no schedule is a measured winner |
| SOPT-004 | Exchange M and N traversal while keeping K consecutive | Test operand reuse/traversal on asymmetric shapes | `optimized_nmk` versus `optimized`; source-level consecutive reuse argument is strongest when there is one K panel |
| SOPT-005 | Strassen on a complete-tile core, native XLA on exact M/N/K tails | Avoid mostly empty padded dot products | `peeled_edges` is explicitly a hybrid, adds calls/slices/assembly, changes error profile; no complete core means a recorded native-only fallback |

The structured history lives in `reports/strassen_optimized_ledger_v001.json`.
Every candidate includes optimization IDs, source version, exact tile, order,
traversal, boundary policy, precision and work/extent proxies in its metadata.
Keep this version when trying a later change; create a new source/log version
and preserve unsuccessful experiments too.

## Variants

- `baseline`: original global zero padding, direct FP32 output-reference updates.
- `masked_edges`: SOPT-001 only.
- `local_accumulators`: SOPT-002 only, with the original padding path.
- `optimized` (default): SOPT-001 and SOPT-002 together.
- `peeled_edges`: SOPT-002 in the full-tile core, plus SOPT-005 native tails.

All use the existing product order `(4,6,5,2,7,3,1)` and M,N,K traversal by
default. Both are explicit parameters, rather than hidden shape-based rules.
The existing M,N,K cost selector is not used: it was calibrated on other kernels
and its coefficients are invalid for these new implementations.

## Boundary handling and numerical semantics

`BlockSpec` gives the body a full-sized block even at an edge, with unspecified
out-of-bounds input values and discarded out-of-bounds output positions. The
new path reads block values and uses `where(valid, value, BF16(0))` **before any
Strassen combination**. All three logical dimensions are masked. Masking only
K would be incorrect because Strassen mixes other quadrants; multiplying garbage
by zero is also incorrect for NaN-filled padding. Fully divisible dimensions
eliminate the corresponding mask from the traced source.

This removes explicit padded external arrays and cropping. It does **not** remove
full-tile dot arithmetic: the old and masked paths have identical logical padded
volume. Metadata separates actual prepared shapes from arithmetic tile coverage.

The local-accumulator first branch never reads output memory. It seeds each of
four accumulators with its first signed product, then preserves that quadrant's
subsequent update sequence. Following panels read the four existing quadrants.
A `lax.cond` guards those reads; value selection is not a valid substitute.
The first panel saves BM*BN source accumulator element additions per output
tile. Later panels retain the same arithmetic count. Signed-zero behavior may
differ. Changing product order or using native tails changes floating-point
ordering more generally, so each candidate receives its own numerical checks.
BF16 cancellation risk remains; a shape-only rule cannot establish accuracy
for arbitrary matrix values.

For the peeled hybrid, let Mc,Nc,Kc be floor multiples of the tile dimensions.
Compute C00 with Strassen over Kc, then add its native K remainder. Compute the
right strip natively over all K and the entire bottom strip natively over all K.
These regions are disjoint and cover the original matrix. Its source dot-work
proxy is

`(7/4)*Mc*Nc*Kc + 2*(M*N*K - Mc*Nc*Kc)`.

The native compiler owns the tails; source proxies do not expose its scheduling.
The complete call includes slicing, all products, tail addition and concatenation.

## Existing pipeline and investigated alternatives

The old Pallas grid already uses compiler-managed HBM/VMEM pipelining. Adding a
manual double buffer would not, by itself, introduce missing pipelining. The
pinned backend has restrictions that differ from current examples: generic
masked loads are unsupported on TPU; ordinary `pallas_call` supports one or two
buffers and rejects lookahead. See the separately reviewed backend notes.

A custom nested/asynchronous DMA pipeline is **deferred**, not implemented or
counted as a speedup. It would need evidence that changing transfer scheduling
beats the existing compiler pipeline. Deeper Strassen recursion and model
epilogue changes are also outside this first implementation. Keeping these
changes separate makes the ablations interpretable.

## Expected speed, before TPU timing

No numeric speedup forecast is claimed. On already aligned, tuned large squares,
both old and new versions perform the same seven dots per tile-panel. Accumulator
changes may help, compile to much the same instructions, or hurt due to live
state. Broad 2x gains from this change are not a reasonable expectation.

Padding-heavy shapes have a more plausible opportunity: masked edges can avoid
preparation/extents, and peeled edges can also reduce dot work. Multiple native
calls and assembly can consume those savings. The existing product order already
scores well under the source proxy. The likely useful result is shape-dependent
improvement with an explicit fallback, if confirmed, rather than unconditional
replacement of the prior Strassen or native XLA implementations.

## Usage

From the repository, describe the exact candidate without running a kernel:

```sh
PYTHONPATH=src python strassen_optimized.py \
  --m 4097 --n 4097 --k 4097 --tile 2048 2048 512 --describe-only
```

Run a small CPU algebra check (no timings):

```sh
PYTHONPATH=src python strassen_optimized.py \
  --m 17 --n 257 --k 259 --tile 16 256 256 --interpret-correctness
```

Use from Python:

```python
import jax
from strassen_mm.kernels_v001 import enable_qualified_mosaic_v7_compat
from strassen_mm.strassen_optimized import make_matmul
compatibility = enable_qualified_mosaic_v7_compat()  # Record with run metadata.
fn = make_matmul((M, N, K), (BM, BN, BK), variant="optimized")
C = jax.jit(fn)(A_bf16, B_bf16)
```

The CLI and ablation runner invoke and record the existing qualified Mosaic
compatibility helper. Python callers opt in explicitly as above. It only applies
the previously qualified serialization adjustment for JAX 0.7.2 with libtpu
0.0.21/0.0.21.1; it does not imply qualification of these new kernels.

Factory construction does not silently cast inputs, search tiles, or benchmark.
The default CLI tile is only an example. Use measured tiles or run the explicit
ablation protocol; no universal optimal tile is implied.

The benchmark runner compares native-default, native64m, frozen Strassen, the
new baseline, individual changes, their combination, alternate traversal and
the labeled hybrid. It supports optional registered schedule candidates. It
writes each run to a new exclusive directory, with raw synchronized timing
samples, numerical eligibility, metadata and direct paired comparisons.
`--interpret-correctness` uses small inputs and records no performance results.

Sources: `src/strassen_mm/strassen_optimized.py`, `strassen_optimized.py`,
`src/strassen_mm/strassen_schedule_v001.py`, and
`tools/benchmark_strassen_optimized_v001.py`. Background/API details are in
`docs/STRASSEN_OPTIMIZED_BACKEND_NOTES_v001.md` and schedule reasoning in
`docs/STRASSEN_OPTIMIZED_SCHEDULE_v001.md`.
