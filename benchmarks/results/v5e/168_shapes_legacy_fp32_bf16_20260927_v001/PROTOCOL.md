# Architecture-specific 168-shape studies

Authorized September 26, 2026 (UTC execution date September 27). This instruction
supersedes step 5 of RESEARCH_SEQUENCE_v001: run v6e first with the completed
joint-tuner kernels; then use the historical v5e code and tuning menu, adding
only BF16 output. Do not port the v6e tuner or memory limits to v5e.

## Architecture guardrails

The recorded diagnostic and the official
[JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html),
[v6e specifications](https://docs.cloud.google.com/tpu/docs/v6e), and
[v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e) explain why a
tile winning on one architecture is not automatically appropriate on the other.
v6e has larger matrix arrays and greater compute throughput relative to memory
bandwidth. Memory capacity, VMEM scheduling, leaf size, reconstruction and
pipeline overlap must all be considered. Specification differences alone do
not prove the cause of a measured performance difference.

| Contract | v6e | v5e |
|---|---|---|
| Kernel lineage | Completed joint pilot; unchanged kernel bodies | Historical September 21 MAIN kernels, unchanged |
| Custom VMEM allowance | 112 MiB | 48 MiB |
| Native scoped VMEM search | default, 32, 48, 64, 96, 112 MiB | Historical default, 32, 48, 64 MiB |
| Custom search | Joint tile, K panel, accumulator strategy, buffers | Historical 16 candidates per custom family |
| Output | FP32 and BF16, separate | Historical FP32 complete call plus one final BF16 conversion |
| Software | Pilot JAX/jaxlib 0.11.2, libtpu 0.0.48 | Historical JAX/jaxlib 0.7.2, libtpu 0.0.21.1 |

The v5e source files were compared byte for byte with the frozen historical
cohort before preparation. Preserve that comparison in validation evidence.
The two implementations and compiler versions differ intentionally. Report
architecture-specific results; do not call the difference a hardware-only
causal effect. A separately authorized matched-code experiment could isolate
that later.

## Shared measurement scope

Preserve all 168 original M,K,N tuples and IDs. Measure default Native, tuned
Native, independently tuned cubic, S1 and S2, separately for FP32 and BF16
output. Inputs/pre-adds remain BF16; accumulation/reconstruction remain FP32.
Final output conversion, ordinary required padding and crop are inside timing
and error. No LLMs or recursion depths 3/4 in these runs.

v6e keeps the full aligned pilot search. Add the existing tile-multiple zero
pad/crop contract around its unchanged kernels for boundary shapes. This is
not a revival of the discarded padding families, native-tail decomposition,
or a padding-specific search. Record padded volume per candidate. For M or N
below 512, add small geometries and retain the three with least padded output
area, plus protected known controls. This dimension-only reduction is frozen
before timing. K panels include rounded-up full K, 512, selected aligned
divisors and historical controls. A memory heuristic above 160 MiB prunes
against the 112 MiB compiler limit; known pilot controls bypass that heuristic.

Independently tune both blocked cubic and full-tile output-accumulator cubic.
The latter retains its existing compiler-managed buffering; do not label that
as an explicitly tuned buffer count. All original aligned pilot candidates
must remain offered. Record the complete registry and reasons for exclusions.

Screen in randomized groups of at most 12 custom candidates with a repeated
Native anchor (three warmups, seven rounds). Freeze family and overall choices
using same-batch Native-normalized latency; confirm on three fresh seeds with
five warmups and 30 paired rounds. Record default Native and matched cubic,
alternative accumulator and buffer controls. Never select again from the
confirmation measurements. Cubic is eligible to win the overall selection.

Use the established FP64 all-K reference: full when affordable, otherwise
seeded 128x128 output samples including edges, plus full-output finiteness.
Store relative L2, maximum absolute error, RMSE, MAE, p50/p99 absolute error,
normwise error, reference scope and sample indices. Gates remain finite,
relative L2 <= 0.02, and max absolute error <= 0.001 + 0.05*max|reference|.
Recommend the frozen custom winner only when all confirmation error gates pass,
mean speedup over tuned Native >= 1.01 and the paired 95% lower bound exceeds 1;
otherwise use tuned Native. No arbitrary-input accuracy guarantee is implied.

## Qualification, progress and archives

Commit source before validation. Test exact integer arithmetic at aligned,
tiny and boundary dimensions, both output contracts, both Strassen depths,
both accumulator strategies and buffers. Check final-only BF16 rounding for
the v5e adapter and preservation of every aligned pilot candidate. Hardware
smoke must pass before the 168-shape sweep.

Execute a tiny, boundary and large shape first to assess actual compilation
and end-to-end tuning cost; keep their original manifest indices and seeds.
The full inherited search has tens of thousands of candidate attempts. It is
substantially larger than the old 168-shape screen; report the candidate count
and observed throughput instead of assuming the old runtime estimate applies.

Every shape receives its own immutable screen and confirmation phases. After
each phase, verify the original artifact manifest, export a compressed bundle
to the dedicated results folder, read it back and compare every artifact, then
make a scoped commit. Preserve the common frozen source once per allocation.
The live dashboard shows actual completed cases, failures, selected timings
and evidence links. Preparation and compilation are not completed measurements.

Dedicated destinations:

- `results/v6e/168_shapes_joint_fp32_bf16_20260927_v001/`
- `results/v5e/168_shapes_legacy_fp32_bf16_20260927_v001/`
- Historical reference stays at `results/v5e/168_shapes_fp32_20260921_v001/`.

One allocation and one active timing worker at a time. Archive and release
before the v5e stage. Never silently switch hardware, mix timing samples from
different allocations, retry an uncertain launch or overwrite failed attempts.
The Mac runs the controller: keep it plugged in, awake, online and lid open.
Its idle-sleep guard does not prevent lid-close sleep. Verify runtime release.
