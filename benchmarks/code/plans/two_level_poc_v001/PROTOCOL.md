# One-level versus two-level Strassen: larger-shape proof of concept

Scope requested: implement two-level Strassen and test a few slightly larger
shapes against one-level Strassen before designing the next AlphaTensor study.
No Native, cubic or AlphaTensor arms are included in this experiment.

## Registered shapes and tiles

Four M x K x N shapes:

- Square: 12288 x 12288 x 12288 (previous proof of concept used 8192 square).
- Qwen gate/up: 16384 x 4096 x 24576.
- Mistral down: 16384 x 14336 x 4096.
- Gemma gate/up: 16384 x 3840 x 30720.

The rectangular cases double M relative to the AlphaTensor proof of concept;
model-derived K and N remain unchanged. All are synthetic matrix products.
Both algorithms receive both tiles (BM, BN, BK): (2048,2048,512) and
(2048,2048,1024). This is eight matched-tile pairs, 16 outcomes. It is a small
registered probe, not a full tuning search or independent confirmation of an
optimal tile. Report every pair; do not hide a losing tile or numerical failure.
The two-level leaves have K=128 or 256, respectively. Larger whole matrices
alone do not enlarge leaf products when the parent tile remains fixed.

## Implementation

Baseline is the preserved one-level Pallas `interleaved_output_accumulator`
kernel. The new Pallas kernel applies the same Strassen identities and product
order [4,6,5,2,7,3,1] at both levels. Each tile/panel has 49 quarter-size products
instead of seven half-size products. Relative to computing the seven outer
products classically, the second level reduces their multiplication work by
12.5%; it adds arithmetic, scratch traffic and scheduling work.

For each outer product, reuse one FP32 half-output-tile scratch buffer, compute
the seven inner products, then accumulate into the persistent FP32 output tile.
Intermediate products remain within the Pallas program. Both algorithms use
BF16 inputs and pre-adds, DEFAULT leaf-dot precision and FP32 accumulation and
output. The new kernel explicitly rounds input combinations to BF16 at both
levels, so increased numerical error is an expected possibility and is measured.
This initial scratch-based kernel does not establish the best two-level schedule.

## Execution and evidence

One new v5e allocation throughout. JAX 0.7.2, jaxlib 0.7.2 and libtpu 0.0.21.1;
same qualified Mosaic compatibility shim. Old/new machine timings are not pooled.
Before allocation, check the Pallas interpreter on exact integer matrices,
including nondivisible dimensions and multiple K panels, plus a Gaussian case.
These CPU checks establish correctness only, never TPU performance.

One Gaussian seed per shape (2026092131 + shape index), identical inputs for
both levels and both tile options. Five warmups and 30 synchronized timed rounds
per arm, rotating the two-arm order. Complete-call timing includes device
padding/crop and excludes compilation and host transfers. Record compilation,
compiler memory analysis, all raw timings, input fingerprints, source hashes,
machine identity, numerical error and failures. VMEM cap is 48 MiB; when known,
compiled argument+output+temporary memory must not exceed 8 GiB. Retain OOM and
compilation failures rather than silently altering a frozen arm.

Reference uses exact BF16 operands cast to host FP32, all K terms and 128 x 128
sampled output positions including edges. Check whole-output finiteness. Preserve
the existing gates: relative L2 <= 0.02 and max absolute error <=
0.001 + 0.05 * max absolute reference. Failed gates invalidate speedup claims;
passing gates do not establish equal error or downstream LLM quality.

Paired 95% bootstrap intervals are exploratory and conditional on this run.
All eight pairs are reported. No minimum across the two tiles is presented as
an unbiased confirmed improvement. Fresh independently tuned comparisons are
deferred until this probe identifies whether further work is useful.

Every execution uses a new archive and scoped commit. No executed source or
result is overwritten. Outer device-phase budget is 40 minutes. Release this
allocation after verified artifact retrieval; retain it only if remote state
is uncertain and recovery is required. Maintain the live progress journal.
