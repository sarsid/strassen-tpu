# Parent fusion integration — October 2, 2026

The inspected parent is `sarsid/strassen-tpu`, `public-preview`, commit
`95be1fb088656a89813b04492e1d77c66b36ccf9`. Source was fetched and inspected in
the publication worktree. Historical 168-shape code/data remain unchanged.

## What the parent does

The [projection kernel](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py)
keeps Strassen products/outputs in VMEM across the contraction. On the final K
panel it can apply SwiGLU/GeGLU, add a residual or bias, normalize Q/K and apply
RoPE, or apply Gemma postprojection RMSNorm before a residual. It writes the
final output rather than materializing those intermediate projection tensors.

Gate/up weights are rearranged offline so corresponding halves occupy the
same output tile. Q/K weights similarly pair the two halves of each head.
Q/K normalization uses matrix-unit reductions with 0/1 segmentation matrices;
head order is restored outside the kernel. Gemma postprojection normalization
requires the full hidden width in one column tile. Its optional S1 gated early
finalization changes the final panel's product order to finish the top output
half before the bottom half. That is a distinct numerical/scheduling candidate.

The [Qwen experiment scripts](https://github.com/sarsid/strassen-tpu/tree/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3)
route selected projection sites inside layer JITs. Their hardware profiles,
import-time compiler settings, and older JAX/libtpu versions are not portable
performance guarantees. We do not import those modules or adopt their tiles.

## Integration decisions

* `fusion_v003.py` defines explicit operation, normalization, rounding, layout,
  output-dtype and auxiliary-operand contracts.
* `kernels_fused_v003.py` exposes the qualified v001 Pallas implementation, which applies epilogues after the final K panel. It reuses
  our S1/S2 pre-add, reconstruction and accumulator strategies. Both blocked
  cubic and full-tile cubic receive the same fusion options. Native remains
  ordinary XLA code so whole-layer compilation can optimize its baseline.
* `model_fused_v003.py` consumes existing checkpoint layer dictionaries for
  Qwen3, Mistral and Gemma3 text. The six sites are Q, K, V, O, gate/up and down.
  It supports fixed unpadded prefill batches, with separate attention for each
  sequence. Gemma local/global RoPE and normalization order remain explicit.
  Incremental KV-cache decoding, quantization and image inputs are not covered.
* `tuner_fused_v002.py` reuses the architecture-study search menu, rechecks
  memory/layout constraints, and logs rejection reasons. It adds full-row
  candidates for Gemma normalization and can add early S1 candidates. It does
  not pick a fused winner from old pure-MM timings.

Default `rounding='model'` preserves projection-to-BF16, activation-to-BF16,
Qwen normalization and RoPE multiplication boundaries before the final output
cast. Parent-style `rounding='accumulator'` computes the epilogue from FP32
accumulators and is a separately named accuracy experiment. Gate/up padding is
applied to each half independently. Padded full-row RMSNorm divides by the
original width, not the padded width. Q/K squared-value reductions request
HIGHEST precision rather than potentially rounding the FP32 squares to BF16.

FP32 and BF16 final projection outputs remain separate groups. The layer
adapter casts projection outputs to the model's BF16 consumer contract; those
casts are timed. An FP32 projection-output group is not an FP32 language model.
Packing depends on tile and epilogue and must be repeated after either changes.
`prepared` excludes offline weight packing but includes activation/auxiliary
padding, output crops, Q/K layout restoration and consumer casts. `complete`
also includes packing. Keep packed weights resident and release redundant
unpacked device copies when measuring a resident model.

Fused output-accumulator kernels need FP32 scratch even with FP32 output,
because the final nonlinear output cannot serve as an accumulation buffer.
Full-tile cubic now exposes one/two input buffers, unlike the historical
compiler-managed variant. Both changes require fresh tuning. Memory estimates
are heuristic pruning decisions, not proof of device feasibility.

Per-projection Native compiler choices cannot all apply inside one whole-layer
JIT. The adapter rejects a selected arm's nonempty compiler options unless they
match the explicitly supplied whole-layer options. Benchmark each whole-layer
compiler setting consistently across algorithms; do not silently inherit the
parent's global VMEM flags. Current target remains the study's JAX/jaxlib 0.11.2,
libtpu 0.0.48, v6e kernel VMEM allowance 112 MiB. Local CPU interpret checks use
JAX 0.7.2 and do not qualify TPU compilation or speed.

## Usage and validation boundary

```python
from strassen_mm.model_fused_v003 import build_layer, native_policy

# Replace selected entries with freshly tuned v6e study-format arms.
policy = native_policy('bfloat16')
layer = build_layer(checkpoint.config, 2048, policy, output_dtype='bfloat16')
packed = layer.prepare_weights(checkpoint.layer(0))
y = layer.prepared(hidden, packed)
y.block_until_ready()
```

Pass `fused_sites=()` for an unfused control under the same mathematical
epilogue contract. Use `rounding='accumulator'` only as a separate experiment.
Use `layer_index` for Gemma's correct local/global layer, and `batch_size` for
the exact prefill profile. Prepared weights belong to that layer's selected
tile/layout; shape checks cannot detect a wrong same-shape packed layout.

Validation coverage: independent sparse-integer oracles across padding/multiple
K panels, Gaussian replay of preserved S1/S2 arithmetic, both final output
dtypes, all epilogues, model rounding versus accumulator rounding, Gemma and
Qwen normalization, full-layer comparison to preserved model primitives,
batched attention and tuner rejection traces. Archived outcomes, not this plan,
determine qualification. No LLM quality or end-to-end speed claim is implied.

After correctness and device compilation: tune actual fused operations on
real checkpoint activations for each model and fixed (batch, sequence) profile,
then independently confirm whole-layer/model latency and prediction quality.
Record raw timings, compilation failures, tile/pipeline/contraction choices,
projection/hidden/logit errors, KL, top-1 agreement, NLL and perplexity. A general
shape selector is unnecessary for these fixed profiles.

## Qualification outcomes and rounding correction

All **54 v6e projection checks passed** (27 FP32-output and 27 BF16-output)
against an independent host NumPy oracle. The largest absolute discrepancies
were 3.815e-6 and 4.024e-7 respectively. These use sparse integer matrices so
Strassen pre-adds are exact; they establish the fused operation's semantics,
not Strassen's error on real weights. Device identity and pinned package
versions were checked, and the owned TPU was released with absence verified.

The CPU integration run completed 96 checks before its final planner-coverage
assertion caught a missing full-row full-cubic candidate for Gemma. That run
remains marked failed. The corrected planner passed four additional scoped
checks, including all four algorithm families and invalid-candidate rejection.
All preceding arithmetic/model checks passed. Gaussian fused versus unfused
cubic layers were bitwise equal for Qwen, Mistral and Gemma in the final CPU
checks. This is constructed-weight testing, not checkpoint-quality evidence.

Two earlier implementation attempts and their failed checks are preserved.
The first hardware pass compiled every kernel but exposed an invalid XLA
reference: converting FP32 to BF16 and back can be optimized away. Merely
expressing BF16 operations did not enforce every boundary in a whole layer.
The final Native/XLA path uses `lax.reduce_precision(..., 8, 7)` at the intended
rounding points, including attention/normalization consumer boundaries. Pallas
retains its explicitly rounded epilogues, verified against the host oracle.
There are no global compiler mutations. The final `model` contract is explicit
rounding, and `accumulator` remains a separately identified alternative.

This deliberately changes the accidental rounding behavior of older compiled
model adapters. In the small CPU fixtures the final strict Native layer differs
from the preserved compiled composition by approximately 0.41–0.54% relative
L2. Do not attribute that difference to Strassen. Before a real-model claim,
requalify the new Native adapter against the official model, and compare all
algorithms under the same rounding contract. The historical Gemma qualification
must not be silently treated as resolved by these small constructed tests.

Evidence runs:

* `runs/20261002T172533Z-fused-integration-cpu-v003-98cd73`
* `runs/20261002T172753Z-fused-registry-cpu-v001-ac378b`
* `runs/20261002T172539Z-fused-device-qualification-v002-ce2815`

The immutable qualification summary records the source hashes, model checks,
per-dtype hardware error extrema, and runtime release receipt. No new real-LLM
speedups, perplexities or accuracy measurements were produced in this task.
