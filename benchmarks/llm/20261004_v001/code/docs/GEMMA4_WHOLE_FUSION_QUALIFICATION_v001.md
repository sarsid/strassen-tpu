# Gemma4 whole-layer fusion qualification

The new `model_gemma4_fused_v001.py` passed three local CPU tests comprising
20 numerical comparisons against layers from the independent, pinned official
Transformers 5.5.0 model. This qualifies the tested synthetic configurations;
it is neither TPU performance evidence nor official-checkpoint qualification.
The active Qwen source bundle and Gemma4 queue gate remain unchanged.

Source commit: `7522c7a67`. Archived execution:
`runs/20261004T012914Z-gemma4-whole-fusion-qualification-v001-855375`.
All metrics are in `artifacts/gemma4-whole-fused-checks.json`. Execution completed
successfully in 26.6 seconds; archive commit: `467032877`.

## Contracts and design choices

The adapter constructs each layer from its actual geometry. Local attention
has six matrix products and global attention five: the latter computes one raw
K/V projection, then applies different K and V normalization. A fictitious
global V policy is rejected. Value RMS, global K normalization/rotation and the
persistent final layer scalar are explicit operations included in layer timing.
Q/K rotation, output/down RMS plus residual, and the gated GELU use the new
Gemma4-capable fused projection primitives. All model consumers store BF16.

Native uses the model's explicit rounding boundaries. Custom kernels can test
model or accumulator rounding with the same final BF16 store; this choice is
recorded and must pass the campaign's predictive gates. Prepared execution
excludes weight packing, while unprepared execution retains the matching layout
transformation. Metadata exposes actual sites, epilogues and operations outside
each fused projection so tuning and timing cannot silently omit them.

The fixture uses two dense synthetic layers, batch 2, sequence 9 and local
window 8, with 256/512 local/global head widths, nonunit learned RMS scales and
layer scalars. Both explicit and XLA attention are compared to official layers;
prepared/unprepared and fused/unfused Native outputs are checked for exactness.
Cubic, full-contraction cubic, S1 with output accumulators, and S2 with product
accumulators are exercised under both rounding contracts using Pallas CPU
interpretation and dense inputs. No device timing is inferred from these tests.

Native layer relative L2 error was at most 0.7271% for XLA attention and 0.0198%
for explicit attention, within the predeclared 2% local qualification limit.
Custom errors were at most 0.8419% with model rounding and 1.7046% with
accumulator rounding, below the existing 10% tuning eligibility limit. That
eligibility limit is not a relaxed predictive gate. Official real-checkpoint
NLL, KL, top-1 agreement and finite-output gates are still required.

## Remaining qualification

The complete streaming and cache path, per-attention-type tuner, modern remote
environment, official checkpoint oracle and launch/recovery bundle are not yet
qualified for Gemma4. Those steps must complete before changing its
`awaiting_qualification` queue state. No Gemma4 TPU has been allocated.
