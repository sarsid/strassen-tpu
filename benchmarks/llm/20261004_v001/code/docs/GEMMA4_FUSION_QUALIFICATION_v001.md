# Gemma4 fusion primitives: local qualification

October 4 preparation. `fusion_v004.py` and `kernels_fused_v005.py` add the
explicit `norm='gemma4'` convention: FP32 normalization followed directly by
the learned scale, without Gemma3's additive offset or Qwen's intermediate
rounding boundary. Existing model/accumulator rounding contracts and BF16
storage boundaries remain explicit. The v6e kernel retains DEFAULT dot
precision, FP32 accumulation, S1/S2 arithmetic, input buffering and tile controls.

These are new source versions. The running Qwen32 session continues using its
original frozen sources and does not import these modules.

Passing local execution:
`runs/20261004T004023Z-gemma4-fusion-qualification-v001-511627/`.
Committed source: `7b229ca56`. Three tests passed, including 27 numerical
comparisons with zero observed error against the pinned official Gemma4
normalization and rotary operations:

- Six reference comparisons cover full-row RMS plus residual and Q/K RMS plus
  local/global RoPE at head widths 256/512, each in model and accumulator
  rounding modes.
- Twenty-one CPU Pallas interpretation comparisons cover the same three
  operations in model rounding mode across Native, blocked/full cubic, and
  S1/S2 with both product and output accumulator schedules. Inputs require row,
  K-panel and column padding, including complete-head layout restoration.
- A zero learned-scale check distinguishes Gemma4's direct scale from Gemma3's
  offset scale.

The matrices use sparse integers so recursive BF16 pre-additions are exact.
That isolates layout, reduction and epilogue correctness. It does not establish
accuracy on dense activations or real weights, validate custom accumulator-mode
device behavior, or measure TPU performance. The original tolerance of
`atol=rtol=0.002` was unchanged; the observed outputs matched exactly.

Next, integrate these projections into a versioned Gemma4 whole-layer adapter.
Keep the global shared raw K projection as one matrix multiplication, followed
by separate K and V postprocessing. Local V needs unscaled RMS; initially it
can follow the V projection outside the custom kernel with that scope recorded
explicitly. Then qualify dense whole layers, both attention types, caching,
per-layer tuning geometry and actual runtime bundles before the official
checkpoint/TPU gate can be cleared. No Gemma4 workload has been marked measured
or qualified for deployment by this local test.
