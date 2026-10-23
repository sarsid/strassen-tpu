# Gemma4 native CPU qualification

October 3: the versioned native adapter passed the selected synthetic numerical
checks against the independently installed official Transformers 5.5.0 model.
This does not clear Gemma4's queue gate or establish real-checkpoint accuracy,
fused-kernel correctness, or TPU performance.

## Isolated official environment

The separate environment is `.runtime_private/gemma4-oracle-v001/`. Its package
requirements were committed before setup. The complete resolved package list,
environment manifest, and installed-source checks are archived in
`runs/20261003T234315Z-gemma4-oracle-environment-v001-73c27b/`.

Installed model, configuration and RoPE sources match the audited upstream
commit `c1c34249fa27deefbd4a377dfbf883a39baf5c6d` byte-for-byte. The stack is
Transformers 5.5.0, PyTorch 2.8.0, JAX/jaxlib 0.7.2 on CPU. Existing validation
environments and the remote Qwen runtime were not modified. No checkpoint
weights were acquired.

## Native adapter

`src/strassen_mm/model_gemma4_v001.py` implements fixed, unpadded text prefill
using native matrix multiplication, DEFAULT precision and FP32 accumulation.
It supports batched inputs, direct learned RMS scales, unscaled value RMS,
shared raw global K/V, local/global geometry, proportional RoPE, layer scalars,
scaled embeddings and softcapped logits. Explicit `reduce_precision` boundaries
preserve intended BF16 intermediate rounding under XLA.

There is no checkpoint reader, incremental cache, custom/fused kernel, campaign
streaming integration or TPU benchmark in this version.

## Independent validation

Passing execution:
`runs/20261003T235101Z-gemma4-native-qualification-v002-acb54f/`.
Source snapshot: `5baa5835c`; test entry: `tests/test_gemma4_native_v002.py`.
All three tests passed, with full metrics in the run's
`artifacts/gemma4-native-metrics.json`.

- Local/global rotary tables and transforms at positions 0, 1, 1023, 1024,
  1025 and 4095 matched exactly at the actual 256/512 head widths. The global
  unrotated coordinate ranges stayed unchanged. Learned and unscaled RMS
  primitives also matched exactly for the tested inputs.
- A complete two-layer model at batch 2, sequence 17, local window 8 matched
  exactly through both layers and final logits, with non-unit learned norm
  weights and layer scalars 0.625 and 1.375.
- A complete model at batch 1, sequence 1029, local window 1024 passed across
  the real window boundary. Maximum propagated layer relative L2 was 0.712%;
  final logit relative L2 was 0.805%. Next-token agreement was 98.833%, mean KL
  was 4.30e-7, and delta NLL was +3.73e-5.

The numerical gates were fixed before execution: layer relative L2 <= 2%,
logit relative L2 <= 3%, and the unchanged campaign predictive gates of
absolute delta NLL <= 0.01, mean KL <= 0.02, and top-1 agreement >= 97%.
These tiny randomly initialized models are semantic tests, not language-model
quality measurements on real pretrained weights.

The initial numerical attempt is retained in
`runs/20261003T234914Z-gemma4-native-qualification-v001-1551f9/`. The tiny-model
fixture's blanket `.to(bfloat16)` had also converted the official nonpersistent
RoPE frequency buffers to BF16, changing the positional encoding. Test v002
reconstructs those buffers through the official FP32 rotary constructor and
asserts their dtype. Adapter code, test inputs, learned weights and thresholds
were unchanged. A future real-checkpoint oracle loader must retain FP32 rotary
buffers and load persistent `layer_scalar` state explicitly.

## Remaining gate

Implement and validate the strict checkpoint reader and independent official
text loader; extend fusion, per-layer tuning geometry, streaming and cached
layouts in new versions; then qualify the actual official checkpoint and TPU
launch/recovery bundle. The ten Gemma4 workloads remain
`awaiting_qualification`. Qwen32 continues with its original frozen source.
