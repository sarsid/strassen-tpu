# Gemma4 checkpoint loading validation

October 4: the strict memory-mapped text checkpoint reader and the independent
official PyTorch loader passed local synthetic checkpoint checks. No official
Gemma4 weights were downloaded, no TPU was allocated, and the campaign's Gemma4
qualification gate remains closed.

Files to use for the next integration step:

- `src/strassen_mm/checkpoint_gemma4_v001.py`: verifies manifest revision,
  file sizes and hashes, configuration identity, tensor payload bounds, required
  text state, per-layer geometry and BF16 parameter dtypes before mapping.
- `tools/load_gemma4_text_v002.py`: independently loads the caller-verified
  local checkpoint with safetensors and the official Transformers 5.5.0 text
  model, without using our tensor parser or shape helper.
- `tests/test_gemma4_checkpoint_v002.py`: synthetic checkpoint round-trip and
  malformed-state checks against the original official tiny model.

Both loaders include persistent layer scalars and preserve their stored BF16 or
FP32 dtype. Global attention has no separate V projection. The optional tied
output-head alias must equal the embeddings; the official model retains shared
parameter storage. Nonpersistent official rotary frequency buffers remain FP32.
Unused multimodal weights are not loaded into the text model.

Passing execution:
`runs/20261004T001650Z-gemma4-checkpoint-qualification-v002-1fd74c/`.
Source snapshot: `39fa543c0`. Four tests passed:

1. Exact round-trip of every text parameter and persistent scalar, read-only
   mapped tensors, preserved tied storage, and unchanged official model logits.
   The native JAX model also passed the declared relative logit L2 <= 3% check
   using the loaded weights, including a non-BF16-representable FP32 scalar.
2. Standalone text prefixes and omitted tied-head aliases load correctly.
3. Both loaders reject missing scalars, an unexpected global V tensor, wrong
   shapes or parameter dtypes, inconsistent tied-head values and duplicate
   canonical text tensors.
4. The strict reader rejects incorrect file hashes, inconsistent configuration
   metadata and escaping file paths.

The first attempt failed at the official loader's initialization import because
Transformers 5.5.0 moved `no_init_weights` from `modeling_utils` to
`initialization`. It remains archived in
`runs/20261004T001510Z-gemma4-checkpoint-qualification-v001-8134b0/`.
Loader/test v002 corrects only that API reference; the reader and acceptance
requirements are unchanged.

Remaining work: versioned fused kernels, global shared-K/V postprocessing,
per-layer tuning, streaming and cache integration, followed by official
checkpoint numerical qualification and actual TPU launch/recovery validation.
These synthetic loading tests are not pretrained-model quality or performance
results. Existing Qwen sources and measurements remain unchanged.
