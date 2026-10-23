# Gemma4-31B adapter preparation

Status: architecture audit only. No Gemma4 checkpoint weights have been acquired
for this preparation, no new adapter has been executed, and no qualification gate
has been cleared. Qwen32 continues with its original frozen source bundle.

The checkpoint remains `google/gemma-4-31B` at revision
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`. Its saved configuration is in
`results/v6e/llm_campaign_20261003_v001/inputs_gemma4_v001/`.

The independent implementation reference is Transformers **5.5.0**, resolved to
commit `c1c34249fa27deefbd4a377dfbf883a39baf5c6d`. The Apache-2.0 source files,
license, retrieval record and SHA256 digests are preserved in
`third_party/transformers_gemma4_v001/`. This is a proposed oracle pin, not a
claim that a compatible environment or checkpoint qualification has passed.

Primary references:

- [Official attention, normalization, decoder and model implementation](https://github.com/huggingface/transformers/blob/c1c34249fa27deefbd4a377dfbf883a39baf5c6d/src/transformers/models/gemma4/modeling_gemma4.py)
- [Official proportional RoPE implementation](https://github.com/huggingface/transformers/blob/c1c34249fa27deefbd4a377dfbf883a39baf5c6d/src/transformers/modeling_rope_utils.py)
- [Official configuration implementation](https://github.com/huggingface/transformers/blob/c1c34249fa27deefbd4a377dfbf883a39baf5c6d/src/transformers/models/gemma4/configuration_gemma4.py)

## Correctness differences that prevent reuse of the Gemma3 adapter

| Operation | Required Gemma4 behavior | Existing adapter mismatch |
|---|---|---|
| RMS normalization | Normalize in FP32, multiply directly by the learned scale in FP32, then cast back | Gemma3 uses an offset scale; Qwen's model-rounding path also inserts a different rounding boundary |
| Attention scaling | Explicit scale of 1.0 | Gemma3 uses its configured inverse square-root attention scalar |
| Value states | RMS-normalize each value head without a learned scale | Existing value projection has no such normalization |
| Global K/V | One raw K projection supplies both branches; K then gets learned normalization and RoPE, V gets unscaled normalization | Existing layer and tuner assume separate K and V matrix products |
| Head geometry | Local: head width 256, 16 KV heads. Global: head width 512, 4 KV heads. Both use 32 query heads | Current geometry helper uses a single head width and KV-head count |
| Final layer output | Apply the decoder's persistent `layer_scalar` after the final residual addition | Existing checkpoint contract has no scalar, and the old oracle loader loads named parameters only |
| Position encoding | Local default RoPE; global proportional RoPE with zero frequencies for unrotated pairs | Existing Gemma3 global RoPE supports full rotation and optional linear scaling |

The final layer scalar must be inspected in the official checkpoint and loaded
with the other persistent model state. Do not assume a unit value merely because
the class initializes it to one. An oracle loader must validate persistent
buffers as well as parameters and explicitly account for tied weight aliases.

Gemma4 uses the familiar sequence of attention, post-attention normalization,
residual addition, pre-MLP normalization, gated GELU MLP, post-MLP normalization
and residual addition, followed by its layer scalar. The checkpoint also uses
scaled token embeddings, a tied output head and a final logit softcap of 30.
Their dtype boundaries must be checked against the pinned oracle.

## Projection geometry

Here `M = batch * sequence`, hidden width is 5376, and MLP width is 21504.

| Site | Sliding attention: (M,K,N) | Global attention: (M,K,N) |
|---|---|---|
| Q | (M,5376,8192) | (M,5376,16384) |
| K | (M,5376,4096) | (M,5376,2048), shared raw source for V |
| V | (M,5376,4096) | No separate matrix product |
| Output | (M,8192,5376) | (M,16384,5376) |
| Packed gate/up | (M,5376,43008) | (M,5376,43008) |
| Down | (M,21504,5376) | (M,21504,5376) |

There are 60 layers: five local layers followed by one global layer, repeated.
The local window is 1024. The chosen checkpoint has no MoE block, cross-layer KV
sharing, per-layer input embedding or double-width MLP. The first adapter should
reject those unsupported feature combinations explicitly.

For global RoPE, the official function constructs 64 nonzero inverse
frequencies using exponents divided by the **full head width 512**, then pads
the frequency vector to length 256 with 192 zeros. Thus the rotated pairs join
coordinates 0–63 with coordinates 256–319. Applying a conventional 128-wide
RoPE to the first 128 contiguous coordinates would be a different operation.
This construction should receive a direct oracle test, including positions
around and beyond the local window boundary.

## Implementation and qualification sequence

1. Add versioned configuration/geometry and checkpoint readers. Validate both
   attention types, all normalization vectors, the scalar state, and the absent
   global V projection. Preserve the raw checkpoint configuration and hashes.
2. Create an isolated oracle environment with the pinned modern Transformers
   stack. Leave the active Qwen runtime and its Transformers 4.56.2 stack alone.
   Build a text-only official loader with strict state matching.
3. Qualify a native Gemma4 layer and complete tiny text model against that
   independent oracle. Exercise local/global attention, non-unit layer scalars,
   non-unit learned norm scales, partial RoPE, window boundaries, embeddings,
   final normalization and softcapping. Random tiny weights validate semantics;
   they do not substitute for official-checkpoint qualification.
4. Add versioned fusion support for Gemma4 normalization. Initially preserve
   the shared raw global K projection and branch its postprocessing without
   duplicating the matrix multiplication. Record which postoperations are inside
   each custom kernel; qualify any later multi-output fusion separately.
5. Extend streaming, prepared-weight cache identity, profile/tuner geometry,
   resident timing, input acquisition and archived-bundle validation in new
   versions. The global layer has five actual projection sites; do not fabricate
   a V tuning result or apply the sliding-layer geometry to it.
6. Qualify against official checkpoint tensors and the independent oracle,
   including all layer types and final logits. Validate the actual archived
   launch commands and recovery behavior. Only then clear the explicit Gemma4
   qualification barrier and permit the existing serial queue to run it.

Keep all five algorithms, BF16 output, DEFAULT arithmetic, the same timing
rounds, resident/full-forward distinction and numerical gates. Changes made for
Gemma4 must not modify or reclassify the saved Qwen results.
