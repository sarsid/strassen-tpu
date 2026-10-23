# Gemma4 official oracle and reusable input-template qualification

The local official-reference export and input-template helpers passed four
tests against a synthetic two-layer checkpoint. They perform no downloads,
TPU allocation or queue changes. Gemma4 remains `awaiting_qualification`.

Sources: `tools/gemma4_oracle_inputs_v001.py` and
`tests/test_gemma4_oracle_inputs_v001.py`, committed in `c9f40149c` before the
archived run `runs/20261004T022640Z-gemma4-oracle-inputs-qualification-v001-0146a3`.
Archive commit: `6bbd2caca`. The four test cases passed; reference setup uses the
independent pinned PyTorch loader, retaining nonunit scalar dtypes and FP32
rotary buffers. No synthetic checkpoint or cache weight bytes were archived.

## Export contract

The exporter requires PyTorch 2.8.0 and Transformers 5.5.0, then verifies the
installed Gemma4 model, configuration and RoPE source hashes against the
qualified official snapshot. Its identity also includes the loader and export
helper hashes. A caller-verified local checkpoint and one int32 sequence of
2--64 tokens produce every layer input/output and all vocabulary logits.
Finite values, output shapes and complete layer coverage are checked. The
result is labelled `reference_exported`, never a passed checkpoint gate.

The test used 64 tokens and both attention types with an eight-token local
window. Every exported activation and logit matched an independent forward
through the original synthetic official model exactly; every artifact hash was
verified. Invalid oracle token types, lengths and batch counts were rejected.

## Input cache contract

A complete template binds the model identity, official/export source identity
and every file hash. It retains full train/test token arrays, provenance, the
fixed oracle and source summary. Reuse verifies the full file set and content
before selecting the requested batch/sequence window. It preserves exact
consecutive int32 tokens, prohibits repetition when the corpus is too short,
and leaves the separate train/test streams and oracle tokens intact.

Tests covered B1/S17, B4/S32 and B8/S16 windows; changed model/source identities,
modified bytes, unlisted files, escaping symlinks and attempts to overwrite an
already published cache were rejected. Synthetic evidence retained its explicit
`actual_checkpoint: false` label after cache reuse.

These helpers do not acquire the pinned 31B checkpoint or claim that its outputs
have been checked. Remaining integration includes verified acquisition, the
modern remote environment, resident and cached model-session entry points,
official-checkpoint numerical gates, and archived launch/recovery qualification.
The current Qwen runtime and frozen measurement sources are unchanged.
