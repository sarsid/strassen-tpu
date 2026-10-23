# LLM Benchmark 3 (Qwen3-14B)

This new experiment was authorized on October 17, 2026. Local checks passed,
the frozen six-batch supervisor has started, and Colab allocation has been
requested. No Qwen14 performance results are claimed in this initial snapshot.
Previous Qwen32 and Gemma benchmark folders are unchanged.

- Qwen/Qwen3-14B, parent-pinned revision 40c069824f4251a91eefaf281ebe4c544efd3e18.
- One v6e, JAX/jaxlib 0.11.2, libtpu 0.0.48 throughout.
- BF16 output; Native, Tuned Native, Tuned Cubic, Tuned Strassen S1.
- Sequence1024; batches1/4/8/16/32/64, executed8/16/32/64/4/1.
- Fresh model-specific per-batch tuning, frozen before confirmation/quality.
- Separate resident projections, block0, warmed layer-sum and cached-disk-offloaded prefill.
- Same repetitions, predictive-quality gates and parent Qwen computation as Benchmark1.

See [the registered protocol](PROTOCOL.md) and [the exact frozen code](code/initial_20261017_v001/source).
The initial local validation failed because the selected Python environment
lacked requests. The unchanged checks passed using the existing Qwen CPU
validation environment; both records are preserved under validation.
These checks cover geometry, legal candidate sets, unchanged computation/timing
functions, reduced CPU capture correctness, true B64 abstract tracing, launcher
integrity, lost responses, checkpoint hashes, and retain-on-error behavior.
They are not TPU compilation or performance results.

Run identity:20261017-qwen3-14b-v001. The local live dashboard is on port8790.
Results and summaries will be added separately with immutable run/attempt IDs.
Only code and numerical records are published; model weights, activation tensors
and credentials remain private. Old measurements will never be overwritten.
