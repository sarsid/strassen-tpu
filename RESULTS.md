# Results (preliminary)

One-level classical Strassen GEMM in JAX Pallas on a single TPU v5e, against
same-run native `jnp.matmul` (BF16 in, FP32 accumulation, BF16 out; 10
synchronized warmups, 20 synchronized samples; predeclared gates). Raw JSONL
evidence with SHA-256 hashes is in `results/`. A full writeup is
forthcoming.

## GEMM level

- Large aligned BF16 GEMMs: `1.071–1.102x` native (squares 8192–16384:
  `1.088–1.100x`; on 8192³, 183 → 200 classical-equivalent TFLOP/s against a
  baseline at ~93% of the chip's 197 TFLOP/s peak).
- Attribution: cubic Pallas kernels on the identical substrate never beat
  native (`0.98–1.00x` full-tile, `0.93–0.95x` quadrant-blocked); the win is
  the seven-product formula. Replicated across sessions.
- FP32 `HIGHEST`: `1.170–1.242x` at bit-level agreement with the matched
  native path (native FP32 costs `5.95–6.42x` native BF16 on these shapes).
- Accuracy cost in BF16: `2.832x` native RMSE, attributed by ablation
  entirely to the extra BF16 rounding of Strassen's input pre-additions.
- Decode shapes (M=32–256) tie or lose and stay native.

## Full model (Mistral-7B v0.1, pinned checkpoint)

- Real layer, both MLP sites: `1.019x` at `0.303%` layer L2.
- An outlier-panel hybrid (offline weight permutation concentrating the
  model's outlier channels + one exact K-slice computed with conventional
  multiplication, `<1%` extra multiplies) cuts full-model error ~3x:
  streamed 32-layer hidden L2 `7.0% → 2.3%`, logit L2 `3.1% → 1.0%`, top-1
  agreement `99.95–100%`, with calibration frozen and validated on disjoint
  text.
- A predeclared 2% worst-case tensor bound remains failed for all variants
  tested; the residual is diffuse bulk pre-add rounding.
- Task-level gate, thresholds registered before evaluation (WikiText-2
  test; loss delta ≤ 0.01, top-1 ≥ 99%, mean next-token KL ≤ 0.005 nats):
  the plain Strassen kernel **fails** (top-1 `98.60%`); the frozen hybrid
  **passes** (loss delta `0.00003`, top-1 `99.26%`, KL `0.00026`).

Reproduction: `COLAB_RUNBOOK.md`. Kernel and dispatcher:
`strassen_pallas.py`.
