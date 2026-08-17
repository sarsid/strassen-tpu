# Colab TPU runbook

This is the shortest reproducible path from this checkout to the saved TPU
results. It uses one named Colab session at a time so no second runtime can
consume compute credits accidentally.

## Fixed contract

- Hardware: one Colab `TPU v5 lite` requested as `v5e1`.
- Runtime measured here: Python 3.12.13, JAX/jaxlib 0.7.2, libtpu 0.0.21.
- CLI measured here: Colab CLI 0.6.0 with OAuth2 authentication.
- Performance protocol: 10 synchronized warmups and 20 synchronized samples;
  compilation, allocation, and host transfer are excluded.
- Comparator: same-run native `jax.jit(jnp.matmul)` with BF16 inputs,
  `Precision.DEFAULT`, FP32 accumulation, and BF16 output.
- Main tile/profile: `max48`, `(BM, BN, BK) = (2048, 2048, 512)`.
- Full-model gate: hidden and logit L2-relative and max-normalized error each
  at most 2%, top-1 agreement at least 95%, and absolute next-token-loss change
  at most 0.01. Do not alter these thresholds after seeing a result.

OAuth credentials and tokens live in the user's CLI configuration and must
never be copied into this repository or a result file.

## One-session lifecycle

Run every command from the repository root. First prove that no runtime is
active, then allocate exactly one v5e session:

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 new -s strassen-v5e --tpu v5e1
colab --auth=oauth2 status -s strassen-v5e
```

If `sessions` lists another runtime, reuse it or stop it before creating
`strassen-v5e`. Do not run `colab new` twice.

The Colab image already supplies JAX, NumPy, `ml_dtypes`, and `requests`. The
real-checkpoint experiments also use the tokenizer and checkpoint downloader:

```bash
colab --auth=oauth2 install -s strassen-v5e transformers huggingface_hub
```

Upload the current source. Colab's Python kernel persists between executions,
so restart it after installing packages and after any upload that changes an
imported module:

```bash
for file in strassen_pallas.py benchmarks/*.py benchmarks/screens/*.py; do
  colab --auth=oauth2 upload -s strassen-v5e "$file" "/content/$(basename "$file")"
done
colab --auth=oauth2 restart-kernel -s strassen-v5e
```

All sources upload flat into `/content`, so the benchmarks' mutual imports
work unchanged regardless of the repository layout.

Do not upgrade JAX or libtpu in the measured session. `strassen_pallas.py`
contains a narrowly guarded compatibility path for the measured JAX/libtpu
pair.

Run one experiment, download its JSONL immediately, and inspect its final
record before moving on:

```bash
colab --auth=oauth2 exec -s strassen-v5e -f benchmark_bf16_dispatch.py --timeout 1800
colab --auth=oauth2 download -s strassen-v5e /content/results/strassen_bf16_dispatch.jsonl results/strassen_bf16_dispatch.jsonl
sha256sum results/strassen_bf16_dispatch.jsonl
```

Some research harnesses deliberately raise `RuntimeError` when a candidate
fails its gate. That nonzero experiment outcome is evidence, not a missing
result: download the JSONL before deciding whether to continue.

When all requested runs and downloads are complete, release the TPU and verify
that the account has no active session:

```bash
colab --auth=oauth2 stop -s strassen-v5e
colab --auth=oauth2 sessions
```

## Experiment order

Use the smallest applicable gate first. A candidate should not consume a
full-checkpoint run until it improves real-layer accuracy and retains a
statistically significant same-run speedup.

| Stage | Command passed to `colab exec -f` | Output | Purpose |
|---|---|---|---|
| Kernel smoke/performance | `benchmark_strassen.py` | `/content/results/strassen_v5e_tuned.jsonl` | Square accuracy and speed smoke test |
| Release GEMMs | `benchmark_bf16_dispatch.py` | `/content/results/strassen_bf16_dispatch.jsonl` | Five promoted large BF16 shapes against native |
| Product schedule | `benchmark_product_order.py` | `/content/results/strassen_product_order.jsonl` | Original versus dependency-spaced seven products |
| Synthetic block | `benchmark_model_block.py` | `/content/results/strassen_model_block.jsonl` | Fixed block; only selected GEMMs vary |
| Real layer | `benchmark_checkpoint_layer.py` | `/content/results/strassen_checkpoint_layer.jsonl` | Pinned Mistral layer 0 accuracy and resident timing |
| Full all-layer gate | `benchmark_checkpoint_model.py` | `/content/results/strassen_checkpoint_model.jsonl` | Streamed 32-layer quality; not latency |
| Sparse frontier | `benchmark_checkpoint_schedules.py` | `/content/results/strassen_checkpoint_schedules.jsonl` | Predeclared uniform placements |
| Terminal calibration | `benchmark_checkpoint_terminal.py` | `/content/results/strassen_checkpoint_terminal.jsonl` | First/last-layer placement screen |
| Frozen holdout | `benchmark_checkpoint_holdout.py` | `/content/results/strassen_checkpoint_holdout.jsonl` | Disjoint text; no policy retuning |
| Equivalent permutations | `benchmark_checkpoint_permutations.py` | `/content/results/strassen_checkpoint_permutations.jsonl` | Eight no-extra-product block permutations |
| Dual formula | `benchmark_checkpoint_dual.py` | `/content/results/strassen_checkpoint_dual.jsonl` | Opposite classical rank-7 basis |
| Power-of-two balance | `benchmark_checkpoint_equilibration.py` | `/content/results/strassen_checkpoint_equilibration.jsonl` | Precomputed channel equilibration |
| Error correlation | `benchmark_checkpoint_error_correlation.py` | `/content/results/strassen_checkpoint_error_correlation.jsonl` | Test whether formula errors can cancel |
| Formula sequences | `benchmark_checkpoint_variant_sequences.py` | `/content/results/strassen_checkpoint_variant_sequences.jsonl` | All-layer rotation under unchanged gate |
| Wider outputs | `benchmark_checkpoint_wide_outputs.py` | `/content/results/strassen_checkpoint_wide_outputs.jsonl` | Retain FP32 across adjacent MLP operations |
| Cubic attribution control | `benchmark_cubic_control.py` | `/content/results/strassen_cubic_control.jsonl` | Native vs two cubic Pallas controls vs promoted Strassen |
| Native flag sensitivity | `run_native_vmem_default.py`, then restart, then `run_native_vmem_48m.py` | `/content/results/native_vmem_default.jsonl`, `/content/results/native_vmem_scoped_48m.jsonl` | Native-only timing with and without the scoped-VMEM flag; each mode needs a fresh kernel |
| Lower-growth formula | `benchmark_checkpoint_powers.py` | `/content/results/strassen_checkpoint_powers.jsonl` | Real-layer screen for the experimental power-of-two rank-7 formula |

For an experiment named `benchmark_X.py`, the normal pattern is:

```bash
colab --auth=oauth2 exec -s strassen-v5e -f benchmark_X.py --timeout 7200
colab --auth=oauth2 download -s strassen-v5e /content/results/strassen_X.jsonl results/strassen_X.jsonl
```

Use a longer timeout for full-checkpoint passes. The first such pass downloads
the two pinned Mistral shards (14.48 GB total) into
`/content/mistral-7b-v0.1`; later checkpoint experiments in the same session
reuse them. Never create a second session to run another experiment in
parallel.

## Iteration discipline

1. Change one arithmetic or scheduling idea at a time.
2. Preserve a same-run matched-native executable, inputs, barriers, warmups,
   timing count, and error metrics.
3. Upload every changed import, restart the persistent kernel, then run.
4. Screen real layer 0 before the streamed 32-layer gate.
5. Require both lower error and retained speed; a tiny layer improvement that
   cannot plausibly close the 2% full-model gate does not advance.
6. Download the raw JSONL even on failure, compute its SHA-256, and add the
   decision to `EXPERIMENTS.md` and `results/README.md`.
7. Stop the session and confirm that `colab sessions` is empty.

## Known scope limits

The full-checkpoint scripts stream one layer of weights at a time, so they are
quality experiments rather than end-to-end latency measurements. The current
calibration corpus deterministically repeats eight passages to fill 8x1024
tokens and contains 190 unique token IDs. Keep the existing results and gate
unchanged, but use a broader frozen corpus before making a deployment-quality
claim.
