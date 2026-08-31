# Colab TPU runbook

This is the reproduction path for the Qwen3 scaling result. Use exactly one
named Colab TPU session so parallel runtimes cannot consume credits or alter
the comparison.

## Measured contract

- Hardware: one Colab `TPU v5 lite` requested as `v5e1`.
- Client: JAX/jaxlib `0.7.2`, libtpu `0.0.21.1`.
- CLI: Colab CLI with OAuth2 authentication.
- Inputs: BF16; GEMM accumulation: FP32; outputs: BF16.
- Permanent arms: ordinary XLA, blocked cubic Pallas, product-aware Strassen.
- Timings: synchronized device execution; compilation and checkpoint transfer
  excluded.
- Promoted tile: `(BM, BN, BK) = (2048, 2048, 512)`.

OAuth credentials remain in the user's CLI configuration. Never place an
authorization code, access token, refresh token, or CLI configuration file in
the repository or a result artifact.

## One-session lifecycle

From the repository root, verify that no runtime is active before allocating:

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 new -s strassen-v5e --tpu v5e1
colab --auth=oauth2 status -s strassen-v5e
```

If `sessions` shows an existing runtime, reuse or stop it. Do not create a
second session.

Install only the checkpoint and corpus dependencies. Do not upgrade JAX or
libtpu inside the measured runtime.

```bash
colab --auth=oauth2 install -s strassen-v5e transformers huggingface_hub datasets
```

Upload the root Python files flat into `/content`, then restart the persistent
Python kernel so it cannot retain stale imports:

```bash
for file in *.py; do
  colab --auth=oauth2 upload -s strassen-v5e "$file" "/content/$file"
done
colab --auth=oauth2 restart-kernel -s strassen-v5e
```

Before spending time on a checkpoint, prove that this runtime is a single TPU
and that a Pallas kernel lowers successfully:

```bash
colab --auth=oauth2 exec -s strassen-v5e -f tools_check_tpu.py --timeout 300
```

The final output must contain `PALLAS_OK` and `TPU_OK`. A TPU device listing by
itself is insufficient because the client and remote Mosaic runtime can drift.

## Reproduction order

The checked-in wrappers already pin the selected tile. Run the permanent
experiments in this order for each model size:

| Stage | 8B | 14B | 32B |
|---|---|---|---|
| Isolated gate/up+SwiGLU | `run_qwen3_scaling_synthetic_8b.py` | `run_qwen3_scaling_synthetic_14b.py` | `run_qwen3_scaling_synthetic_32b.py` |
| Real layer 0 | `run_qwen3_scaling_layer0_8b.py` | `run_qwen3_scaling_layer0_14b.py` | `run_qwen3_scaling_layer0_32b.py` |
| All-layer resident compute | `run_qwen3_scaling_streamed_8b.py` | `run_qwen3_scaling_streamed_14b.py` | `run_qwen3_scaling_streamed_32b.py` |
| Natural-text quality | `run_qwen3_natural_gate_8b.py` | `run_qwen3_natural_gate_14b.py` | `run_qwen3_natural_gate_32b.py` |

Example:

```bash
colab --auth=oauth2 exec -s strassen-v5e -f run_qwen3_scaling_synthetic_8b.py --timeout 7200
colab --auth=oauth2 exec -s strassen-v5e -f run_qwen3_scaling_layer0_8b.py --timeout 7200
colab --auth=oauth2 exec -s strassen-v5e -f run_qwen3_scaling_streamed_8b.py --timeout 14400
colab --auth=oauth2 exec -s strassen-v5e -f run_qwen3_natural_gate_8b.py --timeout 14400
```

Download each JSONL immediately after its run. Inspect its final `verdict`
record before advancing. A failure is evidence and should also be downloaded.

The optional equal-budget tuning wrappers are
`run_qwen3_scaling_tune_{8b,14b,32b}.py`. They are needed to repeat tile
selection, not to reproduce the promoted fixed-tile result.

## Interpretation rules

1. Treat ordinary XLA as the primary deployment-facing control.
2. Treat cubic Pallas as substrate attribution, not a perfectly
   epilogue-matched algorithm-only control.
3. Call the all-layer timing “summed resident layer compute,” never
   end-to-end latency.
4. Do not include checkpoint download or layer streaming in a compute ratio.
5. Call the WikiText result teacher-forced next-token evidence, not generation
   or downstream-task accuracy.
6. Preserve the declared quality thresholds and raw artifacts unchanged.

## Release the TPU

When the requested campaign is complete:

```bash
colab --auth=oauth2 stop -s strassen-v5e
colab --auth=oauth2 sessions
```

The last command should report no active runtime.
