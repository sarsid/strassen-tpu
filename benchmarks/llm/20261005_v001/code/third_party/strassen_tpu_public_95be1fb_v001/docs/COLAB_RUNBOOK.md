# Colab TPU runbook

This is the reproduction path for the Qwen3 inference result. Use exactly one
named Colab TPU session. Do not create parallel sessions.

## Measured contract

- Hardware: one Colab `TPU v5 lite` requested as `v5e1`.
- Client: JAX/jaxlib `0.7.2`, libtpu `0.0.21.1`.
- Inputs: BF16; GEMM accumulation: FP32; outputs: BF16.
- Permanent controls: ordinary XLA and blocked cubic Pallas.
- Candidate: product-aware Strassen Pallas.
- Timings: synchronized device execution; compilation and checkpoint transfer
  excluded.
- Promoted tile: `(BM, BN, BK) = (2048, 2048, 512)`.

OAuth credentials remain in the user's CLI configuration. Never place an
authorization code, access token, refresh token, or CLI configuration file in
the repository or a result artifact.

## Allocate one session

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 new -s strassen-v5e --tpu v5e1
colab --auth=oauth2 status -s strassen-v5e
```

If `sessions` shows an existing runtime, reuse or stop it. Do not call
`colab new` twice.

Install the checkpoint and corpus dependencies without changing JAX or
libtpu:

```bash
colab --auth=oauth2 install -s strassen-v5e transformers huggingface_hub datasets
```

## Upload the promoted experiment

The repository is organized for people; Colab receives the small promoted
experiment as a flat directory so its original imports remain traceable:

```bash
colab --auth=oauth2 upload -s strassen-v5e strassen_pallas.py /content/strassen_pallas.py
colab --auth=oauth2 upload -s strassen-v5e mosaic_compat.py /content/mosaic_compat.py
colab --auth=oauth2 upload -s strassen-v5e tools/check_tpu.py /content/check_tpu.py
for file in experiments/qwen3/*.py; do
  colab --auth=oauth2 upload -s strassen-v5e "$file" "/content/$(basename "$file")"
done
colab --auth=oauth2 restart-kernel -s strassen-v5e
```

Confirm both the single TPU and the Pallas lowering path before downloading a
checkpoint:

```bash
colab --auth=oauth2 exec -s strassen-v5e -f check_tpu.py --timeout 300
```

The final output must contain `PALLAS_OK` and `TPU_OK`. A TPU device listing
alone does not catch Mosaic client/runtime skew.

## Select one job

Colab CLI's `exec -f` command cannot pass command-line arguments. Copy
`experiments/qwen3/job.example.json` to a temporary `job.json` and select one
model and stage:

```bash
cp experiments/qwen3/job.example.json job.json
```

```json
{
  "model": "8b",
  "stage": "synthetic",
  "output_dir": "/content/runs"
}
```

Valid models are `8b`, `14b`, and `32b`. Valid stages are:

| Stage | Measurement |
|---|---|
| `tune` | Equal-budget tile selection; optional when reproducing the pinned result |
| `synthetic` | Isolated gate/up plus SwiGLU |
| `layer` | Complete real checkpoint layer 0 |
| `streamed` | All-layer summed resident compute and frozen task gate |
| `quality` | Pinned WikiText-2 teacher-forced quality gate |

Upload the job, restart the kernel to clear imported model constants, and run
the single entry point:

```bash
colab --auth=oauth2 upload -s strassen-v5e job.json /content/job.json
colab --auth=oauth2 restart-kernel -s strassen-v5e
colab --auth=oauth2 exec -s strassen-v5e -f run.py --timeout 14400
```

Repeat by changing only `model` or `stage` in `job.json`. Run `synthetic`,
`layer`, `streamed`, then `quality` for each model. The fixed runner pins the
promoted tile and executes XLA, cubic Pallas, and Strassen in the same process.

## Preserve outputs

Fresh outputs go to `/content/runs`, separate from the immutable public
artifacts. Download each JSONL immediately and inspect its final `verdict`
record before advancing:

```bash
mkdir -p runs
colab --auth=oauth2 download -s strassen-v5e /content/runs/strassen_qwen3_8b_product_aware_inference.jsonl runs/strassen_qwen3_8b_product_aware_inference.jsonl
```

A failed gate is still evidence. Download it; do not replace a published
artifact or weaken a threshold.

## Interpretation rules

1. Ordinary XLA is the deployment-facing control.
2. Cubic Pallas is substrate attribution, not a perfectly epilogue-matched
   algorithm-only control.
3. The all-layer timing is “summed resident layer compute,” not end-to-end
   latency.
4. Checkpoint transfer, embeddings, and final-logit work are excluded.
5. WikiText is teacher-forced next-token evidence, not generation or
   downstream-task accuracy.

## Release the TPU

```bash
colab --auth=oauth2 stop -s strassen-v5e
colab --auth=oauth2 sessions
```

The last command should report no active runtime.
