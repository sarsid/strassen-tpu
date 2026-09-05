# Strassen GEMM on TPU

Public research snapshot of one-level, tile-local Strassen GEMM in JAX Pallas
(Mosaic), on a single TPU v5e or v6e (Trillium). The kernel replaces eight
quadrant multiplies with seven and folds the surrounding elementwise work —
SwiGLU, residual adds, per-head RMSNorm, RoPE — into the same kernel, so the
intermediate never reaches HBM.

All timings are same-run measurements on one Colab chip. Compilation and
checkpoint transfer are excluded. Every artifact is indexed with its SHA-256
in `evidence/qwen3/README.md`; `tools/verify_evidence.py` checks them.

## Main result

Qwen3-32B, batch 8 x 1024 tokens, real weights:

| Workload | v5e | v6e |
|---|---:|---:|
| **Streamed, all 64 layers (registered gate)** | `1.0544x` | **`1.2950x`** |
| Single transformer block, layer 0 | `1.1566x` | `1.2093x` |
| Pure GEMM, no epilogue, vs XLA | `1.1312x` | `1.1117x` |
| Pure GEMM, vs matched cubic at the same tile | — | `1.0896x` |

The last row is the algorithm by itself. **Rank-7 is worth about 9%**, below
the `8/7 = 1.1429x` ceiling as a genuine Strassen saving should be. Everything
above that comes from fusion and tile selection, not from the multiply count.
That makes this a scheduling and fusion-boundary result rather than a
fast-matmul one.

Quality passes the registered gates at every width (8B/14B/32B): streamed
top-1 agreement `0.99963`, mean KL `1.374e-4` nats, WikiText-2 perplexity
delta under `4e-4`, HellaSwag choice agreement `1.000`, LAMBADA `0.991`.

## What actually produced the speedup

Four findings, each measured on both chips:

1. **Fuse the epilogue or don't route the site.** Every projection wins in
   isolation — `k` most of all, at `1.175x` — yet routing `q`/`k`/`v` through
   the kernel *loses* inside a real block, because it breaks fusion XLA would
   otherwise do. `down` and `o` carry a fusable residual and win. Fusing
   `q_norm`+RoPE into the q/k kernel converts them from losing sites into the
   single biggest win here: **+8% of the block on both chips**.
2. **Never raise the XLA scoped-vmem flag to give the kernel a bigger budget.**
   The two are independent. On one chip at one tile the 128 MiB flag costs
   native XLA `37%` at the layer — but only `4.1%` on a bare GEMM, so the cost
   lands on the surrounding layer work, not the matmul.
3. **Tile heuristics do not port across generations.** A `bk=512` heuristic
   tuned for v5e is near-optimal there and `4.4%` off on v6e, whose 256x256
   MXU wants the whole contraction depth in one panel (`bk=5120`, no K loop).
4. **Product-aware quadrant finalization is v5e-specific.** Worth `-0.62 ms`
   there and `+0.05` to `+0.21 ms` on v6e across eight measurements: the MXU
   got roughly 4.7x faster while the VPU did not, so the epilogue no longer
   has room to hide behind the products.

**Measure in a full transformer block, never in isolation.** Isolated screens
misled in both directions — they said all five projection sites advance, and
they scored the q/k fusion at `3.05x` against a baseline that materialises
work XLA fuses away.

## Layout

| | |
|---|---|
| `strassen_pallas.py` | the kernel: 7 products, 4 FP32 quadrant accumulators, the `swiglu` / `residual_add` / `bias_add` / `qk_norm_rope` epilogues, and the free weight relayouts they need |
| `experiments/qwen3/` | block, streamed, tile-selection and pure-GEMM harnesses |
| `evidence/qwen3/` | raw JSONL for every claim above, hashed and indexed |
| `docs/RESULTS.md` | per-experiment protocol and numbers |
| `docs/COLAB_RUNBOOK.md` | how the runs are driven |
| `tools/verify_evidence.py` | re-hashes every artifact against the index |

## Limitations

- One chip, free-tier Colab; no multi-chip and no serving-stack integration.
- BF16 only. The v5e/v6e MXU also offers INT8; there is no FP8 story here.
- Two-level Strassen is closed: VMEM on v5e, and on v6e a second halving puts
  K at 128 inside a 256-wide MXU.
- The matched cubic control runs at about `93%` of XLA's GEMM efficiency while
  the Strassen substrate reaches `99%`, so "vs cubic" ratios carry a few
  points of substrate artifact and overstate the algorithm.
- The XLA baseline is measured with the scoped-vmem flag set to 48 MiB, which
  beat 128 MiB. XLA has not been measured at its own unset default, so the
  baseline may not be at its optimum.
- The streamed `1.2950x` sits above the single-block `1.2093x` under the same
  policy. The harnesses differ in protocol and the gap is not yet explained;
  it should not be read as the fusion improving with depth.

## Environment

JAX `0.7.2`, libtpu `0.0.21.1`, with the Mosaic IR-v7 compatibility override
in `mosaic_compat.py`. Upgrading either breaks the override.
