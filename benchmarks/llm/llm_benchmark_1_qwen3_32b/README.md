# LLM Benchmark 1 (Qwen3-32B)

Completed October 16, 2026. Six batches × four algorithms, sequence length 1,024, one TPU v6e. All selected runs completed and all registered predictive-quality gates passed. The TPU was released; no further experiment is queued.

**Strassen S1 reduces resident computation time versus Tuned Native at every tested batch size. Offloaded prefill gains are much smaller because serial weight loading and common preparation dominate.**

**Keep B64 separate in cross-batch conclusions:** B1–B32 used JAX/jaxlib 0.7.2 and libtpu 0.0.21.1. B64 used JAX/jaxlib 0.11.2 and libtpu 0.0.48 after compiler stack-overflow failures. Every within-batch comparison uses the same stack. Do not interpret the B32→B64 change as a pure batch-scaling effect.

## Resident transformer computation

Milliseconds: sum of 64 individually warmed layer means, including attention, normalization, residuals and device-memory traffic. Excludes weight transfers, compilation, embeddings, final norm and vocabulary head. This is not uninterrupted fully resident model latency.

| Batch | Tokens | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 speedup vs tuned | S1 time saved vs tuned |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 160.212 | 146.754 | 157.063 | 135.452 | 1.083× | 7.70% |
| 4 | 4,096 | 632.140 | 542.953 | 530.530 | 419.134 | 1.295× | 22.80% |
| 8 | 8,192 | 1,394.543 | 1,128.694 | 1,078.036 | 869.207 | 1.299× | 22.99% |
| 16 | 16,384 | 2,908.855 | 2,293.642 | 2,157.658 | 1,744.330 | 1.315× | 23.95% |
| 32 | 32,768 | 5,968.557 | 4,788.306 | 4,594.628 | 3,952.080 | 1.212× | 17.46% |
| 64* | 65,536 | 9,509.365 | 7,812.359 | 8,056.431 | 5,984.390 | 1.305× | 23.40% |

*B64 uses the separately registered compiler stack.

## Serialized offloaded prefill

Seconds, arithmetic mean of three measured passes per arm after one full warmup. Includes cached-checkpoint reads, common weight preparation/packing, recurring H2D transfers, embeddings, all layers, final norm and last-position logits. Excludes network downloads and compilation. No decode is measured. The common loader performs Strassen-related packing for every arm; this is not a comparison against an independently optimized native serving system.

| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 time saved vs tuned |
|---:|---:|---:|---:|---:|---:|
| 1 | 131.066 | 130.274 | 130.720 | 130.557 | -0.22% |
| 4 | 129.880 | 129.706 | 129.992 | 129.002 | 0.54% |
| 8 | 130.870 | 129.804 | 130.111 | 129.864 | -0.05% |
| 16 | 131.430 | 131.146 | 130.620 | 130.189 | 0.73% |
| 32 | 137.246 | 135.766 | 135.789 | 135.703 | 0.05% |
| 64* | 140.469 | 139.869 | 139.318 | 137.924 | 1.39% |

Small offloaded differences are descriptive results from only three passes, not established deployment gains. Raw samples and ordering are preserved; do not infer an exact transfer overhead by subtracting the two different timing procedures.

## Natural-text predictive quality

Four contiguous WikiText-2 test segments per batch. Each arm sees identical tokens within a batch; larger batches score more tokens, so absolute perplexities are not quality rankings across batch sizes. These are next-token measurements against the matched Native implementation, not downstream task accuracy or an independent Transformers-equivalence test.

| Batch | Positions | Native PPL | Tuned Native PPL | Cubic PPL | S1 PPL | S1 top-1 agreement | S1 KL (nats) | S1 abs NLL delta | S1 logit relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,092 | 8.641057 | 8.643964 | 8.624266 | 8.634545 | 98.045% | 0.001790 | 0.000754 | 3.846% |
| 4 | 16,368 | 7.548506 | 7.546123 | 7.543883 | 7.554202 | 98.283% | 0.001553 | 0.000754 | 3.112% |
| 8 | 32,736 | 8.457450 | 8.457450 | 8.459416 | 8.464308 | 98.262% | 0.001552 | 0.000811 | 3.017% |
| 16 | 65,472 | 8.668520 | 8.667798 | 8.666828 | 8.664493 | 98.289% | 0.001505 | 0.000465 | 3.186% |
| 32 | 130,944 | 8.256327 | 8.255605 | 8.254960 | 8.252154 | 98.296% | 0.001479 | 0.000506 | 3.165% |
| 64 | 261,888 | 8.530697 | 8.530697 | 8.531011 | 8.527593 | 98.244% | 0.001499 | 0.000364 | 3.162% |

All arms pass the natural-text thresholds: absolute NLL delta ≤0.01 nats, KL ≤0.02 nats, top-1 agreement ≥97%, with finite outputs. All repetitive-text gates also pass (NLL ≤0.01, KL ≤0.005, top-1 ≥99%). Numerical differences are nonzero; near-identical perplexity is not identical predictions or proof of a quality improvement.

## Frozen tuning choices

Tiles are `(BM, BN, BK)` in the parent implementation. Q/K uses the fixed published fused tile (BM clipped for B1); it was not independently tuned. V remains native. Cubic and Strassen use their own gate/up and O/down winners. Native tuning selects complete-block scoped VMEM; custom arms retain the parent 48 MiB XLA scope and separate Pallas limits.

| Batch | Native VMEM MiB | S1 gate/up | Cubic gate/up | Fixed Q/K | S1 O | S1 down | Cubic O | Cubic down |
|---:|---:|---|---|---|---|---|---|---|
| 1 | 64.0 | 1024×2560×5120 | 1024×2048×5120 | 1024×1024×5120 | 1024×2560×2048 | 1024×2560×1024 | 1024×5120×512 | 1024×5120×1024 |
| 4 | 48.0 | 1024×2560×5120 | 1024×2560×5120 | 2048×1024×5120 | 2048×512×8192 | 2048×2560×1024 | 2048×512×8192 | 2048×5120×1024 |
| 8 | 48.0 | 1024×2560×5120 | 2048×1024×5120 | 2048×1024×5120 | 2048×512×8192 | 2048×2560×1024 | 1024×1024×8192 | 4096×2560×1024 |
| 16 | 48.0 | 2048×1024×5120 | 2048×1024×5120 | 2048×1024×5120 | 2048×512×8192 | 2048×2560×1024 | 2048×512×8192 | 2048×5120×1024 |
| 32 | 48.0 | 2048×1024×5120 | 2048×1024×5120 | 2048×1024×5120 | 2048×512×8192 | 2048×2560×1024 | 2048×512×8192 | 2048×5120×1024 |
| 64 | 112.0 | 2048×1024×5120 | 2048×1024×5120 | 2048×1024×5120 | 2048×512×8192 | 2048×2560×1024 | 2048×512×8192 | 4096×2560×1024 |

## Method, provenance and limitations

- Official checkpoint: `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`; pinned parent code: `sarsid/strassen-tpu@95be1fb088656a89813b04492e1d77c66b36ccf9`.
- Parent numerical contract: BF16 outputs, DEFAULT operand precision, parent FP32 accumulation and BF16 rounding boundaries. One-level product-aware fused SwiGLU and fused Q/K normalization/RoPE; no S2, padding or decode.
- Parent kernel/math implementation is preserved. This is a four-arm batch-sweep extension of the reproduced parent study, not a byte-for-byte reproduction of the original benchmark. O-site tuning measures its actual plain-projection use; down includes residual. Gate/up tuning uses seeded synthetic operands; confirmation uses official weights and real propagated activations.
- Tuning: two warmups/five samples, frozen before confirmation. Projection and block confirmation: three warmups/20 samples. Each layer: two warmups/five samples, with rotating/reversing arm order. Each arm propagates its own activations. Paired layer-bootstrap intervals are preserved in `summary.json`; they describe that sampling procedure, not variation across allocations.
- B1/B4/B8/B16 come from the original completed sweep. B32 comes from the memory-repaired retry with its original frozen profile. B64 comes from the completed current-stack retry, reusing the exact SHA-verified profile recovered after runtime loss; no incomplete confirmation timings are pooled into a final result.
- B64 tuning provenance is partial: the runtime was lost after tuning completed, and saved snapshots can omit candidate events. The exact selected profile was recovered with its original SHA-256 from two complete frozen-profile records. The package does not claim a complete B64 candidate ledger; final qualification, confirmation, quality and prefill records are complete on the replacement runtime.
- B32 capture buffers were reduced; B64 uses a segmented BF16-boundary capture and qualification. These setup changes are versioned and preserved. Old compiler failures, controller timeout, runtime loss, and the pre-allocation superseded request remain in the failure/history evidence.
- The 32B checkpoint does not fit fully on this single device. Resident-layer savings establish a compute benefit; the separately measured offload pipeline is the actual transfer-inclusive result. Do not present resident-layer totals as whole-model serving latency.

## Contents and offline verification

- [summary.json](summary.json): all four arms, raw timing samples, all 64 layer errors, projection/block results, natural/repetitive errors, prefill passes, selected tiles, versions and token hashes.
- [evidence/index.json](evidence/index.json): original artifact paths mapped to lossless SHA-256-addressed gzip files. Includes all original batch results and failed attempts in the selected study roots; earlier unrelated LLM campaigns remain in their historical folders.
- [provenance/index.json](provenance/index.json): exact source archives/manifests, run completions, environment freezes and result-selection records. Source snapshots are immutable historical evidence; do not execute archived controllers automatically.
- [validation.json](validation.json): integrity, sample-count, finiteness, arithmetic, quality-gate and report-reconstruction checks. No TPU is required.

From this directory run `python3 analyze.py --check`. To regenerate into a fresh directory, run `python3 analyze.py --output /tmp/qwen32-report`. To recover any raw file, find its `blob` in `evidence/index.json` and run `gzip -dc <blob> > <destination>`; the index records its original SHA-256 and size. Source archives are gzip-compressed original tar bytes.

No credentials, checkpoint weights, private authentication state or virtual environments are included. No new experiment was run to build this report.
