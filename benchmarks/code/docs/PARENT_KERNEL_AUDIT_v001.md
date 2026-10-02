# Parent repository audit: v5e, v6e, and the current MM study

Reviewed public branch `public-preview` at commit `95be1fb088656a89813b04492e1d77c66b36ccf9`, fetched directly from GitHub on 2026-09-24 UTC. The cached GitHub landing page described an older v5e-only snapshot; the pinned live source contains v6e and Qwen3 fusion work. Source and raw evidence take precedence over cached pages or stale descriptive metadata.

This is a static code review and verification of existing measurements. No parent code, TPU kernel, or new model experiment was executed. The companion audit script verifies all 33 indexed Qwen3 artifact hashes, snapshots the relevant code/evidence, confirms that our reviewed current files match the executed source snapshots, and records the candidate-space comparison in `audit.json`.

## Overall assessment

There are meaningful untested optimization opportunities. Our latest 168-shape result establishes the behavior of the candidates we actually offered; it does not establish that our v6e search covers the parent repository's approach, or the best achievable Strassen implementation.

The biggest bare-MM gap is deep, shape-dividing K panels coupled with an accumulator/output contract that permits them. The biggest application gap is carrying fusion through the real transformer block. Several fusion mechanisms are already in our older N8/N9 code, but are not active in the recent larger-model composed comparison. Fused q/k normalization plus RoPE is genuinely absent from our current `src/strassen_mm` implementation.

## How the parent implements the generations

Both generations use the same tile-local, one-level classical seven-product formula in `strassen_pallas.py`. The generation-specific choices are supplied through tiles, kernel VMEM budgets, site routing and epilogues; there is no separate v6e Strassen formula. The old generic `tuned_matmul` dispatcher should not be confused with the explicit experiment configurations.

Each program loads A/B tiles through Pallas BlockSpecs, performs BF16 combinations and FP32 dot accumulation, and keeps intermediate products local rather than writing seven matrices to HBM. The ordinary dependency-spaced product order is P4, P6, P5, P2, P7, P3, P1. Both of these mechanisms already exist in our kernels.

For BF16 output, the parent retains four FP32 output quadrants and casts on store. For FP32 output, its output tile itself is the accumulator. Our original one-level implementation has those four-quadrant/output-accumulator choices; our newer v6e deferred path instead retains seven product accumulators and reconstructs after K. This trades less repeated reconstruction for more persistent storage. It is a choice to compare at matched tiles, not an automatic advantage for either implementation.

Parent source: [kernel accumulation](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py#L230-L379), [product order](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py#L556-L573), [output contract](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py#L1044-L1077).

## Concrete comparison

| Mechanism | Parent | Our current status | Implication |
|---|---|---|---|
| Deep contraction panels | v6e bare GEMM selects BM,BN,BK = 2048,1024,5120 at M,K,N = 8192,5120,51200; BK equals all K. The streamed o projection uses BK=8192. | The v6e suite pool caps BK at 4096. No BK=2560,5120,8192,12800. Four/six shapes-only shortlisted tiles; original implementations only on the first two. | A concrete missing search region; test shape divisors and feasible full-K panels. |
| Output precision and storage | Promoted bare GEMM returns BF16 with FP32 accumulation. | The 168-shape grid returns FP32 for every method. New deferred S1 retains seven FP32 half-output products. | BF16 output halves external result bytes relative to FP32, and changes buffering and accumulator choices. This requires a separate matched contract, not silently changing the existing grid. |
| SwiGLU/GeGLU, residual and bias fusion | Work executes before the output leaves Pallas. SwiGLU writes only N/2 channels. | SwiGLU/residual fusion, packing and early finalization already exist in `kernels_n8_v001.py`; the recent larger-model `model_composed_v002.Layer` calls gate/up, activation, down and suffix separately. | Reuse and retune existing code for a new fused application study; do not call this entirely missing functionality. |
| Fused q/k RMSNorm + RoPE | `epilogue='qk_norm_rope'`, paired channel relayout, segmented per-head norm and rotation inside the projection. | q/k/o remain ordinary XLA in the composed prefix. No corresponding fused Pallas epilogue exists. | A substantial new integration opportunity for Qwen3; evaluate within a complete block. |
| Static weight relayout | Gate/up partners and RoPE channel partners are placed in the same tile; prepared layouts are reused. | Older N8 supports packed gate/up weights and separately measured preparation. Recent large-model checkpoints concatenate gate/up conventionally. | Amortize relayout across repeated inference; charge it explicitly to setup. |
| Product-aware finalization | Runs an epilogue when the necessary quadrants finish, while remaining products are still scheduled. | Early SwiGLU/residual schedules already exist in N8, but were not selected for N9 and are inactive in the recent grid/large-model paths. | Retest by generation and full block; historical outcomes do not support turning it on globally. |
| Pallas input pipeline | Compiler-managed Pallas pipeline. | Already present; v6e also explicitly tested one/two/three input buffers and M/N traversal. Production experiment used two buffers. | No newly discovered asynchronous-copy or double-buffering trick to port. |
| Scoped Native VMEM versus custom-kernel allowance | Parent separates a 104/120 MiB Pallas allowance from a 48 MiB process-wide XLA setting. | Our v6e environment has no global scoped-VMEM override. Native is separately compiled with default,32,64,96,112 MiB; Pallas gets its own 112 MiB budget. | Separation is already correct. Add 48 MiB as an additional Native control in a reproduction; do not replace default Native with that one setting. |
| Input-fusion permission | Optional `allow_input_fusion` forwarded to Mosaic; permission does not guarantee fusion. | No explicit corresponding hook in our current kernels. | An experimental ablation. The promoted Qwen3 paths do not provide a non-default value, so this is not established as the source of the headline gain. |
| Exact-panel schedules / alternate formulas | Optional cubic K panels, prefetched schedules and alternative rank-7 formulas remain research paths. | No equivalent exact-panel scheduling in the current tested kernels. | Potential numerical-error control, not a verified general speed optimization. |

Deep-panel source: [pure-GEMM candidates](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3/benchmark_qwen3_v6e_pure_gemm.py#L49-L79), [site divisor search](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3/benchmark_qwen3_site_tile_tune.py#L76-L107).

Fusion sources: [epilogues](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py#L762-L875), [weight layouts](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/strassen_pallas.py#L1315-L1411), [full-block routing](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3/benchmark_qwen3_32b_full_layer_product_inference.py#L108-L175).

## What their measurements actually support

The verified v6e bare-GEMM artifact uses M,K,N = 8192,5120,51200, BF16 output, and identical selected Strassen/cubic tile 2048,1024,5120. Means: Native 5.194331 ms, Strassen 4.672467 ms, matched cubic 5.091220 ms. That is 1.111689× versus their Native and 1.089621× versus matched cubic. This exact shape is absent from our 168-shape grid, so it is not a head-to-head contradiction with one of our rows.

Our current tile rule, if applied to that shape, would offer 4096,2048,1024; 2048,1024,1024; 1024,1024,1024; 512,1024,1024; 1024,512,1024; and 256,1024,512. It cannot discover the parent's winning tile. This is a source-level coverage finding, not a measured speedup from adding the tile.

Their Qwen3-32B fused-q/k on/off artifacts demonstrate a full-block opportunity. For standard fused Strassen on v6e, the on/off times are 14.604924 versus 16.100309 ms; product-aware times are 14.551906 versus 16.229952 ms. Compare the same scheduling arm across the two runs rather than selecting different arms on each side. Separate runs are evidence for a targeted replication, not new paired confidence intervals generated by this audit.

The README calls product-aware finalization v5e-specific and consistently worse on v6e. The indexed artifacts make that categorical statement too strong: with fused q/k enabled on v6e, product-aware is 0.053018 ms faster than standard, with recorded difference CI [-0.071076,-0.034961] ms; with q/k off it is 0.129643 ms slower, CI [0.030329,0.228956] ms. On v5e, the on artifact gives a 0.657888 ms improvement. Treat scheduling benefits as context dependent.

The parent's 1.295× streamed figure means summed resident-layer compute, excluding checkpoint transfers, embeddings and final logits; it is not a fully resident end-to-end model latency. Its README also notes that the gap versus its 1.209× block result is unexplained. Fused application gains must not be called bare-MM gains.

Raw evidence: [pure GEMM](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/evidence/qwen3/strassen_qwen3_32b_v6e_pure_gemm.jsonl), [q/k on](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/evidence/qwen3/strassen_qwen3_32b_full_layer_product_inference_v6e_fusedqk_on.jsonl), [q/k off](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/evidence/qwen3/strassen_qwen3_32b_full_layer_product_inference_v6e_fusedqk_off.jsonl).

## Numerical and protocol differences matter

The parent uses JAX 0.7.2/libtpu 0.0.21.1 even on v6e; our v6e run uses JAX 0.11.2/libtpu 0.0.48. Its bare-GEMM Native baseline fixes scoped VMEM at 48 MiB; the README acknowledges no unset-default measurement. Our stronger default-plus-tuned baseline should be retained when reproducing the optimization.

The parent feeds FP32 quadrant values directly into SwiGLU/residual epilogues and rounds their final output. Our N8 and composed-model paths explicitly round projections to BF16 first, and round SiLU to BF16 before its multiply. These are different numerical boundaries. Fusion can preserve ours, but copying the parent's arithmetic verbatim requires fresh model-quality qualification and a matched fused cubic control to attribute effects.

The parent rejects non-divisible shapes instead of solving our padding/crop cases. It therefore does not provide a direct fix for the 8191/8193 boundary cliffs in our grid.

Its comment that two-level recursion yields K=128 inside a 256-wide MXU applies to BK=512. In our 38 aligned v6e shapes, S2 selected BK=2048 on 33 and BK=4096 on five; their leaf K values are 512/1024. That specific underfill argument cannot explain those S2 losses. Our observed S1 preference remains valid for the tested configurations, but this parent audit does not prove a general impossibility for S2.

Our older N8 findings already showed that complete-call packing costs can erase fusion benefits, whereas prepared/reused layouts can win. They used fixed smaller tiles and M=512/1024/2048; the parent Qwen3 experiment uses 8192 token rows. Those old results are not a reason to rule out the newer, larger v6e fused workloads.

## Suggested next experiment, not launched by this review

1. Reproduce the parent 8192,5120,51200 geometry and add a few existing aligned grid shapes. Keep the current compiler first. Compare parent's four-quadrant accumulation, our original output accumulator and our deferred S1 at identical feasible tiles. Include BK divisors through full K, with compilation feasibility checks; include both BF16-output and FP32-output experiments as separate contracts.
2. Retain default Native and independently tuned Native, adding 48 MiB; retain independently tuned and exact tile-matched cubic. Use fresh confirmation inputs and numerical references. Report kernel allowance separately from Native compiler settings.
3. Reconnect the existing packed SwiGLU/residual paths for a full block with weights prepared once; then add q/k RMSNorm+RoPE. Compare standard versus early finalization within that same block. Include an ordinary whole-block XLA baseline, not only the split composed baseline.
4. Measure projection errors, logits, KL, top-1 agreement and perplexity on real inputs. Record setup/relayout cost separately. Only then scale to resident full-model timing.

The comparison above identifies missing coverage and integration, not a promise to reproduce upstream speedups. No TPU time was consumed for this audit.
