# v6e kernel optimization results

The optimization recovered a reproducible Strassen 1 advantage on two previously
winning v5e shapes. On the independent follow-up v6e allocation, Qwen down
(M,K,N)=(2048,12288,4096) improved 2.65% over tuned Native, and Mistral down
(16384,14336,4096) improved 5.53%. Their first-round gains were 2.21% and 5.44%.
Every one of these four paired 95% intervals excludes no improvement.

Strassen 2 improved after changing reconstruction and accumulator lifetimes,
but the best tested hybrid still took 3.28% and 8.57% longer than tuned Native
on those two shapes. The wide Qwen gate/up shape still favored Native; the
4096-cubed Strassen 1 result did not establish a win. This remains a bounded
optimization, not a universal v6e speedup or a proven optimum.

Across both rounds, 166 screening attempts produced 134 successful measurements
and 32 recorded VMEM compile failures. All 126 fresh-input confirmation cases
passed the numerical gates. Worst sampled relative L2 was below 0.45% for
Strassen 1 and below 1.0% for the tested Strassen 2 finalists; prediction quality
was not evaluated in this isolated-MM experiment.

[First-round device profiles](../runs/20260923T060433Z-v6e-selected-profile-analysis-v002-ca1141/artifacts/PROFILES.md)
and [follow-up device profiles](../runs/20260923T061257Z-v6e-hybrid-profile-analysis-v002-052a9e/artifacts/PROFILES.md)
are preserved separately from ordinary timing samples. Both allocations were
released and verified absent.


2026-09-23. Only Strassen depths 1 and 2 were tested. The four initial shapes had confirmed v5e wins over both default and tuned Native; shape selection preceded the new v6e measurements.

The new kernels either reconstruct in compiler-managed SSA values per K panel, defer all reconstruction until K accumulation finishes, or (depth two follow-up) defer only the outer level. Inputs and every operand combination remain BF16; products, accumulation and outputs remain FP32. Deferral changes FP32 addition order.

The first round screened 136 configurations. The adaptive follow-up screened 30 configurations on two long-K shapes, motivated by the first round's VMEM failures. Each selected method was confirmed with 30 paired rounds on each of three fresh Gaussian inputs. Controls were remeasured on each allocation; no timing samples are pooled across allocations.

Latency below covers a synchronized complete device call, including padding and cropping, excluding compilation and transfers. These are synthetic MM measurements on the historical winning shapes, not a new real-weight or whole-LLM evaluation.

## First round: four v5e-winning shapes

| Shape (M,K,N) | Native default ms | Tuned Native ms | Optimized S1 ms (change) | Optimized S2 ms (change) | Hybrid S2 ms (change) |
|---|---:|---:|---:|---:|---:|
| (16384, 14336, 4096) | 2.929953 | 2.552164 | 2.413359 (-5.44%) | 2.876560 (+12.71%) | — |
| (2048, 12288, 4096) | 0.581245 | 0.527286 | 0.515636 (-2.21%) | 0.565940 (+7.33%) | — |
| (8192, 4096, 24576) | 2.290116 | 2.166588 | 2.205137 (+1.78%) | 2.561314 (+18.22%) | — |
| (4096, 4096, 4096) | 0.443669 | 0.440694 | 0.446174 (+1.24%) | 0.473871 (+7.53%) | — |

Negative changes mean lower latency than tuned Native. Pointwise paired 95% intervals:

- mistral_7b_v03_down_m16384, optimized1: -5.65% to -5.24%; -9.32% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.437%.
- mistral_7b_v03_down_m16384, optimized2: +12.54% to +12.89%; -2.68% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.992%.
- qwen_3_8b_down_m2048, optimized1: -3.53% to -0.82%; -4.94% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.440%.
- qwen_3_8b_down_m2048, optimized2: +6.38% to +8.24%; -4.61% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.984%.
- qwen_3_8b_gate_up_concat_m8192, optimized1: +0.71% to +2.52%; -1.35% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.441%.
- qwen_3_8b_gate_up_concat_m8192, optimized2: +16.82% to +19.21%; -2.04% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.989%.
- shape_m4096_k4096_n4096, optimized1: -1.25% to +4.85%; -2.06% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.447%.
- shape_m4096_k4096_n4096, optimized2: +4.83% to +10.26%; -0.34% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.983%.

[Full timings, errors, selections and failures](</Users/bagheera/Documents/ChatGPT/Faster Strassen TPU/Strassen_MM_Focus/runs/20260923-v6e-opt-v001/operations/report/artifacts/RESULTS.md>)

## Follow-up: two long-K shapes on another allocation

| Shape (M,K,N) | Native default ms | Tuned Native ms | Optimized S1 ms (change) | Optimized S2 ms (change) | Hybrid S2 ms (change) |
|---|---:|---:|---:|---:|---:|
| (16384, 14336, 4096) | 2.913443 | 2.533820 | 2.393772 (-5.53%) | 2.855193 (+12.68%) | 2.750970 (+8.57%) |
| (2048, 12288, 4096) | 0.531176 | 0.474843 | 0.462244 (-2.65%) | 0.519681 (+9.44%) | 0.490420 (+3.28%) |

Negative changes mean lower latency than tuned Native. Pointwise paired 95% intervals:

- mistral_7b_v03_down_m16384, optimized1: -5.74% to -5.33%; -10.00% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.437%.
- mistral_7b_v03_down_m16384, optimized2: +12.43% to +12.91%; -2.70% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.982%.
- mistral_7b_v03_down_m16384, hybrid2: +8.31% to +8.83%; -6.26% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.991%.
- qwen_3_8b_down_m2048, optimized1: -3.80% to -1.55%; -5.62% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.444%.
- qwen_3_8b_down_m2048, optimized2: +7.25% to +13.30%; -3.65% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.998%.
- qwen_3_8b_down_m2048, hybrid2: +2.52% to +4.14%; -9.08% latency change versus its independently measured original kernel. Error gate: True; worst sampled relative L2 0.993%.

[Full timings, errors, selections and failures](</Users/bagheera/Documents/ChatGPT/Faster Strassen TPU/Strassen_MM_Focus/runs/20260923-v6e-hybrid-v001/operations/report/artifacts/RESULTS.md>)

## Validation and interpretation

The first implementation passed 16 CPU exact-integer checks and 10 compiled TPU smoke variants. The hybrid passed four CPU algebra checks, an offline per-shape-control coverage check and a seven-variant compiled TPU smoke. CPU interpretation validates algebra, padding and multiple K panels; actual compiled TPU tests qualify device arithmetic.

Numerical eligibility is unchanged: relative L2 at most 2%, finite output, and the existing maximum-error gate. Floating references use exact BF16-quantized operands and FP64 accumulation over all K for 128 sampled rows by 128 sampled columns. Maximum error is sampled; output finiteness is checked in full. These gates do not establish prediction accuracy.

Device profiles are separate from ordinary latency samples. They identify actual TPU module times but do not expose direct MXU utilization, memory bandwidth, stalls or intra-kernel overlap counters. The initial profile parser assumed aligned host/device clocks and rejected the traces; version 002 validates complete serialized invocation order and stable module identities instead. Both analyses are preserved.

The new source is kernels_v6e_v001.py (per-panel and deferred reconstruction) and kernels_v6e_v002.py (hybrid depth two). No automatic LLM dispatch policy has been changed. All runtime allocations were released after verified artifact retrieval.

