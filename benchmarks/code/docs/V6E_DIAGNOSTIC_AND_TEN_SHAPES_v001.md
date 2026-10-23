# v6e diagnostic and ten historical v5e winners

Completed 2026-09-23. Both TPU allocations were released and verified absent. Only Native and Strassen depths 1 and 2 were tested, with a classical Pallas control.

The revised implementation has repeatable v6e benefits: Strassen 1 wins on nine of ten shapes against the screen-selected tuned Native, with one inconclusive comparison. Strassen 2 wins five, loses four, and is inconclusive on one. These classifications use pointwise 95% paired intervals on fresh synchronized-call timings. Strassen 1 has lower mean latency than Strassen 2 on every tested shape and roughly half its numerical error. The earlier v5e depth-two advantage has not been replicated.

## What the diagnostic established

The runtime reports a v6e chip with two MXUs, 256 MXU columns and 128 MiB VMEM. Published specifications give 918 TFLOP/s BF16 and 1638 GB/s HBM bandwidth. Relative to v5e, matrix compute increased about 4.7 times, HBM bandwidth about twice, and VMEM capacity stayed at 128 MiB. Reducing multiplication therefore has to pay for proportionally more expensive data movement and vector work. This is an architectural explanation to investigate, not a measured attribution of every lost microsecond. Sources: [Google v6e specification](https://docs.cloud.google.com/tpu/docs/v6e), [JAX hardware table](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).

Actual compiler instruction listings show MXU instructions and input transfers in our kernels. They already use TPU lowering, BF16 products, FP32 accumulation and compiler-managed pipelining. The diagnostic does not establish simultaneous saturation of both MXUs. SparseCore or lower-precision modes are not direct replacements for this dense BF16 numerical contract.

Eight exact CPU checks and eight compiled TPU checks passed. Of 48 controlled diagnostic attempts, 44 passed; four triple-buffer variants failed compilation because this `pallas_call` lowering supports only one or two buffers. A separate low-level compiler trace succeeded and its spilled analysis files were recovered and hash-verified.

Controlled device observations on two independent shapes showed:

- Larger tiles improved the original kernels before reconstruction changes. On `(M,K,N)=(8192,8192,4096)`, original S1 fell from 0.856 to 0.708 ms and original S2 from 0.968 to 0.788 ms.
- At the same larger tile, deferred S1 reached 0.646 ms and hybrid S2 0.685 ms, versus tuned Native 0.678 ms. Neither won on the other diagnostic shape, `(4096,4096,16384)`.
- Disabling double buffering made the matched probes about 67–77% slower. Changing M/N traversal changed them by less than 0.2%.
- Native achieved an estimated 811–845 TFLOP/s, about 88–92% of published peak. This is useful FLOPs divided by observed device time, not a hardware occupancy counter.
- Historical tiles already had aligned depth-two leaf K/N dimensions. Alignment alone did not explain the gap. Larger leaves, reconstruction frequency and temporary storage mattered.
- Compiled static listings show additional BF16 combinations, FP32 additions and loads/stores, especially at depth two. Static instruction ratios are not dynamic cycle percentages or measured bandwidth utilization.

The environment was JAX/jaxlib 0.11.2 and libtpu 0.0.48, with xprof-nightly 2.24.2a20260922. The older compiler was not A/B tested on the same allocation; differences across historical runs cannot be assigned solely to the compiler or hardware. Instrumented low-level timings never enter the speedup results.

## What was changed and tested

We retained double buffering and tested larger and differently proportioned tiles. S1 can reconstruct each K panel or retain its seven products and reconstruct once after K. S2's hybrid retains seven outer products while reconstructing the inner level each K panel; its explicit FP32 scratch is 7/4 output tiles instead of 49/16 for fully deferred depth two. These are source-level storage counts, not measured allocator peaks.

The ten-shape screen contained 21 candidates per shape: five Native compiler-memory settings including default, two classical Pallas tiles, four original Strassen candidates, six revised S1 candidates, and four revised S2 candidates. All 210 passed the numerical gates. Finalists were frozen from screening before fresh confirmation: 204 cases passed, representing 68 selected shape/arm combinations on three fresh inputs, with 30 paired randomized timing rounds per input (6,120 timing samples).

Most selected S1 down projections used `(BM,BN,BK)=(2048,2048,2048)` with deferred reconstruction. The three wide projections selected panel reconstruction with `(2048,1024,4096)`: BK covers the full K, so reconstruction occurs only once. Three larger down projections selected S2 hybrid `(4096,1024,2048)`; the others selected hybrid `(2048,2048,2048)`.

This is a bounded search, not proof of optimality. Classical controls were included but were not matched to every selected new tile; gains cannot all be attributed specifically to Strassen algebra independently of tiling and scheduling.

## Ten-shape confirmation

Shapes were frozen before new v6e timings. Each had historical pointwise confidence intervals entirely above one for BOTH depths against BOTH default and tuned Native. They were ranked by historical S2 speedup and the first ten selected. This intentionally favorable set does not estimate a general workload win rate.

Dimensions below are M × K × N for `(M,K) @ (K,N)`. Speedup is tuned Native latency divided by Strassen latency; greater than one is faster. Native settings were selected during screening, never reselected on confirmation.

| Shape | Native ms | S1 ms | S2 ms | S1 speedup [95% CI] | S2 speedup [95% CI] |
|---|---:|---:|---:|---:|---:|
| 2048 × 16384 × 2048 | 0.406 | 0.404 | 0.417 | 1.004 [0.997, 1.012] | 0.974 [0.962, 0.985] |
| 2048 × 12288 × 4096 | 0.507 | 0.477 | 0.495 | 1.064 [1.054, 1.072] | 1.023 [1.016, 1.031] |
| 16384 × 14336 × 4096 | 2.532 | 2.372 | 2.460 | 1.068 [1.066, 1.069] | 1.029 [1.027, 1.031] |
| 16384 × 12288 × 4096 | 2.265 | 2.097 | 2.172 | 1.080 [1.078, 1.082] | 1.043 [1.041, 1.045] |
| 2048 × 14336 × 4096 | 0.567 | 0.544 | 0.567 | 1.042 [1.032, 1.057] | 1.000 [0.989, 1.017] |
| 8192 × 14336 × 4096 | 1.415 | 1.331 | 1.371 | 1.062 [1.056, 1.068] | 1.032 [1.027, 1.038] |
| 8192 × 12288 × 4096 | 1.272 | 1.207 | 1.249 | 1.053 [1.049, 1.058] | 1.018 [1.013, 1.024] |
| 8192 × 4096 × 24576 | 2.290 | 2.020 | 2.471 | 1.134 [1.132, 1.136] | 0.927 [0.925, 0.928] |
| 16384 × 4096 × 24576 | 4.276 | 3.768 | 4.658 | 1.135 [1.132, 1.140] | 0.918 [0.916, 0.922] |
| 16384 × 4096 × 28672 | 4.733 | 4.320 | 5.385 | 1.096 [1.094, 1.096] | 0.879 [0.878, 0.880] |

S1's geometric mean speedup is 1.073×; S2's is 0.983×. The best S1 latency reduction is 11.89%; the best S2 reduction is 4.09%. The largest S2 regression is 13.78% more time. Default Native comparisons are retained in the machine-readable summary; the table uses the stronger screen-selected Native control. For example, the largest Mistral wide projection shows S1 at 1.453× versus default but only 1.096× versus tuned Native.

Intervals are hierarchical paired bootstraps over three inputs and 30 rounds each, conditional on screening decisions, without multiplicity adjustment. Historical and current ratios use their own Native baselines on different hardware/compiler cohorts. The figure must not be interpreted as a controlled hardware-only experiment.

![Historical and current speedups](../runs/20260923T184827Z-v6e-ten-chart-v001-16e718/artifacts/comparison.png)

## Device timing qualification

Headline latency covers a synchronized complete device call, including any padding/crop and host dispatch/wait, excluding compilation and host transfers. Separate ordinary traces contain 20 device-module observations per selected arm on the first fresh input. All ten traces passed coverage and stable executable-identity checks. Their measurements are not pooled into headline intervals.

All nine clear S1 call-time wins also show lower device mean time. Four larger down projections show the S2 improvement in both measurements. The small Qwen down projection `(2048,12288,4096)` differs: S2 wins the synchronized-call comparison by 1.023×, but the separate device trace gives Native 0.252218 ms and S2 0.262547 ms, or 0.961×. Treat that case as a call-time result without confirmed device-compute benefit. Different profiling conditions, sample sets and call overhead prevent subtracting the two measurements to identify a causal overhead component.

The first shape is also instructive: S1 has a small device improvement (0.177442 → 0.171142 ms), but its roughly 0.406 ms synchronized call shows no clear gain. Millisecond-scale full-model benefit cannot be inferred from isolated submillisecond device savings alone.

## Numerical cost

Inputs are fresh Gaussian A/sqrt(K) and Gaussian B, identically quantized to BF16. Products accumulate into FP32; Strassen combinations use BF16. References accumulate the exact quantized operands in host FP64 across all K for 128 sampled rows × 128 sampled columns. Full outputs are checked for finiteness. Values below are the worst across the selected ten shapes and three fresh inputs; each maximum can come from a different case.

| Metric | S1 | S2 |
|---|---:|---:|
| Relative L2 error | 0.4449% | 1.0068% |
| Maximum absolute error, sampled | 0.024970 | 0.065142 |
| RMSE | 0.004439 | 0.009928 |
| Mean absolute error | 0.003307 | 0.007292 |
| 99th percentile absolute error | 0.013463 | 0.031544 |

Native's worst relative L2 error was about 1.66e-7 as a fraction. The Strassen error is substantive and mainly reflects its extra low-precision combinations. All cases passed the predeclared 2% relative-L2 and scaled maximum-error gates, but passing those gates does not establish acceptable prediction quality. These are synthetic matrices at LLM-related dimensions, not real checkpoint/activation inference; no perplexity or prediction-accuracy claim follows from this experiment.

## Optimization priorities supported by these results

1. Retain the measured S1 tile/schedule choices and Native fallback. S1 is the stronger candidate for subsequent resident LLM validation on this set: lower mean latency and lower error than S2.
2. Focus further S2 work on operand-combination reuse, reconstruction traffic and live FP32 storage while keeping large aligned leaf dots. More recursion or smaller aligned leaves alone do not help these results.
3. Test any manual deeper pipeline as an isolated experiment with complete-call timing and VMEM accounting; triple buffering through the tested lowering is currently unsupported. Preserve the proven double-buffer baseline.
4. Before assigning gains specifically to Strassen algebra, add matched classical controls for the selected full-K/aspect-ratio tiles. For fixed-weight preprocessing, include preparation cost and the reuse count needed to amortize it.

## Preserved evidence

- [Architecture protocol and sources](V6E_DIAGNOSTIC_PROTOCOL_v001.md)
- [Controlled diagnostic results and low-level evidence](../runs/20260923T181010Z-v6e-diagnostic-summary-v001-e874b9/artifacts/RESULTS.md)
- [Historical selection evidence](../runs/20260923T174749Z-select-v6e-ten-v001-4c1d55/artifacts/evidence.json)
- [Fresh confirmation table and error decisions](../runs/20260923T184750Z-v6e-ten-summary-v001-03f334/artifacts/RESULTS.md)
- [Machine-readable summary, both Native comparisons and input hashes](../runs/20260923T184750Z-v6e-ten-summary-v001-03f334/artifacts/summary.json)
- [All original/revised/classical family results](../runs/20260923-v6e-ten-v001/operations/report/artifacts/RESULTS.md)
- [Separate device traces and timing tables](../runs/20260923T184803Z-v6e-ten-device-profiles-v001-e3afda/artifacts/PROFILES.md)
- [Diagnostic release receipt](../runs/20260923-v6e-diagnostic-v001/operations/release/execution.log)
- [Ten-shape release receipt](../runs/20260923-v6e-ten-v001/operations/release/execution.log)

Executed sources, failed attempts, raw timing samples, compiler artifacts and correctness measurements remain archived under their original versions.
