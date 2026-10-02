# v6e Native versus one to four tile-recursion levels

The bounded sweep completed on 2026-09-23. Tuned Native was faster on all three
tested shapes. Recursion depth refers to each BM/BN/BK tile panel, with 7, 49,
343 or 2401 leaf products, respectively.

All timings below are milliseconds for A[M,K] × B[K,N], including device padding
and cropping, excluding host transfers and compilation. Inputs were reproducible
Gaussian BF16 matrices; accumulation and output were FP32. These are isolated MM
measurements. Each confirmed row contains 90 measurements: 30 paired rounds on
each of three fresh input seeds, on one v6e allocation.

| Method | M=K=N=8192 | M=K=N=16384 | M=8192, K=8192, N=28672 |
|---|---:|---:|---:|
| Tuned Native | 1.582889 | 10.524688 | 4.855742 |
| Strassen 1 | 1.640126 | 11.093475 | 5.114125 |
| Strassen 2 | 1.798395 | 12.452538 | 5.680628 |
| Strassen 3 | 4.785406 | 49.513281 | 16.104377 |
| Strassen 4 | 15.070172 | 118.510898 | 52.084643 |

Strassen 1 increased latency by 3.62%, 5.40% and 5.32%, respectively. Strassen 2
increased it by 13.61%, 18.32% and 16.99%. Paired confidence intervals excluded a
speedup in every case, conditional on this allocation and frozen selections.

The worst sampled relative L2 errors over the three confirmation inputs were:

| Depth | 8192 square | 16384 square | Wide rectangle |
|---|---:|---:|---:|
| 1 | 0.442% | 0.445% | 0.441% |
| 2 | 0.995% | 0.998% | 0.984% |
| 3 | 2.023% | 2.013% | 2.013% |
| 4 | 4.073% | 4.121% | 4.047% |

Depths 3 and 4 failed the unchanged 2% relative-error gate during confirmation.
Their outputs were finite and they had passed exact integer algebra tests on the
compiled TPU. The failure is a floating-point accuracy criterion, distinct from
an incorrect recursion formula. The reference used exact quantized operands,
FP64 accumulation over all K, and 128 sampled rows × 128 sampled columns. Full
output finiteness was checked; maximum-error numbers cover the sample only.

The screen attempted three tiles per depth and four Native compiler settings per
shape, totaling 48 attempts. Nine configurations failed compilation because of
VMEM capacity. Four measured configurations failed the numerical gate. Native's
96 MiB compiler setting won on all shapes; custom kernels used a 112 MiB allowance.

Depths 1–2 used the prior kernels. New depths 3–4 used scalar loops for the outer
recursion with the inner two levels statically expanded, reusing FP32 VMEM
scratch. These measurements characterize that implementation and this small tile
search. They do not establish an optimum for deeper Strassen. Larger deep tiles
required about 160–209 MiB after register spilling against 128 MiB physical VMEM.
The compact implementation passed exact CPU and compiled TPU algebra checks.
The earlier fully expanded implementation was superseded after expensive CPU
compilation; its interrupted validation and the CPU rounding diagnostics remain
archived. It was not compared for TPU performance.

Selection first preferred candidates passing the frozen screening error gate.
For 16384-square depth 3, the faster 2048³ tile (36.307 ms screening) failed that
gate, so the 1024×2048×2048 tile was selected. That selected tile later also
crossed the gate on fresh input. All candidates, including the faster failed
candidate, remain in the raw screen. Depth 4 had no passing screening candidate;
its fastest finite result was explicitly retained as an ineligible diagnostic.

All result retrieval and reporting completed, and allocation release was verified.

- [Frozen-choice results and confidence intervals](../runs/20260923-v6e-depth4-v001/operations/report/artifacts/RESULTS.md)
- [Full machine-readable timings, numerical metrics and compiler failures](../runs/20260923-v6e-depth4-v001/operations/report/artifacts/results.json)
- [Frozen experiment plan](../runs/20260923-v6e-depth4-v001/plan.json)
- [Verified runtime release](../runs/20260923-v6e-depth4-v001/operations/release/execution.log)
