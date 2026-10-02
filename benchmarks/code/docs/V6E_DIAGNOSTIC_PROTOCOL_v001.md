# v6e diagnostic protocol (2026-09-23)

The prior four-shape sweep does not establish architecture-optimal kernels. This diagnostic first examines a current compiler and controlled ablations on two independent shapes; only after inspecting those results will the ten historical winners be screened. No depths 3/4.

## Hardware facts and implications

v6e has one TensorCore and two MXUs per chip, with a published BF16 peak of 918 TFLOP/s and 1638 GB/s HBM bandwidth ([Google hardware specification](https://docs.cloud.google.com/tpu/docs/v6e)). The JAX hardware table rounds these to 920 TFLOP/s and 1640 GB/s and lists 128 MiB VMEM for both v5e and v6e; v5e has 197 TFLOP/s and 820 GB/s ([JAX hardware table](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)). Compute increased about 4.7x, HBM bandwidth about 2x, and VMEM capacity did not increase. These specifications alone cannot identify an observed kernel bottleneck.

v6e uses 256x256 MXU arrays rather than the preceding 128x128 arrays ([JAX scaling book](https://jax-ml.github.io/scaling-book/tpus/)). The legacy winning tile (BM,BN,BK)=(2048,1024,1024) already produces K/N leaves divisible by 256 at depths 1 and 2. Misalignment alone therefore cannot explain its regression. Depth 2 uses much smaller leaves and more combinations; test larger leaves instead of assuming alignment is sufficient.

Pallas manages asynchronous hardware scheduling through the TPU compiler and normally double-buffers input/output blocks. Explicit buffer counts and grid order can affect overlap, buffer pressure and data reuse ([JAX pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)). SparseCore and INT8/FP8 modes are not free substitutes for a dense BF16 comparison; this experiment keeps the numerical contract.

## Source audit

Original kernels already use Pallas TPU lowering, BF16 dot products with FP32 accumulation, persistent VMEM accumulation and compiler-managed HBM/VMEM pipelining. They do not dispatch a CPU multiply. There is no explicit v6e branch or explicit assignment of products to its two MXUs. Source inspection cannot establish actual dual-MXU occupancy.

At the same padded tile volume, Strassen 1 removes 12.5% of matrix multiply FLOPs and Strassen 2 removes 23.4375%. Depth 1 uses 7 leaf products, depth 2 uses 49. Original output reconstruction occurs inside every K panel. Deferred reconstruction reduces that repeated work but retains intermediate products. Full depth-two deferral needs 49/16 output-tile equivalents of FP32 scratch; hybrid outer deferral needs 7/4. Those are source-level storage counts, not measured allocator peaks.

The historical software was JAX/jaxlib 0.7.2 and libtpu 0.0.21.1, including a qualified Mosaic serialization workaround. This diagnostic pins JAX/jaxlib 0.11.2, libtpu 0.0.48 and xprof-nightly 2.24.2a20260922. Recompile and verify all controls; do not attribute cross-allocation differences solely to the compiler.

## Evidence hierarchy

1. Eight exact CPU checks cover recursion, padding, multiple K panels and output traversal. Compiled TPU integer smoke remains mandatory.
2. Controlled uninstrumented calls and ordinary device traces compare 24 arms on (8192,8192,4096) and (4096,4096,16384), in M,K,N order. Compare old versus larger tiles, reconstruction schedule, input buffers 1/2/3, and M/N traversal. Controls use identical operands and precision.
3. A separate, small LLO-instrumented probe enables custom-call tracing. Its timings do not enter speedup claims. Capability availability and analysis failures are recorded.
4. Fresh, paired ten-shape confirmations follow a frozen screen selection and retain correctness failures.

The experimental XProf LLO tool requires a recent compiler. Its instruction cycle breakdown includes modeled/interpolated durations, and periodic runtime counters are documented for TPU7x+, not v6e. Pallas roofline summaries can misleadingly report zero FLOPs because a custom-call cost model is absent ([OpenXLA custom-call profiling](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md)). We will not interpret that zero as idle MXUs, or call estimated instruction time a direct hardware counter.

## Ten shapes selected before new v6e timings

Evidence: runs/20260923T174749Z-select-v6e-ten-v001-4c1d55/artifacts/evidence.json. Select from the preserved 102-shape v5e scan only when BOTH depths have pointwise 95% paired speedup intervals entirely above one against BOTH default and tuned Native; rank by historical S2 speedup versus tuned Native, then take ten. This deliberately enriched set cannot estimate a general workload win rate. The shapes and source hashes are immutable.
