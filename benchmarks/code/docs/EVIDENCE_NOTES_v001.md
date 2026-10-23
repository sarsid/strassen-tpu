# Evidence interpretation notes — 2026-09-19

The core `benchmark_v001` uses `eligible_for_speedup_claim` as a numerical-pass flag in all phases. Core N4 has no timing, so that flag alone cannot support a speedup claim. A timing claim requires recorded timing samples and an appropriate valid comparison. The N4 actual-weight supplement explicitly sets that field false and records numerical eligibility separately. The audited report treats both N4 runs as accuracy-only evidence.

The fixed numerical gate is finite output, relative L2 error at most 0.02, and maximum checked absolute error at most 0.001 + 0.05 times the maximum checked reference magnitude. It is a synthetic-study eligibility rule, not a guarantee of model quality. Large outputs use a saved sample of rows and columns with all K terms; whole-output finiteness is checked separately.

N2's 60 OOM records are compilation failures under the configured 48 MiB VMEM budget. They include timing scopes separately and are not 60 distinct shapes. The budget is an experiment setting, not a claim about the TPU's total physical memory.

The verified actual-weight acquisition is `runs/20260919T051132Z-acquire-qwen-real-weights-v001-234bb3`, committed as `9bed8bd`. Its source is the public Qwen/Qwen3-0.6B checkpoint at revision `c1899de289a04d12100db370d81485cdf75e47ca`; two layer-0 tensors were downloaded as exact bounded byte ranges. Their original BF16 bit patterns are retained and independently checked by the loader tests. No model inference was performed.
