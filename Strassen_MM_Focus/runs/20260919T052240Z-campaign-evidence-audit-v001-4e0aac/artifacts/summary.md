# N1–N4 v5e v1 experiment summary

Evidence audit: **PASS**.

One logical allocation: `tpu-v5e1-s-kkb-usw1c1-21as5yio8er1q`; device: `TPU v5 lite`.

Case counts include both timing scopes in N1–N3. N4 contains correctness cases only.

| Phase | Cases | Valid | Numerical failures | Errors | Skips | Incomplete |
|---|---:|---:|---:|---:|---:|---:|
| N1 | 352 | 352 | 0 | 0 | 0 | 0 |
| N2 | 432 | 372 | 0 | 60 | 0 | 0 |
| N3 | 96 | 96 | 0 | 0 | 0 | 0 |
| N4 | 480 | 456 | 24 | 0 | 0 | 0 |

## N1: fixed-tile complete-call comparisons

A win requires the paired speedup CI95 to lie above 1 and both numerical gates to pass. A loss lies below 1; a tie is inconclusive.

| Strassen versus | Wins | Ties/inconclusive | Losses | Ineligible/other |
|---|---:|---:|---:|---:|
| cubic_basic | 0 | 25 | 19 | 0 |
| cubic_quadrant | 27 | 17 | 0 | 0 |
| native_xla | 0 | 3 | 41 | 0 |

## N2: bounded tile screening

Best observed complete-call mean for each arm and shape. These selected minima are **screening observations, not confirmed speedups**.

| Shape | Algorithm arm | Eligible / attempted tiles | Best observed tile (BM,BN,BK) | Mean ms |
|---|---|---:|---|---:|
| gemma9_gate_up_m512 | cubic_basic | 5 / 6 | 512,1024,512 | 0.9237 |
| gemma9_gate_up_m512 | cubic_quadrant | 5 / 6 | 512,1024,512 | 0.9938 |
| gemma9_gate_up_m512 | strassen_basic | 5 / 6 | 512,1024,512 | 0.9162 |
| mistral7_gate_up_m512 | cubic_basic | 5 / 6 | 512,1024,512 | 1.036 |
| mistral7_gate_up_m512 | cubic_quadrant | 5 / 6 | 512,1024,512 | 1.115 |
| mistral7_gate_up_m512 | strassen_basic | 5 / 6 | 512,1024,512 | 1.035 |
| qwen32_gate_up_m512 | cubic_basic | 5 / 6 | 512,1024,512 | 1.908 |
| qwen32_gate_up_m512 | cubic_quadrant | 5 / 6 | 512,1024,512 | 2.125 |
| qwen32_gate_up_m512 | strassen_basic | 5 / 6 | 512,1024,512 | 1.912 |
| qwen32_gate_up_m8192 | cubic_basic | 5 / 6 | 1024,1024,512 | 24.95 |
| qwen32_gate_up_m8192 | cubic_quadrant | 5 / 6 | 1024,1024,512 | 27.19 |
| qwen32_gate_up_m8192 | strassen_basic | 5 / 6 | 1024,1024,512 | 24.06 |
| small_m_8 | cubic_basic | 6 / 6 | 512,1024,512 | 0.265 |
| small_m_8 | cubic_quadrant | 6 / 6 | 512,1024,512 | 0.2641 |
| small_m_8 | strassen_basic | 6 / 6 | 512,1024,512 | 0.2574 |
| square_2048 | cubic_basic | 5 / 6 | 1024,512,512 | 0.3791 |
| square_2048 | cubic_quadrant | 5 / 6 | 1024,512,512 | 0.3828 |
| square_2048 | strassen_basic | 5 / 6 | 1024,512,512 | 0.3796 |
| square_4096 | cubic_basic | 5 / 6 | 1024,1024,512 | 1.109 |
| square_4096 | cubic_quadrant | 5 / 6 | 1024,1024,512 | 1.163 |
| square_4096 | strassen_basic | 5 / 6 | 1024,1024,512 | 1.078 |
| square_512 | cubic_basic | 6 / 6 | 512,512,256 | 0.2027 |
| square_512 | cubic_quadrant | 6 / 6 | 512,512,256 | 0.1915 |
| square_512 | strassen_basic | 6 / 6 | 512,512,256 | 0.2058 |
| tail_513 | cubic_basic | 6 / 6 | 512,1024,512 | 0.2411 |
| tail_513 | cubic_quadrant | 6 / 6 | 512,1024,512 | 0.2374 |
| tail_513 | strassen_basic | 6 / 6 | 1024,1024,512 | 0.2588 |
| vary_k_8192 | cubic_basic | 5 / 6 | 1024,512,512 | 0.6646 |
| vary_k_8192 | cubic_quadrant | 5 / 6 | 1024,1024,512 | 0.7024 |
| vary_k_8192 | strassen_basic | 5 / 6 | 1024,1024,512 | 0.6537 |
| vary_m_8192 | cubic_basic | 5 / 6 | 1024,1024,512 | 0.6979 |
| vary_m_8192 | cubic_quadrant | 5 / 6 | 1024,1024,512 | 0.7338 |
| vary_m_8192 | strassen_basic | 5 / 6 | 1024,1024,512 | 0.6796 |
| vary_n_8192 | cubic_basic | 5 / 6 | 1024,1024,512 | 0.6839 |
| vary_n_8192 | cubic_quadrant | 5 / 6 | 1024,1024,512 | 0.7131 |
| vary_n_8192 | strassen_basic | 5 / 6 | 1024,1024,512 | 0.6704 |

## N3: fixed-tile ablations versus plain

| Algorithm | Variant | Wins | Ties/inconclusive | Losses | Ineligible/other |
|---|---|---:|---:|---:|---:|
| cubic_quadrant | interleaved | 0 | 6 | 0 | 0 |
| cubic_quadrant | interleaved_output_accumulator | 4 | 2 | 0 | 0 |
| cubic_quadrant | output_accumulator | 3 | 3 | 0 | 0 |
| strassen | interleaved | 3 | 3 | 0 | 0 |
| strassen | interleaved_output_accumulator | 3 | 3 | 0 | 0 |
| strassen | output_accumulator | 3 | 3 | 0 | 0 |

## N4: numerical stress cases

Every arm/profile contains eight shapes and three seeds. Failures remain in the table; near-cancellation cases use the same frozen gate.

| Arm | Profile | Passed | Numerical failures | Errors/skips | Median relative L2 | Maximum checked relative L2 |
|---|---|---:|---:|---:|---:|---:|
| cubic_basic | cancellation | 24 | 0 | 0 | 2.471e-05 | 2.848e-05 |
| cubic_basic | gaussian | 24 | 0 | 0 | 1.565e-07 | 1.929e-07 |
| cubic_basic | log_scale | 24 | 0 | 0 | 2.941e-07 | 3.39e-07 |
| cubic_basic | outliers | 24 | 0 | 0 | 3.087e-07 | 4.353e-07 |
| cubic_basic | uniform | 24 | 0 | 0 | 1.419e-07 | 1.791e-07 |
| cubic_quadrant | cancellation | 24 | 0 | 0 | 2.549e-05 | 3.112e-05 |
| cubic_quadrant | gaussian | 24 | 0 | 0 | 1.654e-07 | 2.192e-07 |
| cubic_quadrant | log_scale | 24 | 0 | 0 | 3.028e-07 | 3.392e-07 |
| cubic_quadrant | outliers | 24 | 0 | 0 | 3.11e-07 | 4.952e-07 |
| cubic_quadrant | uniform | 24 | 0 | 0 | 1.52e-07 | 2.093e-07 |
| native_xla | cancellation | 24 | 0 | 0 | 2.549e-05 | 3.112e-05 |
| native_xla | gaussian | 24 | 0 | 0 | 1.654e-07 | 2.192e-07 |
| native_xla | log_scale | 24 | 0 | 0 | 3.028e-07 | 3.392e-07 |
| native_xla | outliers | 24 | 0 | 0 | 3.11e-07 | 4.952e-07 |
| native_xla | uniform | 24 | 0 | 0 | 1.52e-07 | 2.093e-07 |
| strassen_basic | cancellation | 0 | 24 | 0 | 0.5269 | 0.54 |
| strassen_basic | gaussian | 24 | 0 | 0 | 0.004392 | 0.004473 |
| strassen_basic | log_scale | 24 | 0 | 0 | 0.003656 | 0.004491 |
| strassen_basic | outliers | 24 | 0 | 0 | 0.003896 | 0.006823 |
| strassen_basic | uniform | 24 | 0 | 0 | 0.004516 | 0.004621 |

## N4 actual-weight supplement

Separately archived 48-case study: actual BF16 Qwen3-0.6B layer-0 q_proj/down_proj weights, synthetic Gaussian activations, M=512/2048, three seeds and four plain algorithms. No real-activation or model-quality claim.

Passed: **48**; numerical failures: **0**; errors: **0**; skips: **0**.

Weight provenance, planned cases and numerical details are retained in the JSON report. This supplement does not change the core N4 count of 480.


## Scope and limits

- These are current fixed-tile elementary kernels and bounded tile screens, not the strongest independently tuned cubic or Strassen baselines.
- N2 minima are best observed screening measurements; they are not fresh confirmations, globally optimal tiles, or a deployed selection rule.
- There are no held-out generalization, v6e replication, model-quality, or end-to-end LLM performance claims in N1--N4.
- Core v1 LLM-associated shapes use synthetic BF16 matrices. The optional separately reported real-weight supplement uses actual weight tensors with synthetic activations; it does not test real activation distributions or model quality.
- Complete-call timing includes device preparation/padding and output cropping; it excludes compilation and host transfers. Prepared-kernel results remain separate in the evidence.
- A confidence interval containing one is inconclusive at this measurement precision; it does not establish equal performance. Intervals are per-comparison, without familywise multiplicity correction.
- Sampled-reference errors cover a saved row-column cross-product using all K, not a certified maximum over the complete output. Whole-output finite checks are separate.
- Machine identity describes one logical allocation/runtime and exposed device attributes, not an independently verified permanent physical-chip serial number.

Exact selected runs, configuration hashes, per-shape comparisons, all screen candidates, and numerical cases are in `summary.json`. Source measurements remain in the supplied immutable run artifacts.
