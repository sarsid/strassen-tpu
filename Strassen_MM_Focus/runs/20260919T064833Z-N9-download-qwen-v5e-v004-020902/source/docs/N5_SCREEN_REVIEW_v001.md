# Independent N5 screening review

Reviewed the completed immutable run `20260919T055706Z-N5-screen-v5e-v004-25842c` using its archived configuration, journal, selection artifact, source manifests and Git records. This is a read-only evidence review; no kernels, benchmarks or numerical reference calculations were rerun. No executed code or results were changed.

**Conclusion:** the recorded selections follow the preregistered eligibility and ranking rules. No issue in this screen blocks interpretation of the separately executed confirmation, provided that confirmation retains the same cohort, frozen choices and precision contract. Screening latencies themselves are descriptive and cannot establish confirmed speedups.

## Evidence and coverage

- Archive commit: `ad22271727d0c4d67ca362610dd742ef6bcb2158`.
- Source commit: `deb7deb00260454df96b7a1fa17a8ec1d3e0529f`.
- Selection artifact SHA256: `e8243a48d32ce6326173d2a3706a21a8129f29b06159a34c1223f9d172cf9943`.
- Complete journal SHA256: `a31f4c49b0bccc54c8b77ff450a7e6cad99cef0996cf6e3a9e43a1afe867c8ce`.
- Campaign SHA256: `535808e9832f4c800c653f5b51eb8707828b2e27c94e705e5bc26c2543da2a5a`.
- Allocation: `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`; device `TPU v5 lite`; JAX/JAXlib `0.7.2`, libtpu `0.0.21.1`. The journal reports an identity match with no unknown fields. This is the new N5–N9 cohort; N1–N4 timings are not pooled with it.

All 320 planned candidate-pair groups completed across the 16 registered shapes. Every shape has exactly 20 attempted cubic candidates, 20 attempted Strassen candidates, and one native candidate, each represented in both timing scopes. That gives 656 candidate–shape attempts and 1,312 result rows: **1,234 `ok`, 78 `oom`**, with no omitted candidate, duplicate result, numerical failure or other error status.

All 1,234 successful rows have exactly seven positive finite raw timings, with rounds 0–6 present exactly once; 8,638 samples were retained. Their arithmetic means match the reported means. Both independent candidate families use the same attempt budget, but different preregistered candidate grids: cubic has two variants across ten tiles; Strassen has four variants across five tiles. This is equal attempted-candidate budgeting, not equal compilation time or equal numbers of feasible candidates.

## Selection eligibility

Independently reconstructed each candidate's two-scope eligibility from the journal: both `call` and `prepared_kernel` must have status `ok`, pass the unchanged numerical gate, and contain seven samples. Ranked eligible candidates by `(complete-call mean, candidate_id)`. All 32 selected custom candidates and all 16 native entries agree with the artifact, including their source record IDs, algorithm, variant, tile, reference scope and numerical error. Native uses candidate ID `native` and public arm ID `native_xla`.

The gate remains finite output, relative L2 error at most 0.02, and maximum absolute error at most `0.001 + 0.05 × max_abs_reference`. The recorded gate flags agree with these metric-level checks. This review did not independently regenerate inputs or recompute matrix products. Reference inputs were the quantized BF16 operands; the archived implementation uses NumPy FP32 references, with all K terms and either full output or a recorded sampled row–column cross product.

For `square_512`, all 20 candidates in each family qualify. For every other shape, 19 cubic and 18 Strassen candidates qualify. Across all shapes, that is 305 eligible cubic attempts, 290 eligible Strassen attempts and 16 native attempts. The 12 additional successful complete-call rows whose prepared scope failed are correctly excluded from selection.

Selected Strassen relative L2 errors range from 0.002796 to 0.004478. Selected cubic errors range from approximately 1.23e-7 to 1.83e-7. Passing this Gaussian-input screen does not resolve the known N4 cancellation failures, certify arbitrary inputs or establish model quality.

## Selected tiles and variants

Matrix shapes below are ordered **(M,K,N)**. Tiles are ordered **(BM,BN,BK)**. Variant abbreviations: **P** = `plain`, **O** = `output_accumulator`, **I** = `interleaved`, **IO** = `interleaved_output_accumulator`. Times are screening complete-call arithmetic means in milliseconds, rounded here to six decimal places; the original artifact retains full precision.

| Shape ID | Matrix (M,K,N) | Cubic tile; variant | Strassen tile; variant | Native ms | Cubic ms | Strassen ms |
|---|---|---|---|---:|---:|---:|
| square_512 | 512,512,512 | 2048,1024,512; O | 512,1024,512; O | 0.240598 | 0.230059 | 0.214113 |
| square_2048 | 2048,2048,2048 | 1024,1024,1024; O | 512,1024,512; I | 0.363184 | 0.334396 | 0.340623 |
| square_4096 | 4096,4096,4096 | 1024,1024,1024; P | 2048,2048,512; IO | 1.025654 | 1.020277 | 0.975251 |
| square_8192 | 8192,8192,8192 | 1024,1024,1024; O | 2048,2048,512; IO | 6.381981 | 6.116295 | 5.594164 |
| vary_m_8192 | 8192,2048,2048 | 1024,2048,512; O | 1024,1024,1024; IO | 0.689211 | 0.640569 | 0.630544 |
| vary_k_8192 | 2048,8192,2048 | 1024,1024,1024; P | 2048,1024,512; I | 0.636330 | 0.592243 | 0.571380 |
| vary_n_8192 | 2048,2048,8192 | 1024,1024,1024; O | 1024,1024,1024; IO | 0.667346 | 0.632471 | 0.613929 |
| mkn_512_8192_2048 | 512,8192,2048 | 512,1024,1024; O | 512,1024,512; P | 0.385492 | 0.351569 | 0.349784 |
| small_m_8 | 8,2048,2048 | 128,512,512; P | 512,1024,512; O | 0.270407 | 0.253201 | 0.281291 |
| small_m_64 | 64,2048,2048 | 512,1024,512; P | 512,1024,512; P | 0.277069 | 0.266617 | 0.266467 |
| tail_513 | 513,513,513 | 1024,1024,512; P | 512,1024,512; P | 0.272216 | 0.241860 | 0.249013 |
| tail_wide | 1023,2047,4095 | 512,1024,512; O | 1024,1024,1024; O | 0.361140 | 0.361663 | 0.356050 |
| qwen32_gate_up_m512 | 512,5120,51200 | 512,1024,1024; O | 512,1024,512; I | 1.683390 | 1.794227 | 1.848131 |
| qwen32_gate_up_m8192 | 8192,5120,51200 | 2048,2048,512; O | 2048,2048,512; IO | 22.740079 | 23.175244 | 21.377826 |
| mistral7_gate_up_m512 | 512,4096,28672 | 512,1024,1024; O | 512,1024,512; IO | 0.910736 | 0.980084 | 0.973283 |
| gemma9_gate_up_m512 | 512,3584,28672 | 512,1024,512; O | 512,1024,512; O | 0.839663 | 0.907449 | 0.898190 |

Cubic selects O on 11 shapes and P on five. Strassen selects IO on six, O on four, I on three and P on three. These counts describe selected configurations; they do not isolate the causal benefit of any variant because tile and variant were selected jointly.

## The 78 scoped VMEM failures

All failures occurred during compilation. Every recorded error reports a **56 MiB scoped VMEM allocation against the configured 48 MiB limit**, exceeding it by 8 MiB. These are compiler-scoped VMEM rejections, not demonstrated exhaustion of device HBM.

| Family | Candidate ID | Complete-call failures | Prepared-kernel failures | Distinct affected shapes |
|---|---|---:|---:|---:|
| Cubic | c_plain_2048_2048_512 | 11 | 15 | 15 |
| Strassen | s_plain_2048_2048_512 | 11 | 15 | 15 |
| Strassen | s_interleaved_2048_2048_512 | 11 | 15 | 15 |
| Total | Three parameter configurations | 33 | 45 | 45 candidate–shape attempts |

Each of these three candidates succeeds in both scopes for `square_512`. Each fails only in `prepared_kernel` for `tail_513`, `small_m_8`, `small_m_64` and `mkn_512_8192_2048`; its complete-call result remains recorded but is not selection-eligible. For the other 11 shapes, both scopes fail. Thus the 78 rows comprise 33 attempts with two failed scopes plus 12 attempts with one failed scope.

The same 2048×2048×512 tile remains feasible for output-accumulator variants and is selected for some larger problems. The evidence therefore does not support declaring that tile universally infeasible. The call/prepared asymmetry is observed compiler behavior; this review does not infer its cause without inspecting compiled programs.

## Interpretation and confirmation requirements

The minimum of 20 noisy screening estimates is subject to selection bias. Native is measured only in the first candidate-pair group for each shape, and selected cubic and Strassen candidates can come from different groups. Consequently, ratios formed from the table are not independently paired confirmation estimates and may reflect within-screen drift. The separate 30-round confirmation compares frozen selected cubic, selected Strassen and native together on a new Gaussian seed, addressing this limitation without retuning.

For small matrices, sub-millisecond host-dispatched complete-call latency and padding can dominate the result. The oversized cubic tile selected for `square_512` is a valid observed screen winner, not evidence of a universal tile optimum. Equal attempted budgets also leave different feasible family counts after the recorded VMEM rejections. Report the attempt budget, feasible counts and fixed VMEM limit together.

Before making a confirmed speedup statement, retain confirmation numerical passes, full paired rounds, both timing scopes, exact selection-file hash and cohort/source checks. Do not replace failures, adjust gates or reinterpret Gaussian eligibility as an automatic accuracy guarantee for real activations or cancellation-sensitive inputs. No issue found here requires changing the frozen screen or its selected candidates.

## Integrity checks performed

Verified every file hash named by the root execution artifact manifest, canonical artifact seal, benchmark source manifest and outer source manifest. Verified the selection's journal, source-manifest, campaign and shape-manifest links and its environment identity. Compared current execution, source and artifact manifests with their bytes in the stated archive commit; compared every outer source-manifest file hash with its blob in the stated source commit. Verified source-commit consistency and the source-tar hash. All checks matched.

Primary evidence: [selections.json](../runs/20260919T055706Z-N5-screen-v5e-v004-25842c/artifacts/selections.json), [results.jsonl](../runs/20260919T055706Z-N5-screen-v5e-v004-25842c/artifacts/results.jsonl), [summary.json](../runs/20260919T055706Z-N5-screen-v5e-v004-25842c/artifacts/summary.json), and [frozen campaign](../runs/20260919T055706Z-N5-screen-v5e-v004-25842c/artifacts/config_snapshot/campaign_n5_n9_v1.json).
