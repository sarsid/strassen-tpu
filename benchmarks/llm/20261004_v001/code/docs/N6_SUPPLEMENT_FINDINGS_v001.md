# N6 supplement: large-shape device profiling findings

**Strassen has the shortest traced TPU device-module duration for both selected large shapes, and the ordinary complete-call measurements retain that ranking before and after profiling.** Its descriptive device speedups are 1.0610× versus native and 1.0883× versus selected cubic for the Qwen-shaped product, and 1.1291× and 1.0932× for the 8192 square.

These are explicitly **post-hoc diagnostic observations** on two favorable, training-derived cases. They supplement the original N6 representatives and ranking reversals; they do not replace them or constitute independent confirmation.

## Provenance and preserved policy

- Run: `20260919T073356Z-N6-v5e-v004-dde5c0`.
- Source commit: `314f0a73985d735b7791f820a64d22450220d24a`.
- Archive commit: `c74d443afcab6d0a06b325b52532bab964b0be61`.
- Journal SHA256: `943ace5f906b0e29f75c4f465b8cacbffe909ce29b861d670ad5e622b24dcb64`.
- Canonical artifact-manifest SHA256: `3cd39d6bcf0f306a4685d8362165f79a1c7158a27ee636806831636f422687d3`.

Independent read-only verification matched all 82 canonical artifact hashes, the archived root manifest against its committed copy, and the supplementary runner against its source commit and snapshot. All six reused benchmark, kernel and selector dependencies match the prior valid N6 recovery. The frozen campaign and N5 selection hashes remain unchanged.

The complete recorded identity equals both N5 confirmation and the prior valid N6 recovery: allocation `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, host `6f3b9b8cc92c`, boot `64fe01c7-4c57-43e9-9f49-6805fb8a887d`, one `TPU v5 lite`, JAX/JAXlib 0.7.2, libtpu 0.0.21.1 and identical remaining package versions, device attributes, compatibility settings and runtime flags. This establishes continuity of the recorded allocation/runtime identity; no physical chip serial was available.

The original median-volume N6 selection policy remains preserved with its original results. The separately recorded supplement rule chooses the two largest M*K*N products whose sealed N5 confirmation passes the numerical gates and has a paired CI95 lower endpoint above 1 against **both** native and selected cubic. It chooses from five qualifying shapes, without using N7 outcomes, retuning a kernel or updating the selector. The [supplement rationale](N6_SUPPLEMENT_RATIONALE_v001.md) and `supplementary_design.json` disclose this different selection rule.

| Shape ID | Matrix (M,K,N) | Selected cubic tile (BM,BN,BK); variant | Selected Strassen tile; variant |
|---|---|---|---|
| qwen32_gate_up_m8192 | 8192,5120,51200 | 2048,2048,512; output_accumulator | 2048,2048,512; interleaved_output_accumulator |
| square_8192 | 8192,8192,8192 | 1024,1024,1024; output_accumulator | 2048,2048,512; interleaved_output_accumulator |

Native uses the original shape. Neither custom implementation requires padding here. BF16 operands and Strassen pre-additions, FP32 accumulation/output, DEFAULT dot precision, the 48 MiB custom-kernel VMEM limit and Gaussian seed 20260920 remain unchanged. These are Qwen-shaped synthetic operands, not model weights or activations.

## Strict trace coverage and device durations

All **12/12 captures passed**, comprising two shapes, three arms and two capture blocks. Independent parsing of every raw compressed trace confirmed exactly one `/device:TPU:0` process and one `XLA Modules` track, containing exactly eight positive, finite, nonoverlapping module events. All **96 individual durations** matched the journal values. The analyzer recorded no named drop, overflow or truncation indicators; this bounded check is not proof that every possible profiler source lost zero events.

The effective profiler mode is the previously qualified `TRACE_COMPUTE_AND_SYNC`, with explicit device tracer level 1. The original campaign bytes, including the original unsupported `TRACE_ONLY` value, remain archived unchanged; the effective override is recorded separately in `supplementary_design.json`. Host tracer level 2, Python tracer level 0 and HLO-proto collection remain fixed.

Each capture follows a warm executable and measures eight synchronized invocations. Block 0 visits native, cubic, then Strassen; block 1 reverses that order. The table reports means of eight instrumented module durations per capture, in milliseconds.

| Shape ID | Implementation | Device block 0 ms | Device block 1 ms |
|---|---|---:|---:|
| qwen32_gate_up_m8192 | Native XLA | 22.304617 | 22.304626 |
| qwen32_gate_up_m8192 | Selected cubic | 22.877360 | 22.877025 |
| qwen32_gate_up_m8192 | Selected Strassen | 21.021821 | 21.022166 |
| square_8192 | Native XLA | 6.013440 | 6.013108 |
| square_8192 | Selected cubic | 5.821843 | 5.821865 |
| square_8192 | Selected Strassen | 5.325792 | 5.325528 |

Both capture orders give the same ranking within each shape. Ratios of the equal-weight block means are descriptive:

| Shape ID | Native / Strassen device mean | Cubic / Strassen device mean |
|---|---:|---:|
| qwen32_gate_up_m8192 | 1.0610× | 1.0883× |
| square_8192 | 1.1291× | 1.0932× |

These 16 instrumented invocations per arm do not supply a new independent replication or a device-speedup confidence interval. The traces establish shorter module durations for these implementations and shapes, without identifying the hardware mechanism.

## Ordinary complete calls before and after profiling

Each ordinary block uses five warmups and 30 paired rounds. All **12/12 result rows passed**. Independent inspection reproduced all reported means from 360 positive finite samples, with complete round coverage. Complete calls include device padding, multiplication and cropping where needed, exclude compilation and host transfers, and are synchronized. The aligned shapes here require no padding.

| Shape ID | Implementation | Before profiling ms | After profiling ms |
|---|---|---:|---:|
| qwen32_gate_up_m8192 | Native XLA | 22.827394 | 22.820175 |
| qwen32_gate_up_m8192 | Selected cubic | 23.298303 | 23.294307 |
| qwen32_gate_up_m8192 | Selected Strassen | 21.429996 | 21.387463 |
| square_8192 | Native XLA | 6.291378 | 6.331507 |
| square_8192 | Selected cubic | 6.097419 | 6.143475 |
| square_8192 | Selected Strassen | 5.605234 | 5.658127 |

| Shape ID | Baseline / Strassen | Before ratio [CI95] | After ratio [CI95] |
|---|---|---|---|
| qwen32_gate_up_m8192 | Native | 1.0652× [1.0636,1.0667] | 1.0670× [1.0647,1.0701] |
| qwen32_gate_up_m8192 | Cubic | 1.0872× [1.0856,1.0890] | 1.0892× [1.0866,1.0925] |
| square_8192 | Native | 1.1224× [1.1209,1.1241] | 1.1190× [1.1162,1.1216] |
| square_8192 | Cubic | 1.0878× [1.0863,1.0896] | 1.0858× [1.0834,1.0881] |

Intervals use 2,000 paired bootstrap resamples, per comparison, without multiplicity correction. The before/after blocks share the same allocation and inputs. They are diagnostic checks, not independent confirmation. Profiling can perturb execution: do not subtract these separately observed call and module timings and label the difference measured dispatch overhead.

## Numerical results under the unchanged gate

Reference values use independent NumPy FP32 multiplication of the exact BF16-quantized operands. Each shape checks a saved 128-row by 128-column output sample, using all K values, and separately tests finiteness over the whole device output. Before and after metrics agree.

| Shape ID | Implementation | Relative L2 error | Maximum absolute error |
|---|---|---:|---:|
| qwen32_gate_up_m8192 | Native XLA | 1.910e-7 | 1.073e-6 |
| qwen32_gate_up_m8192 | Selected cubic | 1.690e-7 | 9.537e-7 |
| qwen32_gate_up_m8192 | Selected Strassen | 0.004437 | 0.025511 |
| square_8192 | Native XLA | 2.192e-7 | 1.669e-6 |
| square_8192 | Selected cubic | 1.780e-7 | 1.192e-6 |
| square_8192 | Selected Strassen | 0.004359 | 0.020615 |

All outputs are finite and pass the fixed requirements: relative L2 ≤ 0.02 and maximum absolute error ≤ 0.001 + 0.05 × maximum absolute reference value. Strassen's roughly 0.44% relative L2 error is substantially larger than either classical reference, while remaining inside this gate. Passing this sampled Gaussian check does not certify arbitrary inputs or model quality. The N4 cancellation failures remain part of the accuracy evidence; no threshold was relaxed.

## Relationship to the original N6 findings

The [original N6 findings](N6_FINDINGS_v001.md) remain necessary context. Native had the shortest device module for all three original representatives. For `vary_m_8192`, Strassen was faster than cubic on-device but slower than native: native/Strassen device mean was 0.9163×, even though ordinary calls favored Strassen over native before and after profiling. For `mkn_512_8192_2048`, both custom calls were shorter than native ordinary calls while native again had the shortest device module. The small-M representative retained substantial custom device disadvantages.

The supplement adds two large cases where ordinary and traced device rankings agree; it does not resolve the cause of those original scope reversals. Both favorable examples were selected using prior positive results, so they cannot estimate the prevalence of device wins or establish held-out generalization.

The original warm-trace HBM-bound contradiction also remains disclosed: the small-M native analytical estimate was 0.009880 ms while its traced module averaged approximately 0.003182 ms. The assumed original-operand HBM traffic per warm module was not established. This supplement does not repair that assumption or justify achieved-bandwidth calculations from modeled bytes.

No measured MXU/vector utilization, overlap, memory-traffic mechanism, model-quality effect or v6e replication follows from this evidence. Keep performance claims anchored in the separate confirmation and held-out studies, with this supplement labeled as a diagnostic. All failed original captures, valid original recovery traces and original representative outcomes remain preserved.

Primary supplement artifacts: [summary](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/summary.json), [journal](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/results.jsonl), [supplementary design](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/supplementary_design.json), [representatives](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/representatives.json), [environment](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/environment.json), and [canonical artifact manifest](../runs/20260919T073356Z-N6-v5e-v004-dde5c0/artifacts/artifact_manifest.json). This review used existing artifacts only; it performed no experiment, test execution, commit or modification of frozen code/results.
