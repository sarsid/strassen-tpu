# N6 device profiling findings

The recovery obtained valid TPU device traces for all 18 captures. **Strassen's device module is faster than selected cubic for the registered win representative, slightly slower for the inconclusive representative, and substantially slower for the small-M loss representative. Native XLA has the shortest traced device-module duration for all three.** Native's device ranking differs from some ordinary complete-call rankings; the two measurement scopes must remain separate.

This is useful diagnostic evidence about the selected implementations. It does not measure MXU/vector utilization or establish that their hardware execution overlaps.

## Recovery and preserved experiment

Run: `20260919T064743Z-N6-v5e-v004-e7bc5e`.

Archive commit: `fe25160abd30be5862c0315b9bbe87558be35b7d`.

Journal SHA256: `39adb04249eb33310b0a758426aacfc6a78ad2d9a4f3f1e0b5d8a6484c9bf537`.

The first N6 run, `20260919T063959Z-N6-v5e-v004-d93a61`, remains archived with all 18 captures marked as device-trace failures. Its unsupported `TRACE_ONLY` setting was a protocol-authoring mistake. The recovery used the documented `TRACE_COMPUTE_AND_SYNC` setting and explicitly enabled device tracing. The resulting successful collection supports the configuration correction; it did not change the measured MM algorithms.

Independent inspection verified identical environment identity, representatives, planned cases, selection inputs, confirmation provenance and selection hashes between original N6 and recovery. The six reused benchmark/kernel/selector dependency files also have identical hashes. The frozen campaign retains SHA256 `535808e9832f4c800c653f5b51eb8707828b2e27c94e705e5bc26c2543da2a5a`; the separate `profile_recovery_override.json` discloses the profiling-only changes. The allocation remains `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, with the same recorded boot, host, device and software identity.

The three shapes were selected by the preregistered lower-median-volume rule within N5 confirmation classes, not by examining these traces. N5 had six Strassen-versus-cubic wins, eight inconclusive cases and two losses. The selected representatives are:

| N5 class | Shape ID | Matrix (M,K,N) | Selected cubic tile (BM,BN,BK); variant | Selected Strassen tile; variant |
|---|---|---|---|---|
| Win | vary_m_8192 | 8192,2048,2048 | 1024,2048,512; output_accumulator | 1024,1024,1024; interleaved_output_accumulator |
| Inconclusive | mkn_512_8192_2048 | 512,8192,2048 | 512,1024,1024; output_accumulator | 512,1024,512; plain |
| Loss | small_m_8 | 8,2048,2048 | 128,512,512; plain | 512,1024,512; output_accumulator |

All inputs remain the original Gaussian seed 20260920 and all numerical gates remain unchanged. These class labels describe the N5 selection basis; they are not assertions that every subsequent diagnostic measurement must reproduce a significant difference.

## Actual device-module durations

Each capture contains one `/device:TPU:0` process and one `XLA Modules` track with exactly eight positive-duration module events. Independent parsing reproduced the reported durations. Across 18 captures, 144 module invocations were retained. The analyzer found no overlapping module intervals or named dropped/overflow/truncation indicators; this is a bounded coverage check, not proof that every possible profiler source lost zero events.

Block 0 profiles native, cubic, then Strassen; block 1 reverses that order. Values below are means of eight module durations per capture, in milliseconds. They exclude the host-side waiting/dispatch interval outside the traced module and remain profiler-instrumented measurements.

| Shape ID | Implementation | Device block 0 ms | Device block 1 ms |
|---|---|---:|---:|
| vary_m_8192 | Native XLA | 0.358877 | 0.358995 |
| vary_m_8192 | Selected cubic | 0.420317 | 0.420340 |
| vary_m_8192 | Selected Strassen | 0.391640 | 0.391817 |
| mkn_512_8192_2048 | Native XLA | 0.092738 | 0.092719 |
| mkn_512_8192_2048 | Selected cubic | 0.101344 | 0.101379 |
| mkn_512_8192_2048 | Selected Strassen | 0.103182 | 0.103161 |
| small_m_8 | Native XLA | 0.003192 | 0.003171 |
| small_m_8 | Selected cubic | 0.015719 | 0.015770 |
| small_m_8 | Selected Strassen | 0.028056 | 0.028016 |

Both capture orders give the same ranking for each shape. Ratios using the average of the two equal-sized blocks are descriptive:

| Shape ID | Cubic device mean / Strassen device mean | Native device mean / Strassen device mean |
|---|---:|---:|
| vary_m_8192 | 1.0730× | 0.9163× |
| mkn_512_8192_2048 | 0.9825× | 0.8988× |
| small_m_8 | 0.5616× | 0.1135× |

For the win representative, the Strassen implementation's device advantage over selected cubic is present in both capture orders. It is not a device advantage over native. For the inconclusive representative, the two custom modules are close and both take longer than native. For M=8, Strassen's module is much longer than either reference. The different selected tile extents and padding are relevant implementation differences, but this experiment does not isolate their effects from arithmetic, layout or compiler scheduling.

## Ordinary complete-call timing before and after profiling

These measurements use synchronized host-observed complete calls, five warmups and 30 paired rounds per arm before and after profiling. All 18 result rows pass the numerical gate. Independent inspection verified 540 positive finite raw samples, exact round coverage and reported arithmetic means.

| Shape ID | Implementation | Before profiling ms | After profiling ms |
|---|---|---:|---:|
| vary_m_8192 | Native XLA | 0.650549 | 0.657936 |
| vary_m_8192 | Selected cubic | 0.659932 | 0.673860 |
| vary_m_8192 | Selected Strassen | 0.632907 | 0.644649 |
| mkn_512_8192_2048 | Native XLA | 0.366344 | 0.371716 |
| mkn_512_8192_2048 | Selected cubic | 0.329335 | 0.327136 |
| mkn_512_8192_2048 | Selected Strassen | 0.333712 | 0.331486 |
| small_m_8 | Native XLA | 0.252021 | 0.240518 |
| small_m_8 | Selected cubic | 0.253736 | 0.231466 |
| small_m_8 | Selected Strassen | 0.263336 | 0.247378 |

For `vary_m_8192`, Strassen/cubic complete-call speedups are 1.0427× [1.0328,1.0525] before and 1.0453× [1.0285,1.0634] after. Against native they are 1.0279× [1.0170,1.0384] and 1.0206× [1.0010,1.0389]. These custom-versus-native complete-call wins coexist with native's shorter traced device module.

For `mkn_512_8192_2048`, Strassen/cubic remains inconclusive before and after: 0.9869× [0.9452,1.0227] and 0.9869× [0.9634,1.0077]. Both custom implementations have shorter ordinary calls than native, although native has the shortest profiled device module.

For `small_m_8`, the before-profile Strassen/cubic comparison is inconclusive at 0.9635× [0.9089,1.0257]; after profiling it is a loss at 0.9357× [0.8787,0.9810]. Native-relative intervals are inconclusive in both ordinary blocks. Mean latencies fall by approximately 4.6%–8.8% across the small-M before/after blocks, underscoring the variability of host-observed sub-millisecond calls.

The recorded intervals are per-comparison paired bootstrap intervals without multiplicity correction. Before/after blocks share one allocation and are diagnostic checks, not independent replications. Do not subtract a profiled device duration from a separately measured ordinary latency and call the difference measured dispatch overhead. Profiling can perturb execution, and module durations and complete-call timings cover different intervals. The ranking reversal is observed; its precise cause is not established here. Headline performance remains the separate N5 confirmation, with both scopes and all outcomes retained.

## Compiler diagnostics and analytical estimates

All nine executables retain compiled HLO text, compiler memory analysis and cost analysis. The compiler reports the following external argument/output sizes, common to all three arms within each shape:

| Shape ID | Argument MiB | Output MiB | Reported temporary MiB |
|---|---:|---:|---:|
| vary_m_8192 | 40 | 64 | 0 |
| mkn_512_8192_2048 | 40 | 4 | 0 |
| small_m_8 | 8.03125 | 0.0625 | 0 |

Zero compiler-reported temporary bytes does not establish zero VMEM scratch, zero register spills or zero additional traffic inside a custom call. The cost-analysis dictionaries omit a FLOP total for the custom kernels. Fields named `utilization0{}` and `utilization1{}` are compiler cost-analysis fields, not sampled MXU or vector-unit utilization. No measured hardware utilization is available from this analysis.

The source-level roofline estimate applies the seven-eighths dot-work factor to padded Strassen shapes and includes estimated vector work separately. It therefore predicts less dot work for the aligned large representatives but not automatically a shorter measured module. For M=8, the selected cubic BM=128 and Strassen BM=512 imply very different padded-work estimates. Those estimates are not runtime instruction counters or proof that the compiler executes every padded operation unchanged.

**The HBM estimate is not a validated lower bound for these warm traces.** The artifact labels a compulsory original-operand I/O estimate as a roofline lower bound. For small-M native it predicts 0.009880 ms, yet the measured device-module mean is approximately 0.003182 ms. Its assumption that all counted original-operand bytes cross HBM within each measured module is therefore not established for this observation. Reused inputs, memory residency, compiler transformations and trace scope require further accounting; this study does not identify which explains the discrepancy. Do not infer achieved bandwidth or utilization by dividing those modeled bytes by the trace duration, or claim that the hardware exceeded a physical bandwidth limit. The frozen estimate remains archived with this explicit limitation.

## What N6 establishes

The corrected profiling configuration obtains the required device evidence without changing the registered MM choices or machine. It shows that the selected Strassen-versus-cubic performance tradeoff has a corresponding device-duration advantage for one representative and device disadvantages for the other two. It also exposes a consequential scope difference: the ordinary custom-versus-native gain in some cases is not a shorter profiled TPU module.

Three training-derived representatives cannot establish a general hardware mechanism or predict all unseen shapes. No MXU/vector utilization, actual coexecution overlap, measured HBM traffic, model quality or cross-generation replication is claimed. The failed original captures, successful recovery traces, ordinary timings and contradictory analytical estimate are all retained as evidence.

Primary artifacts: [summary](../runs/20260919T064743Z-N6-v5e-v004-e7bc5e/artifacts/summary.json), [journal](../runs/20260919T064743Z-N6-v5e-v004-e7bc5e/artifacts/results.jsonl), [profiling override](../runs/20260919T064743Z-N6-v5e-v004-e7bc5e/artifacts/profile_recovery_override.json), [representatives](../runs/20260919T064743Z-N6-v5e-v004-e7bc5e/artifacts/representatives.json), and [recovery rationale](N6_RECOVERY_v001.md). Root and canonical artifact hashes, the committed root manifest and reused dependency hashes were independently checked and matched. No experimental execution or source modification was performed for this review.
