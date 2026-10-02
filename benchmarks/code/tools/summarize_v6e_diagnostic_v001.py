"""Summarize measured ablations and static low-level instruction evidence."""
import json,os,hashlib
from pathlib import Path
root=Path(os.environ['STRASSEN_PROJECT_ROOT']);out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
paths={
 'profiles':root/'runs/20260923T175928Z-v6e-architecture-profiles-v001-6426b4/artifacts/profiles.json',
 'llo':root/'runs/20260923-v6e-llo-spill-recovery-v001/llo-spill-recovery-v002/get_llo_analysis.json',
 'capabilities':root/'runs/20260923-v6e-diagnostic-v001/phases/20260923-v6e-diagnostic-v001-diag-smoke-7aff92/artifacts/capabilities.json'}
profiles=json.loads(paths['profiles'].read_text());llo=json.loads(paths['llo'].read_text());assert llo['success']
modules={}
for m in llo['modules']:
 name=m['hlo_instruction_name']
 if name not in ('fusion','v6e_s1_deferred_buf2_mn.1','v6e_s2_hybrid_buf2_mn.1'):continue
 h=m['opcode_histogram'];mxu=h['OPCODE_VECTOR_MATMUL_MUBR']
 modules[name]=dict(matrix_instructions=mxu,load_store_per_matrix=m['execution_unit_summary']['LOAD_STORE']/mxu,
     bf16_add_sub_per_matrix=(h.get('OPCODE_VECTOR_ADD_BF16',0)+h.get('OPCODE_VECTOR_SUBTRACT_BF16',0))/mxu,
     fp32_add_per_matrix=h.get('OPCODE_VECTOR_ADD_F32',0)/mxu,static_histogram=h,execution_units=m['execution_unit_summary'])
assert len(modules)==3
lines=['# v6e architecture diagnostic results','',
 'Completed before the ten-shape benchmark: eight exact CPU checks, eight compiled TPU integer checks, 48 controlled diagnostic attempts (44 numerical passes; four compile rejections of triple buffering), and a separate successful low-level trace.',
 '', 'JAX/jaxlib 0.11.2, libtpu 0.0.48, xprof-nightly 2.24.2a20260922. The TPU runtime reports v6e, two MXUs, 256 MXU columns, 128 vector lanes, eight vector sublanes, and 128 MiB VMEM. The older executed sources remain unchanged.',
 '', '## Actual device timings','',
 'Each number is the mean of 20 ordinary device-module trace observations, separate from unprofiled synchronized-call measurements. These are controlled exploratory diagnostics, not independent confirmation intervals. Dimensions are M,K,N.', '',
 '| Method | (8192,8192,4096), ms | (4096,4096,16384), ms |','|---|---:|---:|']
by={p['group']:p['arms']for p in profiles['results']};a=by['diagnostic_long_k__profile_0'];b=by['diagnostic_wide__profile_0']
for k in ('native_vmem_96m','cubic_legacy','cubic_large','original1_legacy','original1_large','s1_panel_large','s1_deferred_large','original2_legacy','original2_large','s2_hybrid_large','s1_deferred_buf1','s1_deferred_buf2','s1_deferred_nm','s2_hybrid_buf1','s2_hybrid_buf2','s2_hybrid_nm'):
 lines.append(f'| {k} | {a[k]["mean_ms"]:.6f} | {b[k]["mean_ms"]:.6f} |')
lines += ['', '## Findings', '',
 '- Pallas is using actual matrix instructions and DMA; this is not a scalar fallback. Both Strassen kernels contain OPCODE_VECTOR_MATMUL_MUBR and compiler-managed input transfers. This does not prove simultaneous saturation of both MXUs.',
 '- Larger tiles improve the original kernels substantially. On the long-K diagnostic, original S1 falls from 0.856 to 0.708 ms and original S2 from 0.968 to 0.788 ms. These gains exist before changing the algebraic reconstruction schedule.',
 '- At the same larger tile, deferred S1 reaches 0.646 ms and hybrid S2 0.685 ms, against Native 0.678 ms. On the wide-output diagnostic, the best tested S1 remains at 0.696 ms and S2 at 0.741 ms, versus Native 0.651 ms. A universal win is not supported.',
 '- Disabling input double buffering increases device time by about 67–77% in the matched probes. Reversing M/N traversal changes these same probes by less than 0.2%. Retain double buffering and the simpler existing order.',
 '- Triple buffering is rejected by this pallas_call lowering: only one and two buffers are supported. Manual emit_pipeline is a possible follow-up, not a measured result of this experiment.',
 '- Native delivers approximately 811 and 845 TFLOP/s on the two diagnostics, about 88–92% of the published 918 TFLOP/s peak. This is a FLOP/time estimate; no direct MXU occupancy or stall counters were captured.',
 '- Original winning tiles already have v6e-aligned leaf K/N dimensions. Alignment alone is not the issue. Leaf size, reconstruction and live-memory pressure are relevant.',
 '', '## Low-level compiler evidence','',
 '| Probe kernel | BF16 additions/subtractions per matrix instruction | FP32 additions per matrix instruction | Load/store instructions per matrix instruction |','|---|---:|---:|---:|']
for k,r in modules.items():lines.append(f'| {k} | {r["bf16_add_sub_per_matrix"]:.3f} | {r["fp32_add_per_matrix"]:.3f} | {r["load_store_per_matrix"]:.3f} |')
lines += ['',
 'These are ratios in the compiled static instruction listing, not dynamic operation counts, cycle percentages, memory-bandwidth utilization, or proof of a particular stall. Native and Strassen use different tile/loop structures; raw instruction totals are not comparable runtime totals. The listings nevertheless establish the extra vector and load/store work associated with these concrete implementations.',
 '',
 'The low-level probe used (2048,4096,2048), S1 tile (2048,2048,2048) and S2 tile (2048,1024,2048), in BM,BN,BK order. Native used compiler-selected tiling and cross-program input prefetch on this small probe. The two larger ordinary diagnostic Native executables did not show cross-program prefetch. Low-level instrumented timings and overlapping operation spans are excluded from performance comparisons.',
 '',
 'XProf reported success and emitted actual MXU, vector and load/store instruction listings. The instruction listing also contains validation/helper modules; the table filters them out. The tool spilled 83.2 MiB of analysis and 41.2 MiB of debug text outside the trace directory; both were recovered and hash-verified before release.',
 '', '## Next optimization decisions','',
 'The ten-shape experiment will retain double buffering and test larger/aspect-ratio-adjusted tiles, panel versus deferred S1 reconstruction, and hybrid S2 reconstruction. It will include the original kernels, a classical Pallas control, and five Native compiler-memory settings. All 21 candidates per shape receive common inputs and paired timing rounds; finalists are frozen before three fresh seeds and 30 paired confirmation rounds.',
 '',
 'Further justified work is to reduce BF16 operand combination traffic and FP32 scratch/output traffic, while preserving large leaf dots and pipelining within 128 MiB VMEM. Explicit deeper pipelines and reusable weight transformations require separate complete-call/amortization measurements. This diagnostic does not establish a globally optimal kernel or full utilization of every v6e resource.',
 '',
 'The compiler was upgraded as a complete environment, not A/B tested on the same allocation against 0.7.2. Do not assign cross-run differences solely to the upgrade.',
 '',
 'Sources and architectural rationale: docs/V6E_DIAGNOSTIC_PROTOCOL_v001.md. Raw controlled results: runs/20260923-v6e-diagnostic-v001. No full-model inference or prediction-accuracy experiment is claimed here.', '']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(out/'summary.json').write_text(json.dumps(dict(completed=True,instruction_ratios=modules,input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()for k,p in paths.items()}),indent=2)+'\n')
print('Two hardware profiles and three filtered low-level modules summarized; estimates explicitly separated from counters')
