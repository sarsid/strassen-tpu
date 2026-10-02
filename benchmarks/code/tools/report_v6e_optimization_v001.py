"""Combine two separately confirmed v6e rounds without pooling allocations."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--followup',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);hashes={};rounds=[]
    for label,cohort in [('First round: four v5e-winning shapes',a.first),('Follow-up: two long-K shapes on another allocation',a.followup)]:
        path=cohort/'operations/report/artifacts/results.json';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        report=json.loads(path.read_text());completion=json.loads((cohort/'controller-completion.json').read_text())
        if completion['status']!='completed' or not completion['allocation_released']:raise ValueError('Incomplete or unreleased cohort')
        rounds.append(dict(label=label,cohort=str(cohort),report=report))
    lines=['# v6e kernel optimization results','',
        '2026-09-23. Only Strassen depths 1 and 2 were tested. The four initial shapes had confirmed v5e wins over both default and tuned Native; shape selection preceded the new v6e measurements.','',
        'The new kernels either reconstruct in compiler-managed SSA values per K panel, defer all reconstruction until K accumulation finishes, or (depth two follow-up) defer only the outer level. Inputs and every operand combination remain BF16; products, accumulation and outputs remain FP32. Deferral changes FP32 addition order.','',
        'The first round screened 136 configurations. The adaptive follow-up screened 30 configurations on two long-K shapes, motivated by the first round\'s VMEM failures. Each selected method was confirmed with 30 paired rounds on each of three fresh Gaussian inputs. Controls were remeasured on each allocation; no timing samples are pooled across allocations.','',
        'Latency below covers a synchronized complete device call, including padding and cropping, excluding compilation and transfers. These are synthetic MM measurements on the historical winning shapes, not a new real-weight or whole-LLM evaluation.','']
    for rd in rounds:
        lines += ['## '+rd['label'],'',
            '| Shape (M,K,N) | Native default ms | Tuned Native ms | Optimized S1 ms (change) | Optimized S2 ms (change) | Hybrid S2 ms (change) |',
            '|---|---:|---:|---:|---:|---:|']
        def cell(row):return f'{row["mean_ms"]:.6f} ({100*(row["latency_ratio"]-1):+.2f}%)'
        for shape,r in rd['report']['results'].items():
            dims=r['native']['cases'][0]['kernel_metadata']['shape_mkn']
            lines.append(f'| {tuple(dims)} | {r["native_default"]["mean_ms"]:.6f} | {r["native"]["mean_ms"]:.6f} | {cell(r["optimized1"])} | {cell(r["optimized2"])} | {cell(r["hybrid2"]) if "hybrid2" in r else "—"} |')
        lines += ['','Negative changes mean lower latency than tuned Native. Pointwise paired 95% intervals:','']
        for shape,r in rd['report']['results'].items():
            for family in ('optimized1','optimized2','hybrid2'):
                if family not in r:continue
                row=r[family];lo,hi=row['ci95'];old=r['original1' if family=='optimized1' else 'original2']
                lines.append(f'- {shape}, {family}: {100*(lo-1):+.2f}% to {100*(hi-1):+.2f}%; {100*(row["mean_ms"]/old["mean_ms"]-1):+.2f}% latency change versus its independently measured original kernel. Error gate: {row["numerically_eligible"]}; worst sampled relative L2 {100*row["worst_relative_l2"]:.3f}%.')
        report_path=Path(rd['cohort'])/'operations/report/artifacts/RESULTS.md'
        lines += ['',f'[Full timings, errors, selections and failures](<{report_path}>)','']
    lines += ['## Validation and interpretation','',
        'The first implementation passed 16 CPU exact-integer checks and 10 compiled TPU smoke variants. The hybrid passed four CPU algebra checks, an offline per-shape-control coverage check and a seven-variant compiled TPU smoke. CPU interpretation validates algebra, padding and multiple K panels; actual compiled TPU tests qualify device arithmetic.','',
        'Numerical eligibility is unchanged: relative L2 at most 2%, finite output, and the existing maximum-error gate. Floating references use exact BF16-quantized operands and FP64 accumulation over all K for 128 sampled rows by 128 sampled columns. Maximum error is sampled; output finiteness is checked in full. These gates do not establish prediction accuracy.','',
        'Device profiles are separate from ordinary latency samples. They identify actual TPU module times but do not expose direct MXU utilization, memory bandwidth, stalls or intra-kernel overlap counters. The initial profile parser assumed aligned host/device clocks and rejected the traces; version 002 validates complete serialized invocation order and stable module identities instead. Both analyses are preserved.','',
        'The new source is kernels_v6e_v001.py (per-panel and deferred reconstruction) and kernels_v6e_v002.py (hybrid depth two). No automatic LLM dispatch policy has been changed. All runtime allocations were released after verified artifact retrieval.','']
    (a.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (a.output_dir/'evidence.json').write_text(json.dumps(dict(input_sha256=hashes,cohorts=[r['cohort'] for r in rounds]),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output_dir),rounds=len(rounds))))


if __name__=='__main__':main()
