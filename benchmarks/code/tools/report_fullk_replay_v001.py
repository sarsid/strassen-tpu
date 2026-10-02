"""All fixed pilot replays, retaining within-allocation paired comparisons."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from report_v6e_suite_v001 import paired


def build(cohort,out):
    out.mkdir(parents=True,exist_ok=False);results=[];missing=[];hashes={};counts=Counter()
    shapes=json.loads((cohort/'source/configs/fullk_replay_v001/shapes.json').read_text())['shapes']
    lines=['# Prior pilot tile replay','',
        'The wider study used a bounded output-tile grid. This separate replay retains the earlier pilot winners and omitted pilot output geometries on the 8192 cube and parent shape, in both output precisions. The larger cube tile previously failed to compile with two buffers; one buffer and final BF16 stores are now also tested.',
        'Every comparator was rerun together on this allocation; timing samples are not pooled with the previous allocations. Broad-study Native, short and full choices remain frozen.',
        'Original FP32 kernels are also rerun at their prior winning tiles with two buffers. Their matching new-body comparisons isolate implementation changes at the same geometry and precision.',
        'Three new inputs × 30 paired rounds. Pointwise hierarchical paired 95% intervals, no multiplicity correction. All fixed arms reported, no tuning selection from these results.',
        'BF16 operands/preadds, FP32 accumulation/reconstruction, one final FP32/BF16 store. Both tracks use 112 MiB custom VMEM. Errors use sampled all-K FP64 and full-output finiteness.','']
    for i,s in enumerate(shapes,1):
        receipt=cohort/f'replay-{i:02d}-measure-finished.json'
        if not receipt.exists():missing.append(str(receipt));continue
        root=Path(json.loads(receipt.read_text())['run'])/'artifacts';p=root/'results.jsonl'
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();events=[json.loads(l) for l in p.read_text().splitlines()]
        for g in json.loads((root/'planned_cases.json').read_text()):
            rows=[r for r in events if r.get('group_id')==g['group_id']]
            cases=[r for r in rows if r['event']=='case_result'];samples=[r for r in rows if r['event']=='sample']
            counts.update(r['status'] for r in cases)
            raw={a['arm_id']:{(r['seed'],r['round']):r['elapsed_ms'] for r in samples if r['arm_id']==a['arm_id']} for a in g['arms']}
            methods={}
            for a in g['arms']:
                arm=a['arm_id'];cc=[r for r in cases if r['arm_id']==arm];mm=[r['correctness'] for r in cc if r.get('correctness')]
                methods[arm]=dict(candidate=a,mean_ms=statistics.mean(raw[arm].values()) if raw[arm] else None,
                    eligible=len(cc)==3 and len(raw[arm])==90 and all(r['eligible_for_speedup_claim'] for r in cc),
                    terminal_outcomes=len(cc),case_statuses=[r['status'] for r in cc],
                    errors={k:max(m[k] for m in mm) for k in ('relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error')
                            if mm and all(isinstance(m.get(k),(float,int)) and math.isfinite(m[k]) for m in mm)},comparisons={})
            for arm,m in methods.items():
                depth=m['candidate']['depth'];refs=['Broad_native','Broad_native_default']
                if depth in (1,2):refs += [f'Broad_full_s{depth}',f'Broad_short_s{depth}']
                a=m['candidate']
                if a['tile'] and a['tile'][2]==g['shape']['k'] and a['output_dtype']=='float32' and a['buffers']==2:
                    refs += [f'Original_S{depth}_{a["tile"][0]}_{a["tile"][1]}_b2']
                for ref in refs:
                    if ref not in methods:continue
                    c=paired(raw[ref],raw[arm])
                    if c:c['eligible']=m['eligible'] and methods[ref]['eligible'];m['comparisons'][ref]=c
            def cmp(m,ref):
                c=m['comparisons'].get(ref)
                return '—' if not c else f'{c["speedup"]:.3f} [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}]'+(' INELIGIBLE' if not c['eligible'] else '')
            lines+=['## '+g['group_id'],'', '| Arm | Tile BM, BN, BK | Buffers | Mean ms | vs tuned Native [95%] | vs default Native [95%] | vs broad full-K [95%] | Relative L2 % |','|---|---|---:|---:|---|---|---|---:|']
            for arm,m in methods.items():
                a=m['candidate'];ms='—' if m['mean_ms'] is None else f'{m["mean_ms"]:.5f}'
                error='—' if 'relative_l2' not in m['errors'] else f'{100*m["errors"]["relative_l2"]:.4f}'
                lines.append(f'| {arm} | {a["tile"]} | {a["buffers"] if a["tile"] else "—"} | {ms} | '+cmp(m,'Broad_native')+' | '+cmp(m,'Broad_native_default')+' | '+cmp(m,f'Broad_full_s{a["depth"]}')+f' | {error} |')
            matching=[(arm,m,ref) for arm,m in methods.items() if not arm.startswith('Original_')
                      for ref in m['comparisons'] if ref.startswith('Original_')]
            if matching:
                lines+=['', '| New body at matching full-K tile | vs original body [95%] |', '|---|---|']
                lines += ['| '+arm+' | '+cmp(m,ref)+' |' for arm,m,ref in matching]
            lines.append('');results.append(dict(shape={k:v for k,v in g['shape'].items() if k!='arms_by_dtype'},methods=methods))
    complete=not missing and len(results)==4 and all(m['terminal_outcomes']==3 for r in results for m in r['methods'].values())
    lines+=['Case statuses: '+str(dict(counts))+'.','', 'No LLM prediction accuracy is measured.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(completed=complete,results=results,missing=missing,case_status_counts=dict(counts),input_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(completed=complete,case_status_counts=dict(counts))))
    return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
