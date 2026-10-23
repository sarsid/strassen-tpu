"""Every contraction curve, including infeasible tiles and paired comparisons."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from report_v6e_suite_v001 import paired


def build(cohort,out):
    out.mkdir(parents=True,exist_ok=False);hashes={};results=[];failures=[];missing=[]
    shapes=json.loads((cohort/'source/configs/fullk_v001/shapes.json').read_text())['shapes']
    lines=['# Full-contraction probe on v6e','',
      'Four aligned geometries with FP32 output, plus the parent gate/up geometry with BF16 output. BF16 inputs, FP32 accumulation; no model fusion or boundary handling.',
      'Fixed candidate curves: two output-tile geometries, three divisor BK values including all K. Every arm gets three fresh Gaussian inputs × 30 paired rotated timing rounds. No fastest-candidate selection/confirmation stage.',
      'Complete synchronized device-call latency excludes compilation and transfers. Compiler VMEM failures remain visible. Parent source pinned at 95be1fb; current JAX/libtpu and Gaussian inputs differ from upstream historical data.',
      'Pointwise paired hierarchical 95% intervals, without multiplicity correction. Fastest observed settings are descriptive, not independently confirmed tuning winners.','']
    for index,s in enumerate(shapes,1):
        receipt=cohort/f'fullk-{index:02d}-measure-finished.json'
        if not receipt.exists():missing.append(str(receipt));continue
        root=Path(json.loads(receipt.read_text())['run'])/'artifacts';path=root/'results.jsonl'
        if not path.exists():missing.append(str(path));continue
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        events=[json.loads(l) for l in path.read_text().splitlines()]
        planned=json.loads((root/'planned_cases.json').read_text())[0]
        cases=[r for r in events if r['event']=='case_result'];samples=[r for r in events if r['event']=='sample']
        failures.extend(r for r in events if r['event']=='error')
        raw={a['arm_id']:{(r['seed'],r['round']):r['elapsed_ms'] for r in samples if r['arm_id']==a['arm_id']} for a in planned['arms']}
        methods={}
        for a in planned['arms']:
            arm=a['arm_id'];rows=[r for r in cases if r['arm_id']==arm];metrics=[r['correctness'] for r in rows if r.get('correctness')]
            complete=len(rows)==3 and len(raw[arm])==90;eligible=complete and all(r['eligible_for_speedup_claim'] for r in rows)
            errors={k:max(q[k] for q in metrics) for k in ['relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error']
                    if metrics and all(isinstance(q.get(k),(int,float)) and math.isfinite(q[k]) for q in metrics)}
            methods[arm]=dict(candidate=a,complete=complete,eligible=eligible,
                mean_ms=statistics.mean(raw[arm].values()) if raw[arm] else None,errors=errors,cases=rows,comparisons={},
                terminal_outcomes=len(rows),statuses=[r['status'] for r in rows])
        for arm,m in methods.items():
            a=m['candidate'];refs=['Native_default','Native_48MiB']
            if a['depth'] in (1,2) and s['output_dtype']=='float32':refs.append(f'S{a["depth"]}_incumbent')
            if a['tile']:
                bm,bn,bk=a['tile'];refs.append(f'Blocked_cubic_{bm}_{bn}_{bk}')
                if bk==s['k'] and a.get('role')=='panel_curve':
                    refs += [n for n,z in methods.items() if z['candidate'].get('role')=='panel_curve'
                        and z['candidate']['implementation']==a['implementation'] and z['candidate']['depth']==a['depth']
                        and z['candidate']['tile'][:2]==[bm,bn] and z['candidate']['tile'][2]<bk]
            for ref in dict.fromkeys(refs):
                if ref not in methods:continue
                pair=paired(raw[ref],raw[arm])
                if pair:pair['eligible']=m['eligible'] and methods[ref]['eligible'];m['comparisons'][ref]=pair
        lines+=['## '+s['id']+' — '+' × '.join(str(s[d]) for d in ['m','k','n'])+' → '+s['output_dtype'],'',
            '| Arm | Mean ms | Relative L2 % | Sampled max abs | Eligible |','|---|---:|---:|---:|---|']
        for arm,m in methods.items():
            ms='—' if m['mean_ms'] is None else f'{m["mean_ms"]:.6f}'
            rel='—' if 'relative_l2' not in m['errors'] else f'{100*m["errors"]["relative_l2"]:.4f}'
            ab='—' if 'max_abs_error' not in m['errors'] else f'{m["errors"]["max_abs_error"]:.6g}'
            lines.append(f'| {arm} | {ms} | {rel} | {ab} | {m["eligible"]} |')
        lines+=['','Full-K versus shorter panels with identical output tiles:','',
            '| Full-K arm | Shorter-panel arm | Speedup [95%] | Both numerically eligible |','|---|---|---|---|']
        for arm,m in methods.items():
            a=m['candidate']
            if not a['tile'] or a['tile'][2]!=s['k'] or a.get('role')!='panel_curve':continue
            for ref,c in m['comparisons'].items():
                b=methods[ref]['candidate']
                if b.get('role')=='panel_curve' and b['implementation']==a['implementation'] and b['depth']==a['depth'] and b['tile'][:2]==a['tile'][:2] and b['tile'][2]<s['k']:
                    lines.append(f'| {arm} | {ref} | {c["speedup"]:.4f} [{c["ci95"][0]:.4f}, {c["ci95"][1]:.4f}] | {c["eligible"]} |')
        lines.append('');results.append(dict(shape=s,methods=methods))
    attempted=not missing and len(results)==len(shapes) and all(m['terminal_outcomes']==3 for r in results for m in r['methods'].values())
    status_counts=Counter(status for r in results for m in r['methods'].values() for status in m['statuses'])
    lines+=['## Scope and failures','',f'All planned candidates attempted: {attempted}. Case statuses: {dict(status_counts)}.',
        'Errors use the exact BF16 operands against host FP64 on 128×128 output samples across all K; finiteness checked in full. BF16 output rounding remains included in error. No LLM prediction accuracy is measured.',
        'Native uses per-executable settings, never a process-wide scoped-VMEM ceiling. FP32 custom allowance is 112 MiB; BF16 parent reproduction uses its 104 MiB allowance.','']
    for e in failures:lines.append('- '+e.get('arm_id','unknown')+': '+e.get('status','error')+' — '+e.get('message','').split('\n')[0][:400])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(completed=attempted,results=results,failures=failures,missing=missing,case_status_counts=dict(status_counts),input_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(completed=attempted,groups=len(results),case_status_counts=dict(status_counts))))
    return 0 if attempted else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
