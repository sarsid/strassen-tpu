"""Fresh-confirmation evidence for each precision and contraction strategy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from report_v6e_suite_v001 import paired, classify


def build(cohort,out):
    out.mkdir(parents=True,exist_ok=False);hashes={};results=[];missing=[];failures=[]
    statuses={stage:Counter() for stage in ('screen','confirm')}
    shapes=json.loads((cohort/'source/configs/fullk_v002/shapes.json').read_text())['shapes']
    def read(p):
        raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    for i,s in enumerate(shapes,1):
        prefix=f'fullk-{i:02d}';receipts={k:cohort/f'{prefix}-{k}-finished.json' for k in statuses}
        if not all(p.exists() for p in receipts.values()):missing.append(prefix);continue
        roots={stage:Path(read(p)['run'])/'artifacts' for stage,p in receipts.items()}
        events={}
        for stage,root in roots.items():
            p=root/'results.jsonl';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            events[stage]=[json.loads(l) for l in p.read_text().splitlines()]
            statuses[stage].update(r['status'] for r in events[stage] if r['event']=='case_result')
            failures.extend(dict(batch=prefix,stage=stage,**r) for r in events[stage] if r['event']=='error')
        selections=read(roots['screen']/'selections.json')['selected']
        for g in read(roots['confirm']/'planned_cases.json'):
            cases=[r for r in events['confirm'] if r['event']=='case_result' and r['shape_id']==g['shape']['id']]
            samples=[r for r in events['confirm'] if r['event']=='sample' and r['shape_id']==g['shape']['id']]
            raw={a['arm_id']:{(r['seed'],r['round']):r['elapsed_ms'] for r in samples if r['arm_id']==a['arm_id']} for a in g['arms']}
            methods={}
            for a in g['arms']:
                arm=a['arm_id'];rows=[r for r in cases if r['arm_id']==arm];metrics=[r['correctness'] for r in rows if r.get('correctness')]
                complete=len(rows)==3 and len(raw[arm])==90
                errors={k:max(q[k] for q in metrics) for k in ('relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error')
                        if metrics and all(isinstance(q.get(k),(float,int)) and math.isfinite(q[k]) for q in metrics)}
                for role in a['headline_roles']:
                    methods[role]=dict(candidate=a,complete=complete,eligible=complete and all(r['eligible_for_speedup_claim'] for r in rows),
                        mean_ms=statistics.mean(raw[arm].values()) if raw[arm] else None,errors=errors,
                        case_statuses=[r['status'] for r in rows],terminal_outcomes=len(rows),comparisons={})
            for role,m in methods.items():
                refs=['native','native_default']
                if role.endswith('s1'):refs+=['existing_s1','short_s1','matched_'+role]
                if role.endswith('s2'):refs+=['existing_s2','short_s2','matched_'+role]
                for ref in dict.fromkeys(refs):
                    if ref not in methods:continue
                    b=methods[ref];pair=paired(raw[b['candidate']['arm_id']],raw[m['candidate']['arm_id']])
                    if pair:pair['eligible']=m['eligible'] and b['eligible'];m['comparisons'][ref]=pair
            results.append(dict(shape=g['shape'],methods=methods,selection=selections[g['shape']['id']],
                                confirmed_unique_arms=len(g['arms']),terminal_outcomes=len(cases)))
    complete=not missing and len(results)==2*len(shapes) and all(r['terminal_outcomes']==3*r['confirmed_unique_arms'] for r in results)
    lines=['# Full-contraction shape and precision study','',
        '12 development geometries, each measured with FP32 output and BF16 output separately. BF16 inputs/preadds, FP32 accumulation and reconstruction, one final output conversion.',
        'Screening chooses Native and, for each depth, the best shorter-panel and full-K settings. Frozen choices, existing controls and matched blocked cubic are confirmed on three fresh inputs × 30 paired rotated rounds. No confirmation reselection.',
        'Both output precisions use 112 MiB for custom kernels. Full K offers one/two input buffers; shorter panels use two. No padding, depths 3/4 or LLM fusion.',
        'Pointwise paired hierarchical 95% intervals conditional on screening, without multiplicity correction. These are development shapes, not unseen-shape validation of a selector.',
        'Complete-call timing includes final conversion, excludes compilation/transfers. Numerical errors are sampled 128×128 all-K FP64 comparisons; finiteness covers the entire output.','']
    summary={}
    def val(v):return '—' if v is None else f'{v:.5f}'
    def comp(m,baseline):
        c=m.get('comparisons',{}).get(baseline)
        if not c:return 'unavailable'
        return f'{c["speedup"]:.3f} [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}]'+(' INELIGIBLE' if not c['eligible'] else '')
    for dtype in ('float32','bfloat16'):
        subset=[r for r in results if r['shape']['output_dtype']==dtype];summary[dtype]={}
        lines+=['## '+dtype+' output','']
        for depth in (1,2):
            role=f'full_s{depth}';short=f'short_s{depth}';summary[dtype][str(depth)]={}
            for baseline in (short,'native',f'existing_s{depth}'):
                counts=Counter(classify(r['methods'].get(role,{}),baseline) for r in subset)
                summary[dtype][str(depth)][baseline]=dict(counts)
            lines += [f'### Strassen {depth}','',
                '| M × K × N | Native ms | Existing ms | Tuned shorter ms | Tuned full-K ms | Full vs shorter [95%] | Full vs Native [95%] | Full relative L2 % |',
                '|---|---:|---:|---:|---:|---|---|---:|']
            for r in subset:
                s=r['shape'];m=r['methods'];full=m.get(role,{})
                lines.append('| '+' × '.join(str(s[d]) for d in ('m','k','n'))+' | '
                    +' | '.join(val(m.get(k,{}).get('mean_ms')) for k in ('native',f'existing_s{depth}',short,role))
                    +' | '+comp(full,short)+' | '+comp(full,'native')+' | '
                    +val(100*full['errors']['relative_l2'] if full.get('errors',{}).get('relative_l2') is not None else None)+' |')
            lines+=['',str(summary[dtype][str(depth)]),'']
        lines+=['### Frozen full-K tiles','',
            '| Shape | S1 BM, BN, BK; buffers | S2 BM, BN, BK; buffers |','|---|---|---|']
        for r in subset:
            entries=[]
            for depth in (1,2):
                a=r['methods'].get(f'full_s{depth}',{}).get('candidate')
                entries.append(str(a['tile'])+'; '+str(a['buffers']) if a else 'no eligible candidate')
            lines.append('| '+r['shape']['base_shape_id']+' | '+' | '.join(entries)+' |')
        lines.append('')
    lines+=['## Coverage and failures','',f'All planned batches and confirmation outcomes present: {complete}.',
        'Case status counts: '+str({k:dict(v) for k,v in statuses.items()})+'.',
        'results.json preserves all selected methods, Native default, existing and tile/buffer-matched cubic controls, timing intervals, error metrics and input hashes. Raw screening candidates and failures remain in the per-phase archives.',
        'The matched cubic uses the parent blocked half-tile arithmetic; it matches whole-tile geometry and pipeline buffers, not the S2 leaf decomposition. Numerical gates do not establish LLM prediction quality.','']
    for r in failures:lines.append('- '+r['batch']+'/'+r['stage']+'/'+r.get('arm_id','?')+': '+r.get('message','').split('\n')[0][:300])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(completed=complete,results=results,summary=summary,failures=failures,
        missing=missing,case_status_counts={k:dict(v) for k,v in statuses.items()},input_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(completed=complete,precision_groups=len(results),case_status_counts={k:dict(v) for k,v in statuses.items()})))
    return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
