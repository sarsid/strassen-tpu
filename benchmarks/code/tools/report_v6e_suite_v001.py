"""Report frozen choices on all 168 shapes, retaining incomplete/error outcomes."""
import argparse,hashlib,json,math,statistics
from collections import Counter
from pathlib import Path
import numpy as np

def paired(native,candidate):
    if native.keys()!=candidate.keys() or len(native)!=90:return None
    keys=sorted(native);n=np.array([native[k]for k in keys]).reshape(3,30);c=np.array([candidate[k]for k in keys]).reshape(3,30)
    if not np.isfinite(n).all() or not np.isfinite(c).all() or min(n.min(),c.min())<=0:return None
    rng=np.random.default_rng(239168);si=rng.integers(0,3,(4000,3,1));ri=rng.integers(0,30,(4000,3,30))
    ratios=n[si,ri].mean((1,2))/c[si,ri].mean((1,2))
    return dict(speedup=float(n.mean()/c.mean()),ci95=np.quantile(ratios,[.025,.975]).tolist(),paired_samples=90)

def classify(method,baseline='native'):
    comparison=method.get('comparisons',{}).get(baseline)
    if not method.get('eligible') or not comparison or not comparison['eligible']:return 'unavailable_or_ineligible'
    lo,hi=comparison['ci95']
    return 'win' if lo>1 else 'loss' if hi<1 else 'inconclusive'

def build(cohort,out):
    out.mkdir(parents=True,exist_ok=False);hashes={};allrows=[];status_counts=Counter();compile_ms=0.;missing=[]
    def read(p):
        raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    manifest=read(cohort/'source/configs/v6e_suite_v001/shapes.json')['shapes']
    cfg=read(cohort/'source/configs/v6e_suite_v001/campaign.json')
    profiles_path=cohort/'operations/device-analysis/artifacts/profiles.json'
    profile_data=read(profiles_path) if profiles_path.exists() else {'results':[],'errors':[{'message':'Device analysis unavailable'}]}
    profiles={p['group']:p for p in profile_data['results']}
    for chunk in range(1,13):
        prefix=f'suite-{chunk:02d}'
        receipts={stage:cohort/(prefix+'-'+stage+'-finished.json')for stage in ('screen','confirm')}
        if not all(p.exists() for p in receipts.values()):missing.append(prefix);continue
        roots={stage:Path(read(p)['run'])/'artifacts'for stage,p in receipts.items()}
        selections=read(roots['screen']/'selections.json')['selected']
        groups=read(roots['confirm']/'planned_cases.json')
        events={}
        for stage,root in roots.items():
            p=root/'results.jsonl';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            events[stage]=[json.loads(l)for l in p.read_text().splitlines()]
            status_counts.update(r['status']for r in events[stage]if r['event']=='case_result')
            compile_ms+=sum(r['compile_ms']for r in events[stage]if r['event']=='compilation' and r.get('status')=='ok')
        for g in groups:
            sid=g['shape']['id'];cases=[r for r in events['confirm']if r['event']=='case_result'and r['shape_id']==sid]
            samples=[r for r in events['confirm']if r['event']=='sample'and r['shape_id']==sid]
            role_arms={role:a for a in g['arms']for role in a['headline_roles']}
            raw={a['arm_id']:{(r['seed'],r['round']):r['elapsed_ms']for r in samples if r['arm_id']==a['arm_id']}for a in g['arms']}
            methods={}
            for role in ('native_default','native','cubic','one_level','two_level','matched_one_level','matched_two_level'):
                a=role_arms.get(role)
                if not a:methods[role]=dict(status='no_frozen_candidate',eligible=False);continue
                rows=[r for r in cases if r['arm_id']==a['arm_id']];metrics=[r['correctness']for r in rows if r.get('correctness')]
                complete=len(rows)==3 and len(raw[a['arm_id']])==90
                screen_eligible=selections[sid].get(role,{}).get('status','eligible')=='eligible'
                eligible=complete and screen_eligible and all(r['eligible_for_speedup_claim']for r in rows)
                mm=next((r['kernel_metadata']for r in rows if r.get('kernel_metadata')), {})
                errors={k:max(v[k]for v in metrics)for k in ('relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error')if metrics and all(k in v and isinstance(v[k],(float,int))for v in metrics)}
                device=(profiles.get(g['group_id'],{}).get('arms',{}).get(a['arm_id']))
                methods[role]=dict(candidate=a,eligible=bool(eligible),status='complete'if complete else 'incomplete',
                    mean_ms=statistics.mean(raw[a['arm_id']].values())if raw[a['arm_id']]else None,
                    errors=errors,padded_volume_ratio=mm.get('padded_volume_ratio'),reference_scopes=sorted({v['reference_scope']for v in metrics}),
                    case_statuses=[r['status']for r in rows],device_profile=device,comparisons={})
            for role,m in methods.items():
                if 'candidate'not in m:continue
                for baseline in ('native_default','native','cubic')+ (('matched_'+role,)if role in ('one_level','two_level')else ()):
                    b=methods[baseline]
                    if 'candidate'not in b:continue
                    pair=paired(raw[b['candidate']['arm_id']],raw[m['candidate']['arm_id']])
                    if pair:
                        pair['eligible']=bool(m['eligible']and b['eligible']);m['comparisons'][baseline]=pair
                m['classification_vs_tuned']=classify(m)
                if m.get('device_profile')and methods['native'].get('device_profile'):
                    m['device_speedup_vs_tuned']=methods['native']['device_profile']['mean_ms']/m['device_profile']['mean_ms']
            dims=[g['shape'][d]for d in ('m','k','n')]
            category='small_or_skinny'if min(dims)<1024 else 'large_2048_aligned'if all(d%2048==0 for d in dims)else 'other_large_dimensions'
            allrows.append(dict(shape_id=sid,shape_mkn=dims,category=category,methods=methods,selection=selections[sid],phase=prefix))
    order={s['id']:i for i,s in enumerate(manifest)};allrows.sort(key=lambda r:order[r['shape_id']])
    def aggregate(rows):
        result={}
        for method in ('one_level','two_level'):
            methods=[r['methods'][method]for r in rows]
            valid=[m for m in methods if classify(m)!='unavailable_or_ineligible']
            result[method]=dict(counts=dict(Counter(classify(m)for m in methods)),
                geomean_speedup=math.exp(statistics.mean(math.log(m['comparisons']['native']['speedup'])for m in valid))if valid else None,
                worst_relative_l2=max((m.get('errors',{}).get('relative_l2',0)for m in methods),default=None),
                mean_call_device_direction_disagreements=sum((m['comparisons']['native']['speedup']>1)!=(m['device_speedup_vs_tuned']>1)for m in valid if 'device_speedup_vs_tuned'in m))
        return result
    summary={category:aggregate([r for r in allrows if category=='all'or r['category']==category])for category in ('all','small_or_skinny','large_2048_aligned','other_large_dimensions')}
    lines=['# All 168 development shapes on v6e','',f'{len(allrows)}/168 shapes reported. Missing batches: {missing}.',
        'BF16 inputs and recursive preadds; FP32 accumulation/output. Synthetic Gaussian A/sqrt(K), B. Current compiler pinned to JAX/jaxlib 0.11.2 and libtpu 0.0.48.',
        'Dimension-only tile choices precede timing. Each family freezes its fastest eligible screening mean, then confirms on three fresh inputs × 30 paired rounds. Default and tuned Native plus independently tuned and exact tile-matched cubic controls are retained.',
        'Pointwise hierarchical paired 95% intervals, conditional on frozen screening choices; no multiplicity adjustment. All shapes are development shapes, not unseen selector validation. Complete-call latency includes device padding/crop and host dispatch/wait; transfers and compilation excluded.',
        'Device profiles are separate 20-observation first-input measurements, without direct occupancy/stall counters. They are never pooled into call intervals. Errors use a full or 128×128 sampled FP64 reference across all K; full output finiteness is checked. Numerical gates do not establish prediction quality.','',
        '## Coverage and comparisons','','| Category | Method | Wins | Losses | Inconclusive | Unavailable/ineligible | Geometric mean speedup |','|---|---|---:|---:|---:|---:|---:|']
    for category,v in summary.items():
        for method,q in v.items():
            c=q['counts'];geo=q['geomean_speedup'];lines.append(f'| {category} | {method} | {c.get("win",0)} | {c.get("loss",0)} | {c.get("inconclusive",0)} | {c.get("unavailable_or_ineligible",0)} | {geo:.4f} |'if geo is not None else f'| {category} | {method} | 0 | 0 | 0 | {len(allrows)} | — |')
    lines+=['','## All shapes','','Speedup >1 is faster. S1/S2 columns compare frozen choices to tuned Native. Device ratios are exploratory means, not paired confirmation intervals.','',
        '| M × K × N | Native ms | S1 ms | S2 ms | S1 call speedup [95%] | S2 call speedup [95%] | S1/S2 device speedup | S1/S2 relative L2 |','|---|---:|---:|---:|---|---|---|---|']
    def number(v):return '—'if v is None else f'{v:.5f}'
    def comparison(m):
        q=m.get('comparisons',{}).get('native')
        if not q:return m.get('status','unavailable')
        return f'{q["speedup"]:.3f} [{q["ci95"][0]:.3f}, {q["ci95"][1]:.3f}]'+(' INELIGIBLE'if not q['eligible']else '')
    for r in allrows:
        m=r['methods'];s1=m['one_level'];s2=m['two_level']
        lines.append('| '+' × '.join(map(str,r['shape_mkn']))+' | '+' | '.join(number(m[k].get('mean_ms'))for k in ('native','one_level','two_level'))+' | '+comparison(s1)+' | '+comparison(s2)+' | '+number(s1.get('device_speedup_vs_tuned'))+' / '+number(s2.get('device_speedup_vs_tuned'))+' | '+number(s1.get('errors',{}).get('relative_l2'))+' / '+number(s2.get('errors',{}).get('relative_l2'))+' |')
    lines+=['','## Audit','',f'Screen/confirmation status counts: {dict(status_counts)}. Total recorded successful compile time: {compile_ms/1000:.1f} seconds.',
        f'Device profile groups analyzed: {len(profiles)}. Profile analysis errors: {len(profile_data["errors"])}.',
        'results.json retains candidate choices, matched cubic and default Native intervals, padding ratios, error metrics, device samples and source hashes. All raw failures remain in immutable phase archives.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(completed=len(allrows)==168,summary=summary,results=allrows,missing_batches=missing,
        case_status_counts=dict(status_counts),successful_compile_seconds=compile_ms/1000,profile_errors=profile_data['errors'],input_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(reported_shapes=len(allrows),summary=summary['all'],missing_batches=missing)))
    return 0 if len(allrows)==168 else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
