"""Read-only local interim comparison of frozen Strassen-2 headline choices."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime,timezone
import numpy as np
from report_mlsys_campaign_v001 import checked_phase,sha,dump

LABELS={'native':'Tuned Native','cubic':'Tuned cubic','one_level':'Tuned Strassen 1','native_default':'Default Native'}

def paired_speedup(samples,group_id,baseline,candidate):
    left=samples[(group_id,baseline)];right=samples[(group_id,candidate)]
    if set(left)!=set(right) or len(left)!=3:raise ValueError('Expected three paired fresh seeds')
    seeds=sorted(left);aa=[];bb=[]
    for seed in seeds:
        if set(left[seed])!=set(right[seed]) or len(left[seed])!=30:raise ValueError('Expected 30 paired rounds per seed')
        rounds=sorted(left[seed]);aa.append([left[seed][r]for r in rounds]);bb.append([right[seed][r]for r in rounds])
    aa=np.asarray(aa);bb=np.asarray(bb)
    if not(np.isfinite(aa).all()and np.isfinite(bb).all()and(aa>0).all()and(bb>0).all()):raise ValueError('Invalid timing')
    rng=np.random.default_rng(int(hashlib.sha256((group_id+'|'+baseline+'|'+candidate).encode()).hexdigest()[:16],16))
    seed_idx=rng.integers(0,3,(4000,3));round_idx=rng.integers(0,30,(4000,3,30))
    a=aa[seed_idx[:,:,None],round_idx].mean(axis=(1,2));b=bb[seed_idx[:,:,None],round_idx].mean(axis=(1,2))
    bounds=np.quantile(a/b,[.025,.975])
    return dict(speedup=float(aa.mean()/bb.mean()),ci95=[float(x)for x in bounds],baseline_mean_ms=float(aa.mean()),strassen2_mean_ms=float(bb.mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);args=p.parse_args()
    cohort=args.cohort.resolve();out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(exist_ok=False)
    # Snapshot the receipt list once. A phase finishing later belongs to a later scan.
    receipts=sorted(cohort.glob('*-confirm-finished.json'));sources=[];rows=[];identities=[]
    for receipt in receipts:
        run,summary,status=checked_phase(cohort,receipt)
        if status!='completed':continue
        art=run/'artifacts';cfg=json.loads((art/'config_snapshot/campaign.json').read_text())
        study=cfg['campaign_id']
        if study not in ('mlsys_shapes_v001','mlsys_llm_shapes_v001'):continue
        identity=json.loads((art/'environment.json').read_text())['identity'];identities.append(identity)
        stats=json.loads((art/'confirmation_statistics.json').read_text())['by_shape']
        groups=json.loads((art/'planned_cases.json').read_text());group_by_shape={g['shape']['id']:g for g in groups}
        samples=defaultdict(lambda:defaultdict(dict));cases=defaultdict(list)
        for line in (art/'results.jsonl').open():
            event=json.loads(line);key=(event.get('group_id'),event.get('arm_id'))
            if event['event']=='sample':
                seed,round_id=event['seed'],event['round']
                if round_id in samples[key][seed]:raise ValueError('Duplicate sample')
                samples[key][seed][round_id]=event['elapsed_ms']
            elif event['event']=='case_result'and event.get('scope')=='call':cases[key].append(event)
        sources.append(dict(receipt=str(receipt),receipt_sha256=sha(receipt),run=str(run),
            confirmation_sha256=sha(art/'confirmation_statistics.json'),raw_sha256=sha(art/'results.jsonl')))
        for shape_id,record in stats.items():
            group=group_by_shape[shape_id];shape=group['shape'];headlines=record['headline'];gid=group['group_id']
            values={};valid={};accuracy={};tiles={}
            for method,info in headlines.items():
                arm=info.get('candidate_id');observed=cases.get((gid,arm),[])
                good=bool(info.get('available')and info.get('all_inputs_eligible')and info.get('screen_selection_eligible'))
                good=good and len(observed)==3 and all(r.get('eligible_for_speedup_claim')and r['status']=='ok'for r in observed)
                valid[method]=good
                timings=[v for seed in samples.get((gid,arm),{}).values()for v in seed.values()]
                values[method]=statistics.mean(timings)if timings else None
                accuracy[method]=max((r['correctness']['relative_l2']for r in observed if r.get('correctness')),default=None)
                tiles[method]=next((r['kernel_metadata'].get('tile_bm_bn_bk')for r in observed if r.get('kernel_metadata')),None)
            candidate=headlines.get('two_level',{}).get('candidate_id');comparisons={}
            for baseline in LABELS:
                if not(valid.get('two_level')and valid.get(baseline)):continue
                interval=paired_speedup(samples,gid,headlines[baseline]['candidate_id'],candidate)
                if baseline=='native':
                    frozen=headlines['two_level']['comparison_vs_tuned_native']
                    interval.update(speedup=1/frozen['candidate_over_reference_latency_ratio'],
                        ci95=[1/frozen['ci95'][1],1/frozen['ci95'][0]])
                interval.update(mean_win=interval['speedup']>1,ci_win=interval['ci95'][0]>1,ci_loss=interval['ci95'][1]<1)
                comparisons[baseline]=interval
            eligible_means={k:v for k,v in values.items()if valid.get(k)and v is not None}
            fastest=min(eligible_means,key=eligible_means.get)if eligible_means else None
            rows.append(dict(study=study,shape_id=shape_id,m=shape['m'],k=shape['k'],n=shape['n'],mean_ms=values,
                numerically_eligible=valid,max_relative_l2=accuracy,tiles=tiles,comparisons=comparisons,
                all_five_eligible=len(eligible_means)==5,fastest_mean_method=fastest,
                strassen2_clear_win_all_four=len(comparisons)==4 and all(c['ci_win']for c in comparisons.values()),evidence=str(art)))
    if identities and any(x!=identities[0]for x in identities):raise ValueError('Different execution identities')
    if len({(r['study'],r['shape_id'])for r in rows})!=len(rows):raise ValueError('Duplicate confirmation shape')
    results={}
    for study in ('mlsys_shapes_v001','mlsys_llm_shapes_v001'):
        rr=[r for r in rows if r['study']==study];summary={}
        for method in LABELS:
            pairs=[r['comparisons'][method]for r in rr if method in r['comparisons']]
            summary[method]=dict(comparable_shapes=len(pairs),mean_wins=sum(c['mean_win']for c in pairs),ci_wins=sum(c['ci_win']for c in pairs),
                ci_losses=sum(c['ci_loss']for c in pairs),geomean_speedup=math.exp(statistics.mean(math.log(c['speedup'])for c in pairs))if pairs else None,
                best_speedup=max((c['speedup']for c in pairs),default=None))
        top=sorted([r for r in rr if 'native'in r['comparisons']],key=lambda r:r['comparisons']['native']['speedup'],reverse=True)[:6]
        results[study]=dict(confirmed_shapes=len(rr),all_five_eligible=sum(r['all_five_eligible']for r in rr),comparisons=summary,
            strassen2_fastest_mean_all_five=sum(r['all_five_eligible']and r['fastest_mean_method']=='two_level'for r in rr),
            strassen2_ci_win_all_four=sum(r['strassen2_clear_win_all_four']for r in rr),best_vs_native=top)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),cohort=str(cohort),results=results,source_snapshots=sources,
        comparison='Frozen screen-selected configurations on fresh confirmation inputs; no confirmation reselection.',
        confidence='Pointwise 95% hierarchical paired-bootstrap intervals. Tuned Native comparisons reuse frozen recorded intervals. Other pairwise comparisons are exploratory, computed from the same paired rounds using4000 vectorized resamples. No multiplicity adjustment.',
        scope='Interim synthetic matrix results only. Main shapes are ordered by increasing useful volume, so remaining shapes are systematically larger; completed-shape win rates are not estimates of the whole suite.',
        deployment='Local read-only scan. No TPU commands or experiment changes.')
    dump(out/'summary.json',report);dump(out/'shape_results.json',rows)
    lines=['# Interim Strassen 2 scan','',report['comparison'],'',report['scope'],'',report['confidence'],'']
    for study,result in results.items():
        lines.extend(['## '+study,'',f'Confirmed shapes: {result["confirmed_shapes"]}. Strassen2 has the lowest mean of all five on {result["strassen2_fastest_mean_all_five"]}.','',
            '| Baseline | Comparable | Strassen2 mean wins | 95% CI wins | 95% CI losses | Geometric mean speedup |','|---|---:|---:|---:|---:|---:|'])
        for method,c in result['comparisons'].items():
            geo='—'if c['geomean_speedup']is None else f'{c["geomean_speedup"]:.3f}x'
            lines.append(f'| {LABELS[method]} | {c["comparable_shapes"]} | {c["mean_wins"]} | {c["ci_wins"]} | {c["ci_losses"]} | {geo} |')
        lines.extend(['','### Largest observed advantages versus tuned Native','', '| M × K × N | Speedup [95% CI] | Strassen2 relative L2 max | Lowest mean |','|---|---:|---:|---|'])
        for r in result['best_vs_native']:
            c=r['comparisons']['native'];lines.append(f'| {r["m"]} × {r["k"]} × {r["n"]} | {c["speedup"]:.3f}x [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}] | {r["max_relative_l2"]["two_level"]:.5g} | {r["fastest_mean_method"]} |')
        lines.append('')
    with (out/'RESULTS.md').open('x')as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items()if x!='best_vs_native'}for k,v in results.items()},indent=2))

if __name__=='__main__':main()
