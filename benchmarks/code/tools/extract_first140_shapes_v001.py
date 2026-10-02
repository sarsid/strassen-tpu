"""Immutable local export of exactly the first ten confirmed main-shape blocks."""
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import statistics
from report_mlsys_campaign_v001 import checked_phase, sha, dump
from scan_strassen2_interim_v001 import paired_speedup

METHODS = ('native_default', 'native', 'cubic', 'one_level', 'two_level')
LABELS = dict(zip(METHODS, ('Native default', 'Native tuned', 'Cubic', 'Strassen 1', 'Strassen 2')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort', type=Path, required=True);args=p.parse_args()
    cohort=args.cohort.resolve();out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(exist_ok=False)
    cfg=json.loads((cohort/'source/configs/mlsys_shapes_v001/campaign.json').read_text())
    manifest=json.loads((cohort/'source/configs/mlsys_shapes_v001'/cfg['shape_manifest']).read_text())['shapes']
    order={s['id']:i+1 for i,s in enumerate(manifest)}
    rows=[];details=[];pairs=[];sources=[];identities=[];outcomes=0
    for chunk in range(1,11):
        phase=f'MAIN-{chunk:02d}-confirm';receipt=cohort/(phase+'-finished.json')
        run,summary,status=checked_phase(cohort,receipt)
        if status!='completed':raise ValueError('Required phase is incomplete: '+phase)
        audit=json.loads((cohort/'operations'/(phase+'-audit')/'completion.json').read_text())
        if audit['status']!='succeeded':raise ValueError('Required scientific audit failed')
        art=run/'artifacts';identities.append(json.loads((art/'environment.json').read_text())['identity'])
        stats=json.loads((art/'confirmation_statistics.json').read_text())['by_shape']
        groups=json.loads((art/'planned_cases.json').read_text());group_by_shape={g['shape']['id']:g for g in groups}
        samples=defaultdict(lambda:defaultdict(dict));cases=defaultdict(list);observed_keys=set()
        for line in (art/'results.jsonl').open():
            r=json.loads(line);key=(r.get('group_id'),r.get('arm_id'))
            if r['event']=='sample':
                if r['round'] in samples[key][r['seed']]:raise ValueError('Duplicate timing sample')
                samples[key][r['seed']][r['round']]=r['elapsed_ms']
            elif r['event']=='case_result' and r.get('scope')=='call':
                unique=(*key,r['seed'])
                if unique in observed_keys:raise ValueError('Duplicate terminal case')
                observed_keys.add(unique);cases[key].append(r);outcomes+=1
        if len(observed_keys)!=sum(len(g['inputs'])*len(g['arms']) for g in groups):raise ValueError('Incomplete phase coverage')
        sources.append(dict(phase=phase,receipt=str(receipt),receipt_sha256=sha(receipt),run=str(run),
                            raw_sha256=sha(art/'results.jsonl'),statistics_sha256=sha(art/'confirmation_statistics.json')))
        for sid,rec in stats.items():
            group=group_by_shape[sid];shape=group['shape'];gid=group['group_id'];headline=rec['headline']
            dmap={};row=dict(order=order[sid],shape_id=sid,m=shape['m'],k=shape['k'],n=shape['n'],phase=phase,
                geometry='Square' if shape['m']==shape['k']==shape['n'] else 'Rectangle',
                sampling_group='; '.join(sorted({s.get('sampling_group','') for s in shape.get('source_rows',[])})),
                source_rows=shape.get('source_rows',[]))
            row['classical_gflop']=2*row['m']*row['k']*row['n']/1e9
            for method in METHODS:
                h=headline[method];arm=h.get('candidate_id');observed=cases.get((gid,arm),[])
                eligible=bool(h.get('available') and h.get('all_inputs_eligible') and h.get('screen_selection_eligible')
                    and len(observed)==3 and len({r['seed'] for r in observed})==3
                    and all(r.get('eligible_for_speedup_claim') is True and r['status']=='ok' and r['timing']['sample_count']==30 for r in observed))
                # This first-140 workbook promises complete five-way comparisons.
                if not eligible:raise ValueError('Ineligible headline requires explicit missing-data handling: '+sid+'/'+method)
                timings=[v for seed in samples[(gid,arm)].values() for v in seed.values()]
                if len(timings)!=90 or any(not math.isfinite(x) or x<=0 for x in timings):raise ValueError('Invalid timing coverage')
                correctness=[r['correctness'] for r in observed];meta=observed[0]['kernel_metadata']
                mean=statistics.mean(timings);frozen=h['comparison_vs_tuned_native']
                if abs(mean-frozen['candidate_mean_ms'])>max(1e-10,mean*1e-10):raise ValueError('Frozen mean mismatch')
                tile=meta.get('tile_bm_bn_bk');metrics={name:max(r[name] for r in correctness) for name in
                    ('relative_l2','normwise_error','max_abs_error','rmse','p99_abs_error')}
                scopes={r['reference_scope'] for r in correctness}
                if len(scopes)!=1:raise ValueError('Mixed reference scopes')
                uncertainty=rec['tile_uncertainty'].get(method,{})
                alternatives=[r for r in uncertainty.get('confirmed_candidates',[]) if r['candidate_id']!=arm and r.get('all_inputs_eligible')]
                indistinguishable=sum(r['comparison_to_frozen_winner']['ci95'][0]<=1<=r['comparison_to_frozen_winner']['ci95'][1] for r in alternatives)
                faster=sum(r['comparison_to_frozen_winner']['ci95'][1]<1 for r in alternatives)
                d=dict(order=row['order'],shape_id=sid,method=method,label=LABELS[method],m=row['m'],k=row['k'],n=row['n'],
                    candidate_id=arm,mean_ms=mean,eligible=eligible,samples=90,fresh_inputs=3,tile=tile,
                    padded_volume_ratio=meta['padded_volume_ratio'],reference_scope=next(iter(scopes)),
                    reference_sample_count=min(r['sample_count'] for r in correctness),all_k_used=row['k'],
                    confirmed_alternatives=len(alternatives),indistinguishable_alternatives=indistinguishable,
                    faster_alternatives=faster,phase=phase,**{'max_'+k:v for k,v in metrics.items()})
                dmap[method]=d;details.append(d)
            row['methods']=dmap
            # Give tuned Native precedence when it is the same measured default arm.
            row['fastest_mean_method']=min(('native','native_default','cubic','one_level','two_level'),key=lambda m:dmap[m]['mean_ms'])
            row['all_five_eligible']=True;row['comparisons']={}
            for baseline in METHODS[:-1]:
                comparison=paired_speedup(samples,gid,headline[baseline]['candidate_id'],headline['two_level']['candidate_id'])
                if baseline=='native':
                    frozen=headline['two_level']['comparison_vs_tuned_native']
                    comparison.update(speedup=1/frozen['candidate_over_reference_latency_ratio'],ci95=[1/frozen['ci95'][1],1/frozen['ci95'][0]])
                lo,hi=comparison['ci95'];comparison['result']='S2 faster' if lo>1 else 'S2 slower' if hi<1 else 'Inconclusive'
                row['comparisons'][baseline]=comparison
                pairs.append(dict(order=row['order'],shape_id=sid,m=row['m'],k=row['k'],n=row['n'],baseline=baseline,
                                  label=LABELS[baseline],phase=phase,**comparison))
            rows.append(row)
    if len(rows)!=140 or set(r['order'] for r in rows)!=set(range(1,141)):raise ValueError('Not exactly the first 140 manifest shapes')
    if any(x!=identities[0] for x in identities):raise ValueError('Mixed execution identity')
    rows.sort(key=lambda r:r['order']);details.sort(key=lambda r:(r['order'],METHODS.index(r['method'])));pairs.sort(key=lambda r:(r['order'],METHODS.index(r['baseline'])))
    comparisons={}
    for baseline in METHODS[:-1]:
        selected=[r['comparisons'][baseline] for r in rows]
        comparisons[baseline]=dict(mean_wins=sum(r['speedup']>1 for r in selected),ci_wins=sum(r['result']=='S2 faster' for r in selected),
            ci_losses=sum(r['result']=='S2 slower' for r in selected),inconclusive=sum(r['result']=='Inconclusive' for r in selected),
            geomean_speedup=math.exp(statistics.mean(math.log(r['speedup']) for r in selected)))
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),shape_count=140,total_candidate_input_outcomes=outcomes,
        headline_rows=len(details),samples_per_headline=90,identity=identities[0],comparisons=comparisons,
        fastest_mean_counts=dict(Counter(r['fastest_mean_method'] for r in rows)),source_snapshots=sources,
        scope='Exactly MAIN-01 through MAIN-10 confirmation. Synthetic Gaussian operands; first 140 of 168 shapes, ordered by increasing volume. Excludes all LLM shape/model phases.',
        timing='Complete device call including padding, multiplication and crop; excludes host transfer and compilation. Arithmetic mean of 3 fresh inputs by 30 paired rounds.',
        uncertainty='Pointwise 95% hierarchical paired bootstrap, 4000 replicates. Tuned Native interval reused from frozen statistics; other S2 pairs recomputed from paired rounds. No multiplicity correction or allocation-to-allocation uncertainty.',
        selection='Screen-selected configuration remains frozen. Lowest mean across algorithm families is descriptive, not an independently qualified selector. Tile alternatives do not establish global optimality.',
        accuracy='Worst of 3 fresh inputs versus host FP64 multiplication of exact BF16 inputs. Reference is full or sampled output (all K), as labeled. Relative L2 gate is 0.02. FP32 outputs.',
        labels=LABELS)
    dump(out/'results.json',dict(summary=summary,shapes=rows,algorithms=details,pairwise=pairs))
    flat=[]
    for r in rows:
        flat.append(dict(order=r['order'],m=r['m'],k=r['k'],n=r['n'],shape_id=r['shape_id'],phase=r['phase'],
            **{m+'_ms':r['methods'][m]['mean_ms'] for m in METHODS},
            **{m+'_max_relative_l2':r['methods'][m]['max_relative_l2'] for m in METHODS},
            fastest_mean=LABELS[r['fastest_mean_method']],s2_vs_native=r['comparisons']['native']['speedup'],
            s2_vs_native_ci_low=r['comparisons']['native']['ci95'][0],s2_vs_native_ci_high=r['comparisons']['native']['ci95'][1]))
    with (out/'first_140_shapes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_snapshots','identity')},indent=2))


if __name__=='__main__':main()
