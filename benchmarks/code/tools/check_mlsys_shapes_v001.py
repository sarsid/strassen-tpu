"""Offline fault checks for the shape inventory, fair search and fresh selection."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from strassen_mm import benchmark_mlsys_shapes_v001 as study


def main():
    root=Path(__file__).resolve().parents[1];checks=0;counts={}
    for name,expected in [('mlsys_shapes_v001',168),('mlsys_llm_shapes_v001',18)]:
        cfg=json.loads((root/'configs'/name/'campaign.json').read_text())
        shapes=json.loads((root/'configs'/name/'shapes.json').read_text())['shapes']
        assert len(shapes)==expected;checks+=1
        assert study.validate_campaign(cfg,shapes)==dict(native=4,cubic=16,one_level=16,two_level=16);checks+=1
        full=study.plan_groups(cfg,shapes,'screen')
        assert len(full)==expected*16 and sum(len(g['arms'])for g in full)==52*expected;checks+=1
        assert full==study.plan_groups(cfg,shapes,'screen');checks+=1
        chunks=[]
        for start in range(0,len(shapes),14):chunks.extend(study.plan_groups(cfg,shapes,'screen',start=start,count=14))
        assert chunks==full;checks+=1
        for s in shapes:
            groups=[g for g in full if g['shape']['id']==s['id']]
            assert len({i['seed']for g in groups for i in g['inputs']})==1;checks+=1
        short_shapes=shapes[:2];screen=study.plan_groups(cfg,short_shapes,'screen');cases=[]
        for group in screen:
            for arm in group['arms']:
                family_candidates=cfg['candidate_families'][arm['family']]
                rank=next(i for i,c in enumerate(family_candidates)if c['candidate_id']==arm['candidate_id'])
                cases.append(dict(shape_id=group['shape']['id'],group_id=group['group_id'],arm_id=arm['arm_id'],
                    scope='call',status='numerical_failure'if rank==0 else 'ok',eligible_for_speedup_claim=rank!=0,
                    timing=dict(mean_ms=1+rank*.005,sample_count=9 if rank==1 else 10),
                    correctness=dict(relative_l2=.1 if rank==0 else .001,finite=True)))
        selected=study.select_candidates(cases,cfg,short_shapes)
        for shape in short_shapes:
            for family in study.FAMILIES:
                row=selected[shape['id']][family]
                assert row['winner']['candidate_id']==cfg['candidate_families'][family][2]['candidate_id'];checks+=1
                assert len(row['confirmation_candidates'])<=4 and row['eligible'];checks+=1
        confirmed=study.plan_groups(cfg,short_shapes,'confirm',selected)
        assert len(confirmed)==2 and all(len(g['arms'])<=17 for g in confirmed);checks+=1
        assert all(any(a['candidate_id']=='native_default'for a in g['arms'])for g in confirmed);checks+=1
        assert {i['seed']for g in screen for i in g['inputs']}.isdisjoint({i['seed']for g in confirmed for i in g['inputs']});checks+=1
        assert all(len(g['inputs'])==3 for g in confirmed);checks+=1
        for invalid in [cases[:-1],cases+[cases[0]]]:
            try:study.select_candidates(invalid,cfg,short_shapes)
            except ValueError:checks+=1
            else:raise AssertionError('Missing/duplicate candidate result must fail')
        failed=copy.deepcopy(cases)
        for r in failed:r.update(status='oom',eligible_for_speedup_claim=False,timing=None,correctness=None)
        unavailable=study.select_candidates(failed,cfg,short_shapes)
        assert all(x[f]['winner']is None for x in unavailable.values()for f in study.FAMILIES);checks+=1
        assert all(len(g['arms'])==1 for g in study.plan_groups(cfg,short_shapes,'confirm',unavailable));checks+=1
        finite_bad=copy.deepcopy(cases)
        for r in finite_bad:r.update(status='numerical_failure',eligible_for_speedup_claim=False)
        fallback=study.select_candidates(finite_bad,cfg,short_shapes)
        assert all(x[f]['winner']is not None and not x[f]['eligible']for x in fallback.values()for f in study.FAMILIES);checks+=1
        bad=copy.deepcopy(cfg);bad['candidate_families']['two_level'][0]['tile']=[16,256,256]
        try:study.validate_campaign(bad,shapes)
        except ValueError:checks+=1
        else:raise AssertionError('Misaligned two-level tile accepted')
        counts[name]=dict(shapes=expected,screen_groups=len(full),screen_outcomes=52*expected,
            maximum_confirmation_outcomes=expected*17*3)
    numpy_checks='not available'
    try:
        import numpy as np
        study.base.np=np
        cfg=json.loads((root/'configs/mlsys_shapes_v001/campaign.json').read_text())
        cache=study.install_reference_policy(cfg)
        a=np.asarray([[1.,2.,3.],[4.,5.,6.]],dtype=np.float32);b=np.asarray([[1.,2.],[3.,4.],[5.,6.]],dtype=np.float32)
        ref,rows,cols,info=study.base.make_reference(a,b,cfg,77)
        assert ref.dtype==np.float64 and np.array_equal(ref,[[22,28],[49,64]])and info['reference_scope']=='full';checks+=1
        assert study.base.make_reference(a,b,cfg,77)[0]is ref;checks+=1
        changed=a.copy();changed[0,0]=2
        assert not np.array_equal(study.base.make_reference(changed,b,cfg,77)[0],ref);checks+=1
        g=dict(group_id='paired',inputs=[dict(seed=i)for i in range(3)])
        sample={}
        for i in range(3):
            sample[('paired','ref',i)]=[dict(round=r,elapsed_ms=10+r*.1+i)for r in range(30)]
            sample[('paired','candidate',i)]=[dict(round=r,elapsed_ms=(10+r*.1+i)*.8)for r in range(30)]
        interval=study.comparison_interval(SimpleNamespace(samples=sample),g,'ref','candidate',cfg['selection_policy'])
        assert abs(interval['candidate_over_reference_latency_ratio']-.8)<1e-12 and all(abs(x-.8)<1e-12 for x in interval['ci95']);checks+=1
        sample[('paired','candidate',0)].pop()
        assert study.comparison_interval(SimpleNamespace(samples=sample),g,'ref','candidate',cfg['selection_policy'])is None;checks+=1
        cache.clear();numpy_checks='passed'
    except ImportError:pass
    result=dict(passed=True,checks=checks,inventory=counts,numpy_reference_and_bootstrap_checks=numpy_checks,
        scope='Offline planning, failure handling, seed slicing, reference and interval invariants. Does not claim TPU execution or hardware performance.')
    target=os.environ.get('STRASSEN_EXECUTION_DIR')
    if target:
        out=Path(target)/'artifacts';out.mkdir(exist_ok=True)
        with (out/'summary.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
