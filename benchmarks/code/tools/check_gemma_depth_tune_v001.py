"""Check frozen Gemma search coverage and selection failure handling offline."""
import copy
import json
import os
from pathlib import Path
from strassen_mm.benchmark_gemma_depth_tune_v001 import candidates,make_groups,select_candidates


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/gemma_depth_tune_v001/campaign.json').read_text())
    shapes=json.loads((root/'configs/gemma_depth_tune_v001/shapes.json').read_text())['shapes']
    cc=candidates(cfg);gg=make_groups(cfg,shapes,'screen');checks=0
    assert len(gg)==9 and len(cc)==20 and gg==make_groups(cfg,shapes,'screen');checks+=1
    assert sorted(a['candidate_id'] for g in gg for a in g['arms'])==sorted(c['candidate_id'] for c in cc);checks+=1
    assert len({g['inputs'][0]['seed'] for g in gg})==1;checks+=1
    cases=[]
    for family in cfg['arms']:
        group=[c for c in cc if c['family']==family]
        for rank,c in enumerate(group):
            # Exclude the fastest failed candidate and the next incomplete one.
            cases.append(dict(shape_id=shapes[0]['id'],arm_id=c['candidate_id'],group_id='g',
                status='numerical_failure' if rank==0 else 'ok',eligible_for_speedup_claim=rank!=0,
                timing=dict(mean_ms=rank+1,sample_count=9 if rank==1 else 10),correctness=dict(relative_l2=.01)))
    selected=select_candidates(cases,cfg,shapes)
    for family in cfg['arms']:
        assert selected[shapes[0]['id']][family]['candidate']==[c for c in cc if c['family']==family][2];checks+=1
    confirm=make_groups(cfg,shapes,'confirm',selected)
    assert len(confirm)==1 and len(confirm[0]['arms'])==4 and len(confirm[0]['inputs'])==3;checks+=1
    assert {i['seed'] for i in confirm[0]['inputs']}.isdisjoint({g['inputs'][0]['seed'] for g in gg});checks+=1
    invalid=copy.deepcopy(selected);invalid[shapes[0]['id']]['two_level']['candidate']['tile']=[32,512,512]
    try:make_groups(cfg,shapes,'confirm',invalid)
    except ValueError:checks+=1
    else:raise AssertionError('Out-of-grid confirmation accepted')
    for c in cases:c['eligible_for_speedup_claim']=False
    try:select_candidates(cases,cfg,shapes)
    except ValueError:checks+=1
    else:raise AssertionError('No eligible candidate accepted')
    result=dict(passed=True,checks=checks,screen_groups=9,screen_candidates=20,confirmation_outcomes='9 if default Native selected, else 12',new_kernel_code=False)
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
