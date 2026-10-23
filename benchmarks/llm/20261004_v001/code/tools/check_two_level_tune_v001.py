"""Check tuning/confirmation plan invariants and selection before device work."""
import copy
import json
import os
from pathlib import Path
from strassen_mm.benchmark_two_level_tune_v001 import make_groups, select_tiles


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/two_level_tune_v001/campaign.json').read_text())
    shapes=json.loads((root/'configs/two_level_tune_v001/shapes.json').read_text())['shapes']
    groups=make_groups(cfg,shapes,'screen');checks=0
    assert len(groups)==16 and groups==make_groups(cfg,shapes,'screen');checks+=1
    for shape in shapes:
        gg=[g for g in groups if g['shape']['id']==shape['id']]
        assert sorted(g['tile'] for g in gg)==sorted(cfg['tiles']);checks+=1
        assert len({g['inputs'][0]['seed'] for g in gg})==1;checks+=1
    for g in groups:
        assert len(g['arms'])==2 and all(a['tile']==g['tile'] for a in g['arms']);checks+=1
        assert all(n%d==0 for n,d in zip(g['tile'],(32,512,512)));checks+=1
    cases=[]
    for g in groups:
        rank=cfg['tiles'].index(g['tile'])
        for arm in cfg['arms']:
            # Fastest-looking tile fails eligibility; next fastest has incomplete timings.
            cases.append(dict(shape_id=g['shape']['id'],arm_id=arm,group_id=g['group_id'],
                status='numerical_failure' if rank==0 else 'ok',eligible_for_speedup_claim=rank!=0,
                timing=dict(mean_ms=rank+1,sample_count=9 if rank==1 else 10),
                kernel_metadata=dict(tile_bm_bn_bk=g['tile']),correctness=dict(relative_l2=.01)))
    selected=select_tiles(cases,cfg,shapes)
    assert all(s[a]['tile']==cfg['tiles'][2] for s in selected.values() for a in cfg['arms']);checks+=1
    confirm=make_groups(cfg,shapes,'confirm',selected)
    assert len(confirm)==2 and sum(len(g['inputs'])*len(g['arms']) for g in confirm)==12;checks+=1
    screen_seeds={i['seed'] for g in groups for i in g['inputs']}
    confirm_seeds={i['seed'] for g in confirm for i in g['inputs']}
    assert len(confirm_seeds)==6 and screen_seeds.isdisjoint(confirm_seeds);checks+=1
    invalid=copy.deepcopy(cases)
    for r in invalid:r['status']='oom';r['eligible_for_speedup_claim']=False
    try:select_tiles(invalid,cfg,shapes)
    except ValueError:checks+=1
    else:raise AssertionError('No-eligible-tile selection must fail')
    result=dict(passed=True,checks=checks,screen_groups=16,screen_outcomes=32,confirmation_groups=2,confirmation_outcomes=12,
                scope='Offline plan and selection checks; unchanged kernels retain previous algebra validation')
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
