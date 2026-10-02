"""Check broad-shape coverage, controls, chunking, padding and fresh selection."""
import json,math,os
from pathlib import Path
from strassen_mm.benchmark_v6e_suite_v001 import plan_groups,chosen_tiles
cfg=json.loads(Path('configs/v6e_suite_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/v6e_suite_v001/shapes.json').read_text())['shapes']
assert shapes==json.loads(Path('configs/mlsys_shapes_v001/shapes.json').read_text())['shapes']
assert len(shapes)==168 and len({(s['m'],s['k'],s['n'])for s in shapes})==168
allgroups=plan_groups(cfg,shapes,'screen');chunks=[]
for i in range(12):chunks.extend(plan_groups(cfg,shapes,'screen',start=i*14,count=14))
assert chunks==allgroups
assert sum(len(g['arms'])for g in allgroups)==4992
selections={};padding=[]
for g in allgroups:
    arms=g['arms'];sid=g['shape']['id'];assert len({a['arm_id']for a in arms})==len(arms)
    assert sum(a['family']=='native'for a in arms)==5
    cubic={tuple(a['tile'])for a in arms if a['family']=='cubic'}
    for a in arms:
        assert a.get('depth',0)in(0,1,2)
        if a['tile']:
            assert tuple(a['tile'])in cubic
            assert all(v%q==0 for v,q in zip(a['tile'],(32,512,512)))
            m,k,n=[g['shape'][d]for d in ('m','k','n')];bm,bn,bk=a['tile']
            padding.append(math.ceil(m/bm)*bm*math.ceil(k/bk)*bk*math.ceil(n/bn)*bn/(m*k*n))
    selections[sid]={f:dict(candidate=next(a for a in reversed(arms)if a['family']==f),status='eligible')for f in cfg['arms']}
confirm=plan_groups(cfg,shapes,'confirm',selections)
for g in confirm:
    roles={role:a for a in g['arms']for role in a['headline_roles']}
    assert {'native','native_default','cubic','one_level','two_level','matched_one_level','matched_two_level'}<=roles.keys()
    for f in ('one_level','two_level'):assert roles[f]['tile']==roles['matched_'+f]['tile']
    assert len(g['inputs'])==3 and len({a['arm_id']for a in g['arms']})==len(g['arms'])
assert set(range(cfg['screen_seed'],cfg['screen_seed']+168)).isdisjoint({s+i for s in cfg['confirm_seeds']for i in range(168)})
smoke=plan_groups(cfg,shapes,'smoke');assert len(smoke)==5 and all(len(g['arms'])==7 for g in smoke)
# Exercise a family with no measured candidate: confirmation retains Native and other families.
sid=shapes[0]['id'];selections[sid]['two_level']={'status':'no_measured_candidate'}
fallback=plan_groups(cfg,shapes,'confirm',selections,count=1)[0]
assert not any('two_level'in a['headline_roles']for a in fallback['arms'])
assert any('native_default'in a['headline_roles']for a in fallback['arms'])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
result=dict(completed=True,shapes=168,batches=12,screen_cases=4992,qualification_cases=35,
    confirmation_case_upper_bound=168*7*3,tiles_offered=[4,6],max_offered_padding_ratio=max(padding),
    evidence='Small matrices deliberately retain visible padding cost; no shape is silently excluded.')
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
