"""Verify ten historical choices, complete paired groups and fresh confirmation."""
import json,os
from pathlib import Path
from strassen_mm import benchmark_v6e_ten_v001 as ten
b=ten.benchmark
cfg=json.loads(Path('configs/v6e_ten_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/v6e_ten_v001/shapes.json').read_text())['shapes']
history=json.loads(Path('configs/v6e_ten_v001/v5e_evidence.json').read_text())['selected']
assert [s['id']for s in shapes]==[r['shape_id']for r in history]
assert len(shapes)==10
for row in history:
 for compare in ('one_level_comparisons','comparisons'):
  for baseline in ('native','native_default'):assert row[compare][baseline]['ci95'][0]>1
screen=b.groups(cfg,shapes,'screen',None);assert len(screen)==10
for g in screen:
 assert len(g['arms'])==21 and len(set(a['arm_id']for a in g['arms']))==21
 assert all(a.get('depth',0)in(0,1,2)for a in g['arms'])
selection={s['id']:{family:dict(candidate=dict(c[0],arm_id=c[0]['candidate_id'],family=family),status='eligible')for family,c in cfg['candidate_families'].items()}for s in shapes}
confirm=b.groups(cfg,shapes,'confirm',selection)
assert len(confirm)==10 and all(len(g['inputs'])==3 for g in confirm)
assert set(cfg['confirm_seeds']).isdisjoint([cfg['screen_seed']])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
(out/'summary.json').write_text(json.dumps(dict(completed=True,shapes=10,screen_cases=210,confirmation_seeds=3,paired_rounds=30,depths=[1,2]),indent=2)+'\n')
print('Ten shapes have both historical wins; 210 paired screen cases; fresh 3x30 confirmation plan verified')
