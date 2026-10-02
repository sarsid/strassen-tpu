"""Regression-check the new reporter against the completed ten-shape experiment.

This creates an explicitly labeled analysis fixture, never new TPU measurements.
"""
import json,os,shutil
from pathlib import Path
from report_v6e_suite_v001 import build,classify,paired
root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
fixture=out/'analysis_fixture_not_new_measurements';fixture.mkdir()
old=root/'runs/20260923-v6e-ten-v001'
known=json.loads((root/'runs/20260923T184750Z-v6e-ten-summary-v001-03f334/artifacts/summary.json').read_text())
def save(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2)+'\n')
cfgpath=fixture/'source/configs/v6e_suite_v001'
for name in ('campaign.json','shapes.json'):
 save(cfgpath/name,json.loads((old/'source/configs/v6e_ten_v001'/name).read_text()))
for stage in ('screen','confirm'):
 source=Path(json.loads((old/('opt-'+stage+'-finished.json')).read_text())['run'])/'artifacts'
 dest=fixture/('phase-'+stage)/'artifacts';dest.mkdir(parents=True)
 shutil.copy2(source/'results.jsonl',dest/'results.jsonl')
 if stage=='screen':
  original=json.loads((source/'selections.json').read_text());selected={}
  for sid,choice in original['selected'].items():
   selected[sid]={k:choice[k]for k in ('native','cubic')}
   for depth,role in (('1','one_level'),('2','two_level')):
    row=next(r for r in known['rows']if r['shape_id']==sid)
    selected[sid][role]=choice[row['methods'][depth]['family']]
  save(dest/'selections.json',dict(selected=selected))
 else:
  groups=json.loads((source/'planned_cases.json').read_text())
  for g in groups:
   choices=selected[g['shape']['id']]
   for a in g['arms']:
    a['headline_roles']=[k for k,v in choices.items()if v['candidate']['arm_id']==a['arm_id']]
    if a['arm_id']=='native_default':a['headline_roles'].append('native_default')
  save(dest/'planned_cases.json',groups)
 save(fixture/('suite-01-'+stage+'-finished.json'),dict(run=str(dest.parent)))
profile=root/'runs/20260923T184803Z-v6e-ten-device-profiles-v001-e3afda/artifacts/profiles.json'
save(fixture/'operations/device-analysis/artifacts/profiles.json',json.loads(profile.read_text()))
code=build(fixture,out/'regression_report');assert code==1 # Ten-shape fixture is explicitly incomplete for a 168-shape report.
reported=json.loads((out/'regression_report/results.json').read_text())
assert len(reported['results'])==10 and not reported['completed']
for r in reported['results']:
 expected=next(x for x in known['rows']if x['shape_id']==r['shape_id'])
 for depth,role in (('1','one_level'),('2','two_level')):
  m=r['methods'][role];oldmethod=expected['methods'][depth]
  assert abs(m['comparisons']['native']['speedup']-oldmethod['vs_tuned']['speedup'])<1e-12
  assert m['errors']['relative_l2']==oldmethod['worst_errors']['relative_l2']
assert reported['summary']['all']['one_level']['counts']=={'inconclusive':1,'win':9}
assert reported['summary']['all']['two_level']['counts']=={'loss':4,'win':5,'inconclusive':1}
assert classify({'eligible':False,'comparisons':{'native':{'eligible':True,'ci95':[2,3]}}})=='unavailable_or_ineligible'
assert paired({(0,0):2},{(0,0):1})is None
save(out/'summary.json',dict(completed=True,regression_shapes=10,existing_speedups_and_errors_reproduced=True,qualified_win_counts_reproduced=True,
    incomplete_campaign_and_invalid_measurement_gates=True,new_tpu_measurements=0))
print('Historical ten-shape regression, missing-coverage and ineligible-result checks passed')
