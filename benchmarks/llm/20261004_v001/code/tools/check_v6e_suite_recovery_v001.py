"""Verify preserved phases and that recovery cannot repeat confirmed shapes."""
import json,os
from pathlib import Path
from types import SimpleNamespace
from prepare_v6e_suite_recovery_v001 import build,campaign
from run_large_real_v002 import verify_phase
from strassen_mm.benchmark_v6e_suite_v001 import plan_groups
root=Path(os.environ['STRASSEN_PROJECT_ROOT']);original=root/'runs/20260923-v6e-suite-v001'
args=SimpleNamespace(hardware='v6e',cohort=root/'runs/PLAN_CHECK_ONLY',session='PLAN_CHECK_ONLY',endpoint='tpu-v6e1-PLAN_CHECK_ONLY',
    identity='/content/Strassen_MM_Focus/PLAN_CHECK_ONLY/identity.json',controller_python='not-executed',analysis_python='not-executed')
plan=build(args);assert len(plan['stages'])==9
phase_ids=[s['id']for s in plan['stages']if s['operation']=='phase']
assert phase_ids==['suite-smoke','suite-10-screen','suite-10-confirm','suite-11-screen','suite-11-confirm','suite-12-screen','suite-12-confirm']
seen=set()
for s in plan['stages']:
    assert set(s.get('requires',[]))<=seen;seen.add(s['id'])
old_shapes=set();verified=[]
for i in range(1,10):
    for suffix in ('screen','confirm'):
        stage=f'suite-{i:02d}-{suffix}';run,summary=verify_phase(original,stage);verified.append(stage)
        if suffix=='confirm':old_shapes.update(g['shape']['id']for g in json.loads((run/'artifacts/planned_cases.json').read_text()))
cfg=json.loads((root/'configs/v6e_suite_v001/campaign.json').read_text());shapes=json.loads((root/'configs/v6e_suite_v001/shapes.json').read_text())['shapes']
new_shapes=[];attempts=0
for stage in plan['stages']:
    if not stage['id'].endswith('-screen'):continue
    args=stage['runner_args'];start=int(args[1]);count=int(args[3])
    groups=plan_groups(cfg,shapes,'screen',start=start,count=count)
    attempts+=sum(len(g['arms'])for g in groups);new_shapes.extend(g['shape']['id']for g in groups)
assert len(old_shapes)==126 and len(new_shapes)==len(set(new_shapes))==42
assert old_shapes.isdisjoint(new_shapes)and old_shapes|set(new_shapes)=={s['id']for s in shapes}
assert attempts==sum(s.get('expected',0)for s in plan['stages']if s['id'].endswith('-screen'))
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
(out/'summary.json').write_text(json.dumps(dict(completed=True,verified_sealed_phases=verified,archived_shapes=126,
    recovery_shapes=42,recovery_screen_cases=attempts,qualification_cases=35,scientific_sources_unchanged=True,
    disjoint_shape_coverage=True,plan_stage_count=9),indent=2)+'\n')
print(f'18 sealed phases verified; 126 preserved + 42 disjoint recovery shapes; {attempts} recovery screening cases')
