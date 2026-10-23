"""Replay real frozen pilot evidence through the new reporter; no TPU timing."""
import hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_arch_study_v001 import build,recommendation
from prepare_arch_v6e_v001 import build as plan
from strassen_mm.tuner_arch_study_v001 import groups

root=Path(os.environ['STRASSEN_PROJECT_ROOT']);out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
cohort=out/'fixture';cohort.mkdir();cfg=cohort/'source/configs/arch_v6e_168_v001';cfg.mkdir(parents=True)
original=root/'runs/20260926-joint-final-v001/operations/report/artifacts/results.json'
previous=json.loads(original.read_text())
shapes=json.loads((root/'configs/joint_v001/shapes.json').read_text())
(cfg/'shapes.json').write_text(json.dumps(shapes))
(cfg/'campaign.json').write_bytes((root/'configs/joint_v001/campaign.json').read_bytes())
parents=['20260924-joint-v001']*3+['20260926-joint-continuation-v001','20260926-joint-recovery-v001','20260926-joint-final-v001']
for i,parent in enumerate(parents,1):
    for stage in ('screen','confirm'):
        (cohort/f'arch-{i:03d}-{stage}-finished.json').write_bytes((root/'runs'/parent/f'joint-{i:02d}-{stage}-finished.json').read_bytes())
assert build(cohort,out/'report')==0
result=json.loads((out/'report/results.json').read_text())
assert result['completed'] and len(result['results'])==12
for a,b in zip(result['results'],previous['results']):
    for role in ('native','s1','s2'):
        assert a['methods'][role]['mean_ms']==b['methods'][role]['mean_ms']
        assert a['methods'][role]['errors']==b['methods'][role]['errors']
        assert a['methods'][role]['comparisons']['native']==b['methods'][role]['comparisons']['native']
native=dict(eligible=True,candidate=dict(depth=0,family='native',arm_id='native'))
cubic=dict(eligible=True,candidate=dict(depth=0,family='cubic',arm_id='cubic'),comparisons=dict(native=dict(speedup=1.1,ci95=[1.09,1.11])))
rule=dict(minimum_mean_speedup=1.01,minimum_ci95_lower=1)
assert recommendation(dict(native=native,overall=cubic),rule)['status']=='custom'
p=plan(SimpleNamespace(hardware='v6e',cohort=Path('fixture'),session='fixture',endpoint='fixture',identity='fixture',controller_python='python',analysis_python='python'))
assert len([s for s in p['stages'] if s['operation']=='phase'])==337
assert len({s['id'] for s in p['stages']})==len(p['stages'])
seen=set()
for s in p['stages']:
    assert set(s.get('requires',[]))<=seen
    if s['id'].endswith('confirm'):assert s['selection_stage'] in seen
    seen.add(s['id'])
hashes={}
for name in ('kernels_v001.py','kernels_v002.py','kernels_two_level_v001.py','benchmark_mlsys_shapes_v001.py'):
    current=(root/'src/strassen_mm'/name).read_bytes();old=(root/'runs/20260921-mlsys-main-v5e-v001/source/src/strassen_mm'/name).read_bytes()
    assert current==old;hashes[name]=hashlib.sha256(current).hexdigest()
(out/'summary.json').write_text(json.dumps(dict(completed=True,pilot_replayed_contracts=12,latencies_errors_and_intervals_exact=True,cubic_can_win=True,phase_count=337,historical_v5e_unchanged_sha256=hashes),indent=2)+'\n')
