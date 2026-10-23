"""Verify the v5e extension preserves the historical experimental contract."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_arch_v5e_v001 import build
from strassen_mm.benchmark_mlsys_shapes_v001 import plan_groups,validate_campaign

root=Path(os.environ['STRASSEN_PROJECT_ROOT']);out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
original=json.loads(Path('configs/mlsys_shapes_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/mlsys_shapes_v001/shapes.json').read_text())['shapes']
counts={}
for suffix,dtype in [('fp32','float32'),('bf16','bfloat16')]:
    path=Path('configs/arch_v5e_168_'+suffix+'_v001');cfg=json.loads((path/'campaign.json').read_text())
    assert cfg['precision']['output_dtype']==dtype
    for key in original:
        if key not in ('campaign_id','precision'):assert cfg[key]==original[key],key
    assert {k:v for k,v in cfg['precision'].items() if k not in ('output_dtype','output_adapter')}=={k:v for k,v in original['precision'].items() if k!='output_dtype'}
    for name in ('shapes.json','distributions.json'):assert (path/name).read_bytes()==Path('configs/mlsys_shapes_v001',name).read_bytes()
    validate_campaign(cfg,shapes)
    groups=plan_groups(cfg,shapes,'screen');counts[dtype]=sum(len(g['arms']) for g in groups)
    assert counts[dtype]==168*52
p=build(SimpleNamespace(hardware='v5e',cohort=Path('fixture'),session='fixture',endpoint='fixture',identity='fixture',controller_python='python',analysis_python='python'))
assert len([s for s in p['stages'] if s['operation']=='phase'])==50
seen=set()
for s in p['stages']:
    assert set(s.get('requires',[]))<=seen
    if s['operation']=='phase' and s['id'].endswith('confirm'):assert s['selection_stage'] in seen
    seen.add(s['id'])
assert len(seen)==len(p['stages'])
assert p['transport']=='tools/run_phase_v004.py' and p['hardware']=='v5e'
(out/'summary.json').write_text(json.dumps(dict(completed=True,unchanged_historical_contract=True,screen_outcomes=counts,phase_count=50,source='No v6e tuning or compiler parameters imported'),indent=2)+'\n')
