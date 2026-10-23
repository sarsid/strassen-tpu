"""Exercise the completed-report path with explicitly synthetic fixtures."""
import json
import os
from pathlib import Path
import shutil
import sys
from strassen_mm.tuner_joint_v001 import expanded, groups, select
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_joint_v001 import build


out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
cohort=out/'synthetic_fixture';config=cohort/'source/configs/joint_v001';config.mkdir(parents=True)
for name in ('campaign.json','shapes.json','distributions.json'):
    shutil.copyfile(Path('configs/joint_v001')/name,config/name)
cfg=json.loads((config/'campaign.json').read_text());shapes=json.loads((config/'shapes.json').read_text())['shapes']
def write(p,v):p.write_text(json.dumps(v)+'\n')
for index,s in enumerate(shapes,1):
    screen=groups(cfg,[s],'screen');rows=[]
    for g in screen:
        for a in g['arms']:
            speed=1.0 if not a['depth'] else .90+.02*a['depth']+(.01 if a['accumulator']=='products' else 0)
            rows.append(dict(event='case_result',shape_id=g['shape']['id'],group_id=g['group_id'],arm_id=a['arm_id'],
                status='ok',eligible_for_speedup_claim=True,timing=dict(mean_ms=speed,sample_count=7),correctness=dict(relative_l2=.004)))
    selections=select(rows,cfg,expanded([s]))
    plans=groups(cfg,[s],'confirm',selections)
    for stage in ('screen','confirm'):
        run=cohort/f'fixture-{index:02d}-{stage}';artifacts=run/'artifacts';artifacts.mkdir(parents=True)
        write(cohort/f'joint-{index:02d}-{stage}-finished.json',dict(run=str(run)))
        if stage=='screen':
            write(artifacts/'selections.json',dict(selected=selections))
            events=rows
        else:
            write(artifacts/'planned_cases.json',plans)
            write(artifacts/'environment.json',dict(identity=dict(test_fixture=True,device_kind='FAKE; NOT A TPU MEASUREMENT')))
            events=[]
            for g in plans:
                for a in g['arms']:
                    # BF16 custom recommendations fail the practical speed margin.
                    speed=1.0 if not a['depth'] else (0.92 if g['shape']['output_dtype']=='float32' else 0.995)
                    for inp in g['inputs']:
                        events.append(dict(event='case_result',shape_id=g['shape']['id'],arm_id=a['arm_id'],
                            status='ok',eligible_for_speedup_claim=True,correctness=dict(relative_l2=.004,max_abs_error=.02,rmse=.004,mean_abs_error=.003,p50_abs_error=.002,p99_abs_error=.01,normwise_error=.0001)))
                        events.extend(dict(event='sample',shape_id=g['shape']['id'],arm_id=a['arm_id'],seed=inp['seed'],round=j,elapsed_ms=speed) for j in range(30))
        (artifacts/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
report=out/'synthetic_report';assert build(cohort,report)==0
data=json.loads((report/'results.json').read_text());assert data['completed'] and len(data['results'])==12
for r in data['results']:
    expected='custom' if r['shape']['output_dtype']=='float32' else 'native'
    assert r['recommendation']['status']==expected
    for depth in (1,2):
        assert f's{depth}_other_accumulator' in r['methods']
        assert f's{depth}_other_buffers' in r['methods']
assert len(json.loads((report/'candidate_decisions.json').read_text()))==12
assert len(json.loads((report/'tuner_policy.json').read_text())['entries'])==12
assert 'Tuner design choices' in (report/'TUNER_DESIGN.md').read_text()
write(out/'summary.json',dict(completed=True,synthetic_fixture_only=True,precision_groups=12,
    checks=['full_report_path','all_decision_ledgers','separate_precision_policies','margin_fallback','matched_counterfactuals','design_document']))
print('Synthetic report validation passed; no performance measurements.')
