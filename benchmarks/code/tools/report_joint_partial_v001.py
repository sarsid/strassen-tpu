"""Finalize the verified three-shape subset after the user's quota shutdown."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from report_joint_v001 import build
from run_large_real_v002 import verify_phase

root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
cohort = root / 'runs/20260924-joint-v001'
release = json.loads((cohort / 'operations/quota-closeout-v001/release.log').read_text())
assert release['verified_absent'] is True
assert release['endpoint'] == json.loads((cohort / 'plan.json').read_text())['allocation_id']
summaries = {}
for index in (1, 2, 3):
    for phase in ('screen', 'confirm'):
        name = f'joint-{index:02d}-{phase}'
        _, summaries[name] = verify_phase(cohort, name)
out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
assert build(cohort, out) == 1  # Expected: the original six-shape study is incomplete.
data = json.loads((out / 'results.json').read_text())
assert len(data['results']) == 6 and set(data['missing']) == {'tall', 'k12288', 'large_cube'}
banner = ('# Partial study: three of six shapes completed\n\n'
          'Stopped at the user’s request to preserve token quota for cleanup and documentation. '
          'The TPU was released and verified absent before this report was generated. '
          'The tall shape, K=12288 shape and 16384 cube were not run. '
          'They are missing evidence, not algorithm failures.\n\n')
for name in ('RESULTS.md', 'TUNER_DESIGN.md'):
    path = out / name
    path.write_text(banner + path.read_text())
policy_path = out / 'tuner_policy.json'
policy = json.loads(policy_path.read_text())
policy.update(full_campaign_completed=False, confirmed_shapes=3, planned_shapes=6,
              missing_shapes=data['missing'], stop_reason='user_token_budget')
policy_path.write_text(json.dumps(policy, indent=2) + '\n')
state = json.loads((cohort / 'progress.json').read_text())
for stage in state['stages']:
    if stage['id'] == 'joint-03-confirm':
        receipt = json.loads((cohort / 'joint-03-confirm-finished.json').read_text())
        events = [json.loads(line) for line in (Path(receipt['run']) / 'artifacts/results.jsonl').read_text().splitlines()]
        cases = [r for r in events if r['event'] == 'case_result']
        stage.update(state='succeeded', completed=len(cases), expected=len(cases),
                     succeeded=sum(r['status'] == 'ok' for r in cases), failed=sum(r['status'] != 'ok' for r in cases))
    elif stage['id'] == 'release':
        stage.update(state='succeeded', detail='Released during user-requested quota closeout; verified absent.')
    elif stage['id'] == 'report':
        stage.update(state='succeeded', detail='Partial three-shape report saved.',
                     evidence=[dict(label='Partial results', path=str((out / 'RESULTS.md').relative_to(root)))])
    elif stage['state'] in ('running', 'not_started'):
        stage.update(state='cancelled', detail='Unrun: stopped at user token-budget request.')
state.update(state='cancelled', stage='quota-closeout', allocation_released=True,
             remote_execution_uncertain=False, updated_utc=datetime.now(timezone.utc).isoformat(),
             detail='Stopped at user request after 3/6 shapes; TPU released and partial report saved.',
             next_step='Review the saved results; remaining three shapes are deferred.')
(cohort / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')
closeout = dict(status='closed_partial', allocation_released=True, verified_absent=True,
                verified_confirmed_shapes=3, planned_shapes=6, missing_shapes=data['missing'],
                report=str(out), full_experiment_completed=False, utc=datetime.now(timezone.utc).isoformat())
(cohort / 'user-closeout-v001.json').write_text(json.dumps(closeout, indent=2) + '\n')
(out / 'CLOSEOUT.json').write_text(json.dumps(closeout, indent=2) + '\n')
print(json.dumps(closeout))
