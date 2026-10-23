"""Regression tests for shape counts and restoring a running cohort's progress."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import supervise_arch_studies_v003 as module
from run_region_cohort_v001 import read, write

out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
folder = out / 'supervisor'
folder.mkdir()
write(folder / 'supervisor-state.json', dict(hardware='v6e', current_cohort=None))
write(folder / 'history.json', dict(cohorts=[]))
s = module.Supervisor(folder)
s.complete = {f'arch-{i:03d}': {} for i in range(1, 61)}
s.baseline = [dict(id='saved-arch-060-confirm', completed=63)]
s.baseline_results = [dict(stage='arch-060-confirm', candidate_id='saved')]
stages = [dict(id=prefix + phase, kind='analysis' if prefix else 'measurement', state=state)
    for phase, state in [('arch-060-confirm', 'succeeded'), ('arch-061-confirm', 'succeeded'), ('arch-062-confirm', 'running')]
    for prefix in ('', 'export-')]
rows = [dict(stage='arch-060-confirm', candidate_id='duplicate'),
    dict(stage='arch-061-confirm', candidate_id='new')]
s.publish(child=dict(stages=stages, selected_results=rows))
result = read(folder / 'progress.json')
assert '61/168' in result['title']
assert [row['candidate_id'] for row in result['selected_results']] == ['saved', 'new']
assert len(result['stages']) == 5
assert sum(stage['id'].endswith('arch-060-confirm') for stage in result['stages']) == 1

# A live cohort may be in the short gap between confirmation and export.
# Restarting the supervisor must not race its existing export process.
cohort = out / 'running-cohort'
run = cohort / 'phases/example'
(run / 'artifacts').mkdir(parents=True)
write(run / 'artifacts/summary.json', dict(case_status_counts={'ok': 7}))
(run / 'artifacts/results.jsonl').write_text('')
item = dict(run=str(run), phase='arch-061-confirm', cohort=str(cohort), evidence=str(cohort / 'arch-061-confirm-finished.json'))
s.state['current_cohort'] = str(cohort)
with patch.object(module, 'collect', return_value={'arch-061': {'confirm': item}}), \
        patch.object(s, 'call') as called:
    s.refresh_completed()
    assert not called.called
s.state.update(hardware='v5e', current_cohort=None)
s.complete = {'fp32-01': {}}
s.baseline = []
s.baseline_results = []
s.publish(child=dict(stages=[dict(id='bf16-01-confirm', kind='measurement', state='succeeded'),
    dict(id='export-bf16-01-confirm', kind='analysis', state='succeeded')]))
assert '2/24' in read(folder / 'progress.json')['title']
write(out / 'summary.json', dict(completed=True, checks=['exports_do_not_count_as_shapes',
    'restored_cohort_stages_and_rows_not_duplicated', 'live_export_is_not_restarted', 'v5e_output_blocks_count_once']))
print(json.dumps(dict(completed=True, checks=4)))
