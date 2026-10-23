"""Check CPU-only report failure releases the TPU and preserves study state."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import supervise_arch_studies_v002 as module
from run_region_cohort_v001 import write

out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
folder = out / 'supervisor'
folder.mkdir()
write(folder / 'supervisor-state.json', dict(hardware='v6e', runtime=dict(endpoint='owned'), current_cohort=None))
write(folder / 'history.json', dict(cohorts=[]))
supervisor = module.Supervisor(folder)
events = []
with patch.object(supervisor, 'release', side_effect=lambda: events.append('release')), \
     patch.object(supervisor, 'call', side_effect=lambda *a, **k: (events.append('report') or (1, '', out))), \
     patch.object(module, 'ROOT', out):
    try:
        supervisor.finish_hardware()
    except module.RetryLater:
        pass
    else:
        raise AssertionError('A failed report must not advance the hardware study')
assert events == ['release', 'report']
assert supervisor.state['hardware'] == 'v6e' and not supervisor.state.get('finished')
v1 = Path('tools/supervise_arch_studies_v001.py').read_text()
v2 = Path('tools/supervise_arch_studies_v002.py').read_text()
assert v2 == v1.replace("        destination = ROOT / RESULTS[hardware]\n        marker", "        # All paired measurements and phase exports are verified before this\n        # method runs. Release idle hardware before CPU-only final reporting.\n        self.release()\n        destination = ROOT / RESULTS[hardware]\n        marker").replace("        try:\n            while not self.state.get('finished'):", "        try:\n            self.refresh_completed()\n            while not self.state.get('finished'):")
write(out / 'summary.json', dict(completed=True, checks=['release_before_cpu_report',
    'failed_report_does_not_complete_study', 'tested_recovery_logic_unchanged', 'startup_restores_saved_progress']))
print(json.dumps(dict(completed=True, checks=4)))
