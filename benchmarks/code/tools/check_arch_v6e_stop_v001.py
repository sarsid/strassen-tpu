"""CPU lifecycle checks: finish v6e, release, stop, and never allocate v5e."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import supervise_arch_studies_v004 as module
from run_region_cohort_v001 import write, sha


out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
checks = []


def fixture(name):
    root = out / name
    folder = root / 'supervisor'
    folder.mkdir(parents=True)
    (root / 'results/v6e').mkdir(parents=True)
    write(folder / 'supervisor-state.json', dict(hardware='v6e',
        runtime=dict(hardware='v6e', endpoint='owned'), current_cohort=None, finished=False))
    write(folder / 'history.json', dict(cohorts=[]))
    return root, module.Supervisor(folder)


root, supervisor = fixture('successful-final-report')
events = []


def release():
    events.append('release')
    supervisor.state['runtime'] = None


def report(label, command, timeout):
    events.append('report')
    dest = Path(command[command.index('--output-dir') + 1])
    dest.mkdir(parents=True)
    write(dest / 'results.json', dict(completed=True))
    return 0, '', dest.parent


with patch.object(module, 'ROOT', root), patch.object(module, 'RESULTS', {'v6e':'results/v6e'}), \
     patch.object(module, 'commit'), patch.object(supervisor, 'release', side_effect=release), \
     patch.object(supervisor, 'call', side_effect=report):
    supervisor.finish_hardware()
assert events == ['release', 'report', 'release']
assert supervisor.state['hardware'] == 'v6e' and supervisor.state['finished']
assert supervisor.state['runtime'] is None and supervisor.state['automatic_followups'] == []
assert supervisor.state['completed_hardware'] == ['v6e']
assert (root / 'results/v6e/COMPLETE.json').exists()
checks.append('successful_report_releases_and_finishes_v6e_without_transition')

root, supervisor = fixture('failed-report-then-retry')
events = []
calls = 0


def report_retry(label, command, timeout):
    global calls
    calls += 1
    if calls == 1:
        events.append('report_failed')
        return 1, '', root
    return report(label, command, timeout)


def refresh():
    supervisor.complete = {'only-unit': {}}


def wait(seconds, detail):
    assert supervisor.state['hardware'] == 'v6e' and not supervisor.state.get('finished')
    assert supervisor.state['runtime'] is None
    events.append('wait_for_report_retry')


with patch.object(module, 'ROOT', root), patch.object(module, 'RESULTS', {'v6e':'results/v6e'}), \
     patch.object(module, 'commit'), patch.object(module, 'check_contract'), \
     patch.object(module, 'units', return_value=['only-unit']), \
     patch.object(module.subprocess, 'Popen', return_value=MagicMock(pid=999999)), \
     patch.object(supervisor, 'release', side_effect=release), \
     patch.object(supervisor, 'call', side_effect=report_retry), \
     patch.object(supervisor, 'refresh_completed', side_effect=refresh), \
     patch.object(supervisor, 'publish') as publish, \
     patch.object(supervisor, 'wait', side_effect=wait), \
     patch.object(supervisor, 'ensure_runtime') as allocate:
    supervisor.run()
    allocate.assert_not_called()
    assert 'no subsequent studies queued' in publish.call_args.args[0]
assert events == ['release', 'report_failed', 'wait_for_report_retry', 'release', 'report', 'release']
done = json.loads((supervisor.folder / 'completion.json').read_text())
assert done['completed'] and done['scope'] == 'v6e_only'
assert done['hardware'] == ['v6e'] and done['automatic_followups'] == []
checks.append('failed_final_report_retries_without_allocating_or_advancing')
checks.append('run_loop_exits_and_records_v6e_only_completion')

root, supervisor = fixture('existing-final-marker')
dest = root / 'results/v6e'
(dest / 'report').mkdir()
write(dest / 'report/results.json', dict(completed=True))
write(dest / 'COMPLETE.json', dict(results_sha256=sha(dest / 'report/results.json')))
with patch.object(module, 'ROOT', root), patch.object(module, 'RESULTS', {'v6e':'results/v6e'}), \
     patch.object(supervisor, 'release', side_effect=release), patch.object(supervisor, 'call') as report_call:
    supervisor.finish_hardware()
    report_call.assert_not_called()
assert supervisor.state['finished'] and supervisor.state['hardware'] == 'v6e'
checks.append('sealed_final_report_finishes_without_another_report_or_hardware')

for name, field in [('wrong-study', 'study'), ('wrong-runtime', 'runtime')]:
    root, supervisor = fixture(name)
    if field == 'study':
        supervisor.state['hardware'] = 'v5e'
    else:
        supervisor.state['runtime']['hardware'] = 'v5e'
    with patch.object(supervisor, 'assignments') as assignments:
        try:
            supervisor.ensure_runtime()
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-scope hardware was accepted')
        assignments.assert_not_called()
    checks.append(name + '_rejected_before_provider_request')

write(out / 'summary.json', dict(completed=True, checks=checks, measurement_code_changed=False))
print(json.dumps(dict(completed=True, checks=checks)))
