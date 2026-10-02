"""Exercise launch-loss recovery, refusal paths, and exact continuation coverage."""
import contextlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import patch
import run_phase_v007 as transport
from run_region_cohort_v001 import write, sha
import prepare_arch_continuation_v001 as prepare
from report_arch_continuation_v001 import make_view

out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
checks = []


def exercise(name, lost=True, absent=False, wrong_source=False, download_failures=0, hardware="v6e"):
    root = out / name
    root.mkdir()
    run = root / 'test-run'
    run.mkdir()
    (run / 'source').mkdir()
    write(run / 'source/plan.json', dict(hardware=hardware))
    (run / 'source.tar').write_bytes(b'original committed source')
    payload = root / 'payload/test-run'
    (payload / 'artifacts').mkdir(parents=True)
    (payload / 'input-source.tar.gz').write_bytes((run / 'source.tar').read_bytes())
    done = dict(status='completed', run_id=run.name, phase='arch-test-confirm', allocation_id='same-endpoint')
    write(payload / 'completion.json', done)
    write(payload / 'launch.json', dict(**done, archive_sha256='wrong' if wrong_source else sha(run / 'source.tar')))
    write(payload / 'artifacts/summary.json', dict(completed=True, case_status_counts={'ok': 1}))
    package = root / 'fixture.tar.gz'
    with tarfile.open(package, 'w:gz') as packed:
        packed.add(payload, arcname=run.name)
    calls = []
    terminal = {}

    class Archive:
        utc = staticmethod(lambda: 'test-time')
        exclusive_json = staticmethod(write)

        def begin(self, *args, **kwargs):
            return run

        def finish(self, path, status, **details):
            terminal.update(status=status, **details)
            return 'test-revision'

    def response(value):
        return json.dumps(dict(kind='remote_stream', text=json.dumps(value) + '\n')) + '\n'

    def invoke(command, **kwargs):
        expected_control = 'colab_control_v003.py' if hardware == 'v6e' else 'colab_control_v002.py'
        assert Path(command[1]).name == expected_control
        op = command[2]
        calls.append(command)
        code = 0
        stdout = ''
        if op == 'exec-file':
            script = Path(command[command.index('--file') + 1]).name
            if script == 'launch_phase_v004.py':
                code = int(lost)
                stdout = 'Connection was lost.' if lost else response(dict(kind='phase_launched'))
            elif absent:
                code = 1
                stdout = 'Run directory does not exist'
            else:
                empty = dict(offset=0, next_offset=0, data_base64='', more=False)
                stdout = response(dict(kind='phase_progress', transport='base64_bytes_v002',
                    run_id=run.name, phase='arch-test-confirm', completion=done,
                    archive_ready=dict(archive='/content/Strassen_MM_Focus/test.tar.gz', sha256=sha(package)),
                    summary=dict(completed=True), log=empty, results=empty, worker_log=empty))
        elif op == 'download':
            count = sum(c[2] == 'download' for c in calls)
            if count <= download_failures:
                code = 1
                stdout = 'TransportError'
            else:
                Path(command[command.index('--local') + 1]).write_bytes(package.read_bytes())
        return SimpleNamespace(returncode=code, stdout=stdout, stderr='')

    argv = ['run_phase_v006.py', '--phase', 'arch-test-confirm', '--session', 'test',
        '--endpoint', 'same-endpoint', '--expected-identity', '/identity.json',
        '--controller-python', 'python', '--timeout-seconds', '60']
    with patch.object(transport, 'archive', Archive()), patch.object(transport.subprocess, 'run', invoke), \
            patch.object(transport.time, 'sleep', lambda seconds: None), \
            patch.object(transport, 'emit', lambda *args, **kwargs: None), \
            patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
        code = transport.main()
    launches = [c for c in calls if any('launch_phase_v004.py' in arg for arg in c)]
    assert len(launches) == 1, 'A lost acknowledgment must never cause a repeated launch'
    if absent or wrong_source or download_failures >= 5:
        assert code == 1 and terminal['remote_may_still_be_running'] is True
    else:
        assert code == 0 and terminal['remote_may_still_be_running'] is False
        assert terminal['launch_acknowledged'] == (not lost)
    checks.append(dict(check=name, passed=True, launch_count=len(launches), download_count=sum(c[2] == 'download' for c in calls)))


exercise('lost_ack_completed_worker')
exercise('acknowledged_launch', lost=False)
exercise('missing_worker_no_duplicate', absent=True)
exercise('wrong_source_rejected', wrong_source=True)
exercise('transient_download_recovered', lost=False, download_failures=2)
exercise('persistent_download_retained', lost=False, download_failures=5)
exercise('historical_v5e_controller', lost=False, hardware='v5e', download_failures=1)

from arch_resilience_v001 import BASE, collect, units, check_contract
import prepare_arch_resilient_v001 as resilient
from supervise_arch_studies_v001 import Supervisor, RetryLater
root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
history = dict(cohorts=[dict(path=str(BASE), hardware='v6e'),
    dict(path=str(root/'runs/20260927-arch-v6e-continuation-v001'), hardware='v6e')], recoveries={})
complete = collect(history, 'v6e')
assert len(complete) == 60 and 'arch-060' not in complete
assert len(units('v6e')) == 168 and len(units('v5e')) == 24
historyfile = out / 'history.json';write(historyfile, history)
args = SimpleNamespace(hardware='v6e', cohort=out/'future', history=historyfile,
    session='test', endpoint='test', identity='/identity.json', controller_python='python', analysis_python='python')
plan = resilient.build(args)
phases = [s for s in plan['stages'] if s['operation'] == 'phase']
assert len(phases) == 217 # smoke and 108 complete screen/confirm pairs
assert phases[1]['id'] == 'arch-060-screen'
assert not any(s.get('releases_allocation') for s in plan['stages'])
assert plan['automatic_recovery'] and plan['transport'] == 'tools/run_phase_v007.py'
ids = {s['id'] for s in plan['stages']}
assert all(set(s.get('requires',[])) <= ids for s in plan['stages'])
check_contract(root)
checks.append(dict(check='60_saved_108_remaining_whole_units_unchanged_contract', passed=True))

folder = out/'supervisor-fixture';folder.mkdir()
write(folder/'supervisor-state.json', dict(hardware='v6e', runtime=None, pending_request=None))
write(folder/'history.json', history)
sup = Supervisor(folder)
with patch.object(sup, 'assignments', side_effect=RetryLater('offline')), patch.object(sup, 'call') as call:
    try:sup.ensure_runtime()
    except RetryLater:pass
    else:raise AssertionError('Network outage must wait')
    assert not call.called
sup.state['pending_request'] = dict(epoch=__import__('time').time(), path='not-read', session='pending')
with patch.object(sup, 'assignments', return_value=[]), patch.object(sup, 'call') as call:
    try:sup.ensure_runtime()
    except RetryLater:pass
    else:raise AssertionError('Uncertain create must not immediately repeat')
    assert not call.called
sup.state['pending_request'] = None
with patch.object(sup, 'assignments', return_value=[dict(endpoint='unrelated')]), patch.object(sup, 'call') as call:
    try:sup.ensure_runtime()
    except RetryLater:pass
    else:raise AssertionError('Unrelated runtime must not be replaced')
    assert not call.called
checks.append(dict(check='offline_uncertain_and_unrelated_allocations_never_create_duplicates', passed=True))

# Assert cross-allocation screen/confirmation pairs are rejected before reporting.
import arch_resilience_v001 as helpers
fake=out/'cross-allocation';fake.mkdir()
for phase in ('screen','confirm'):
    run=fake/phase;(run/'artifacts').mkdir(parents=True)
    write(run/'artifacts/environment.json',dict(identity=dict(allocation_id=phase)))
    write(fake/('unit-'+phase+'-finished.json'),dict(status='completed'))
with patch.object(helpers,'check_contract'), patch.object(helpers,'units',return_value=['unit']), \
     patch.object(helpers,'verify_phase',side_effect=lambda cohort,stage:(fake/stage.split('-')[-1],dict(completed=True))):
    try:helpers.collect(dict(cohorts=[dict(path=str(fake),hardware='v6e')]),'v6e')
    except ValueError as error:assert 'device identities' in str(error)
    else:raise AssertionError('Cross-allocation pairing accepted')
checks.append(dict(check='cross_allocation_pair_rejected',passed=True))
write(out/'summary.json',dict(completed=True,checks=checks,saved_v6e_shapes=60,remaining_v6e_shapes=108))
print(json.dumps(dict(completed=True,checks=len(checks),saved_v6e_shapes=60,remaining_v6e_shapes=108)))
