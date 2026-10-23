"""Exercise launch-loss recovery, refusal paths, and exact continuation coverage."""
import contextlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import patch
import run_phase_v006 as transport
from run_region_cohort_v001 import write, sha
import prepare_arch_continuation_v001 as prepare
from report_arch_continuation_v001 import make_view

out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
checks = []


def exercise(name, lost=True, absent=False, wrong_source=False):
    root = out / name
    root.mkdir()
    run = root / 'test-run'
    run.mkdir()
    (run / 'source').mkdir()
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
    if absent or wrong_source:
        assert code == 1 and terminal['remote_may_still_be_running'] is True
    else:
        assert code == 0 and terminal['remote_may_still_be_running'] is False
        assert terminal['launch_acknowledged'] == (not lost)
    checks.append(dict(check=name, passed=True, launch_count=len(launches)))


exercise('lost_ack_completed_worker')
exercise('acknowledged_launch', lost=False)
exercise('lost_ack_missing_worker_stops', absent=True)
exercise('wrong_source_archive_rejected', wrong_source=True)
prior = json.loads((prepare.PRIOR / 'plan.json').read_text())
args = SimpleNamespace(hardware='v6e', cohort=out / 'future-cohort', session=prior['session'],
    endpoint=prior['allocation_id'], identity=prior['remote_identity'],
    controller_python=prior['controller_python'], analysis_python=prior['analysis_python'])
plan = prepare.build(args)
measurement = [s for s in plan['stages'] if s['operation'] == 'phase']
indices = {int(s['runner_args'][1]) for s in measurement}
assert len(measurement) == 280 and len(indices) == 140
assert indices.isdisjoint(plan['completed_shape_indices'])
assert indices | set(plan['completed_shape_indices']) == set(range(168))
assert measurement[0]['id'] == 'arch-028-screen'
assert plan['transport'] == 'tools/run_phase_v006.py'
checks.append(dict(check='unchanged_measurement_sources_and_exact_140_shape_continuation', passed=True))
view = out / 'partial-input-view'
sources = make_view(prepare.PRIOR, view, partial=True)
assert len(sources) == 56
recovered = next(s for s in sources if s['phase'] == 'arch-027-confirm')
assert recovered['evidence'].endswith('reconciliation.json')
checks.append(dict(check='28_completed_shapes_verified_including_recovered_confirmation', passed=True))
write(out / 'summary.json', dict(completed=True, checks=checks, remaining_shapes=140,
    measurement_phases=280, completed_shapes=28))
print(json.dumps(dict(completed=True, checks=len(checks), completed_shapes=28, remaining_shapes=140)))
