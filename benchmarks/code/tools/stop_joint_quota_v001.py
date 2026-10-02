"""Bounded user-requested shutdown, TPU release, and partial-results report."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from run_large_real_v002 import verify_phase

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'runs/20260924-joint-v001'
OUT = COHORT / 'operations/quota-closeout-v001'


def now(): return datetime.now(timezone.utc).isoformat()
def save(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def checked_signal(pid, expected, sig):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    if result.returncode:
        return False
    if expected not in result.stdout:
        raise RuntimeError('Refusing to signal a different process: ' + str(pid))
    os.kill(pid, sig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    plan = json.loads((COHORT / 'plan.json').read_text())
    parent = json.loads((COHORT / 'controller-process.json').read_text())['pid']
    expected = str(COHORT / 'source/tools/run_large_real_v002.py') + ' run --cohort ' + str(COHORT)
    checked_signal(parent, expected, signal.SIGSTOP)
    save(OUT / 'started.json', dict(utc=now(), reason='User requested shutdown before token quota exhaustion',
         controller_pid=parent, archive_grace_seconds=180, source_file=str(Path(__file__).resolve())))
    deadline = time.monotonic() + 180
    while not (COHORT / 'joint-03-confirm-finished.json').exists() and time.monotonic() < deadline:
        time.sleep(5)
    verified = []
    errors = []
    for i in (1, 2, 3):
        phase = f'joint-{i:02d}-confirm'
        if (COHORT / (phase + '-finished.json')).exists():
            try:
                verify_phase(COHORT, phase)
                verified.append(i)
            except Exception as error:
                errors.append(dict(phase=phase, error=str(error)))
    checked_signal(parent, expected, signal.SIGKILL)
    child_record = COHORT / 'operations/joint-03-confirm/process.json'
    if child_record.exists():
        child = json.loads(child_record.read_text())['pid']
        checked_signal(child, 'run_large_real_v002.py phase --cohort ' + str(COHORT), signal.SIGTERM)
    command = [plan['controller_python'], str(COHORT / 'source/runtime/release_allocation_v002.py'),
               '--session', plan['session'], '--expect-endpoint', plan['allocation_id']]
    save(OUT / 'release-command.json', dict(command=command, utc=now()))
    with (OUT / 'release.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=150)
    released = result.returncode == 0 and '"verified_absent": true' in (OUT / 'release.log').read_text()
    save(OUT / 'release.json', dict(released=released, endpoint=plan['allocation_id'], utc=now()))
    report = OUT / 'artifacts'
    with (OUT / 'report.log').open('w') as log:
        report_result = subprocess.run([plan['analysis_python'], str(COHORT / 'source/tools/report_joint_v001.py'),
            '--cohort', str(COHORT), '--output-dir', str(report)], stdout=log, stderr=subprocess.STDOUT, timeout=300)
    report_data = json.loads((report / 'results.json').read_text()) if (report / 'results.json').exists() else None
    banner = ('# Partial study — stopped at the user’s token-budget request\n\n'
              f'{len(verified)} of 6 planned shapes completed confirmation. '
              f'TPU release verified: {released}. '
              'Unrun shapes are not failures or negative results. The original protocol and raw archives are preserved.\n\n')
    for name in ('RESULTS.md', 'TUNER_DESIGN.md'):
        path = report / name
        if path.exists(): path.write_text(banner + path.read_text())
    state = json.loads((COHORT / 'progress.json').read_text())
    for stage in state['stages']:
        if stage['id'] == 'joint-03-confirm' and 3 in verified:
            receipt = json.loads((COHORT / 'joint-03-confirm-finished.json').read_text())
            records = [json.loads(line) for line in (Path(receipt['run']) / 'artifacts/results.jsonl').read_text().splitlines()]
            cases = [r for r in records if r['event'] == 'case_result']
            stage.update(state='succeeded', completed=len(cases), expected=len(cases),
                         succeeded=sum(r['status'] == 'ok' for r in cases), failed=sum(r['status'] != 'ok' for r in cases))
        elif stage['state'] in ('running', 'not_started'):
            stage.update(state='cancelled', detail='Stopped at user token-budget request')
    state.update(state='cancelled', stage='quota-closeout', updated_utc=now(), heartbeat_utc=now(),
                 allocation_released=released, remote_execution_uncertain=not released,
                 detail=f'User-requested stop after {len(verified)}/6 shapes; TPU released={released}.',
                 next_step='Review saved partial results and tuner design; remaining shapes require a future authorized run.')
    save(COHORT / 'progress.json', state)
    completion = dict(status='closed_partial' if released and report_data else 'attention', utc=now(),
                      allocation_released=released, verified_confirmed_shapes=verified, verification_errors=errors,
                      report_exit_code=report_result.returncode, full_experiment_completed=False,
                      missing_shapes=report_data['missing'] if report_data else None)
    save(OUT / 'completion.json', completion)
    paths = [str(OUT.relative_to(ROOT)), 'runs/20260924-joint-v001/progress.json',
             'runs/20260924-joint-v001/user-stop-request-v001.json']
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Preserve user-requested quota shutdown and partial joint results', '--', *paths], cwd=ROOT, check=True)
    print(json.dumps(completion), flush=True)
    return 0 if released and report_data else 1


if __name__ == '__main__':
    raise SystemExit(main())
