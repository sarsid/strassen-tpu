"""Park local orchestration during the final CPU export; leave its child alone.

This is a reversible operator pause, not a completed campaign. Resume requires
explicit user instruction and checked SIGCONT of these exact two processes.
No experiment source, plan, remote worker or runtime allocation is changed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from run_large_real_v002 import ROOT, atomic, commit, read, utc, verify_phase


def process(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'state=,command='],
                          capture_output=True, text=True, check=False).stdout.strip()


def checked_signal(pid, expected, sig):
    value = process(pid)
    if not value or value.startswith('Z') or expected not in value:
        raise RuntimeError(f'Process {pid} does not match the recorded owner')
    os.kill(pid, sig)


def park(pid, expected):
    checked_signal(pid, expected, signal.SIGSTOP)
    for _ in range(100):
        value = process(pid)
        if value.startswith('T') and expected in value:
            return
        time.sleep(.01)
    raise RuntimeError(f'Process {pid} did not acknowledge SIGSTOP')


def later_started(cohort, plan, target):
    index = next(i for i, s in enumerate(plan['stages']) if s['id'] == target)
    return [s['id'] for s in plan['stages'][index+1:]
            if (cohort/'operations'/s['id']/'execution.json').exists()]


def main(folder, target, out):
    if not folder.is_relative_to(ROOT/'runs') or not out.is_relative_to(folder):
        raise ValueError('Pause evidence must remain inside the recorded supervisor folder')
    lock = (out/'guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(folder/'supervisor-state.json')
    active = read(folder/'active-source.json')
    if state['hardware'] != 'v6e' or active['entrypoint'] != 'supervise_arch_studies_v004.py':
        raise ValueError('Expected the authorized v6e-only supervisor')
    cohort = Path(state['current_cohort'])
    plan = read(cohort/'plan.json')
    export_stage = 'export-' + target + '-confirm'
    next(s for s in plan['stages'] if s['id'] == export_stage)
    controller = read(cohort/'controller-process.json')['pid']
    supervisor = read(folder/'supervisor-process.json')['pid']
    expected_controller = str(cohort/'source/tools/run_large_real_v003.py') + ' run --cohort ' + str(cohort)
    expected_supervisor = str(folder/active['snapshot']/'source/tools'/active['entrypoint']) + ' --folder ' + str(folder)
    for pid, expected in [(controller, expected_controller), (supervisor, expected_supervisor)]:
        if expected not in process(pid):
            raise RuntimeError('Expected live process is absent')
    initial = dict(utc=utc(), status='armed', target=target, export_stage=export_stage,
                   cohort=str(cohort), controller_pid=controller, supervisor_pid=supervisor,
                   expected_controller=expected_controller, expected_supervisor=expected_supervisor,
                   runtime=state['runtime'], heartbeat_status='PAUSED', allocation_policy='retain_for_short_pause')
    atomic(out/'armed.json', initial)
    atomic(folder/'operator-pause-request.json', dict(**initial, evidence=str(out.relative_to(ROOT))))
    guard = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        operation = cohort/'operations'/export_stage
        while not (operation/'process.json').exists():
            if later_started(cohort, plan, export_stage):
                raise RuntimeError('The requested pause boundary has already been crossed')
            if read(folder/'supervisor-state.json').get('current_cohort') != str(cohort):
                raise RuntimeError('The cohort changed before the requested pause boundary')
            if expected_controller not in process(controller):
                raise RuntimeError('Controller exited before the requested pause boundary')
            time.sleep(.2)
        # The export is CPU-only and runs in a separate process. Parking just
        # its parent prevents every subsequent stage while export/commit finish.
        park(controller, expected_controller)
        park(supervisor, expected_supervisor)
        if later_started(cohort, plan, export_stage):
            raise RuntimeError('A later stage started before parking; preserve all evidence')
        child = read(operation/'process.json')['pid']
        expected_child = str(cohort/'source/tools/export_arch_study_v001.py')
        deadline = time.monotonic() + 1800
        while True:
            value = process(child)
            if not value or value.startswith('Z'):
                break
            if expected_child not in value:
                raise RuntimeError('Export child identity changed')
            if time.monotonic() > deadline:
                raise TimeoutError('Export did not finish within its normal 30-minute budget')
            time.sleep(1)
        evidence = []
        for part in ('screen', 'confirm'):
            phase = target + '-' + part
            run, _ = verify_phase(cohort, phase)
            base = ROOT/plan['results_relative']/'phases'/cohort.name
            meta = base/(phase+'.json')
            archive = base/(phase+'.tar.gz')
            saved = read(meta)
            if saved['sha256'] != hashlib.sha256(archive.read_bytes()).hexdigest():
                raise RuntimeError('Dedicated export checksum mismatch')
            for path in (meta, archive):
                rel = str(path.relative_to(ROOT))
                subprocess.run(['git', 'ls-files', '--error-unmatch', '--', rel], cwd=ROOT,
                               check=True, stdout=subprocess.DEVNULL)
                subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=ROOT, check=True)
            evidence.append(dict(phase=phase, run=str(run), exported=str(meta.relative_to(ROOT))))
        log = (operation/'execution.log').read_text()
        receipts = []
        for line in log.splitlines():
            try:
                receipts.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not any(r.get('exported') is True and r.get('phase') == target+'-confirm' for r in receipts):
            raise RuntimeError('Export process did not report successful committed export')
        detail = f'Paused after {target}: screen, confirmation and dedicated exports verified. TPU retained for the short pause.'
        for path in [cohort/'progress.json', folder/'progress.json']:
            value = read(path)
            value.update(state='paused', detail=detail, next_step='Wait for explicit user instruction before resuming.',
                         updated_utc=utc(), heartbeat_utc=utc(), operator_paused=True)
            if path.parent == folder:
                value['title'] = f'v6e: paused after {target}; results saved'
            atomic(path, value)
        done = dict(initial, completed_utc=utc(), pause_verified=True,
                    status='paused', measurement_code_changed=False, later_stages_started=[], evidence=evidence)
        atomic(out/'completion.json', done)
        atomic(folder/'operator-pause-request.json', dict(done, evidence=str(out.relative_to(ROOT))))
        subprocess.run([str(ROOT.parent/'.venv-reconcile/bin/python'), str(ROOT/'tools/record_progress_v001.py'),
                        '--kind', 'activity', '--id', 'current', '--state', 'idle', '--title', 'Paused safely after arch-130',
                        '--detail', detail, '--next-step', 'Safe to disconnect. Resume only when the user requests it.',
                        '--evidence', str((out/'completion.json').relative_to(ROOT))], check=True)
        commit([out, folder/'operator-pause-request.json', cohort/'progress.json', folder/'progress.json',
                ROOT/'status/work_progress_v001.jsonl'], 'Archive user-requested safe pause after arch-130')
        print(json.dumps(done), flush=True)
    except BaseException as error:
        for pid, expected in [(controller, expected_controller), (supervisor, expected_supervisor)]:
            if expected in process(pid):
                park(pid, expected)
        atomic(out/'attention.json', dict(utc=utc(), error=str(error), traceback=traceback.format_exc(),
                                        safe_to_disconnect=False, parents_parked=True))
        value = read(folder/'progress.json')
        value.update(state='attention', detail='Pause guard needs inspection: '+str(error), updated_utc=utc(),
                     next_step='Inspect pause evidence; orchestration is parked and active children are preserved.')
        atomic(folder/'progress.json', value)
        raise
    finally:
        guard.terminate()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.folder.resolve(), a.target, a.out.resolve())
