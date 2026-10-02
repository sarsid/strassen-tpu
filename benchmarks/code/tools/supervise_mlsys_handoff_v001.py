"""Local-only wait for final shape completion, then one same-allocation continuation.

Never contacts the TPU itself. Pauses only the verified original controller parent
while its independent final phase finishes. After sealed, committed evidence, it
retires that parent before starting the frozen recovery controller. A watchdog
resumes the original on supervisor loss before takeover intent; after that fence
it never resumes a parent that could release a continuation's allocation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import run_mlsys_unattended_v001 as ctl

ROOT = ctl.ROOT
FINAL = 'MAIN-12-confirm'
PRE_TRANSFER = {'pause_pending', 'paused'}


def process(pid):
    result = subprocess.run(['/bin/ps', '-ww', '-p', str(int(pid)), '-o', 'lstart=', '-o', 'stat=', '-o', 'command='],
                            text=True, capture_output=True, env=dict(os.environ, LC_ALL='C'))
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode:
        raise RuntimeError('Cannot verify local process: ' + result.stderr.strip())
    parts = result.stdout.strip().split(None, 6)
    if len(parts) != 7:
        raise RuntimeError('Unexpected ps identity output')
    return dict(pid=int(pid), started=' '.join(parts[:5]), state=parts[5], command=parts[6])


def matches(actual, expected, zombie=False):
    return bool(actual and actual['pid'] == expected['pid'] and actual['started'] == expected['started']
                and (actual['command'] == expected['command'] or (zombie and 'Z' in actual['state'])))


def checked_signal(expected, sig):
    if not matches(process(expected['pid']), expected):
        raise RuntimeError('Refusing signal: process identity changed')
    os.kill(expected['pid'], sig)


def inactive(expected):
    actual = process(expected['pid'])
    if actual is None:
        return True
    if not matches(actual, expected, zombie=True):
        # Original PID has been reused, so the recorded process is no longer active.
        return actual['started'] != expected['started']
    return 'Z' in actual['state']


def forbidden(original):
    return any((original / p).exists() for p in ('operations/release', 'failure-cleanup', 'controller-completion.json'))


def require_clean(paths):
    names = [str(Path(p).relative_to(ROOT)) for p in paths]
    out = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all', '--', *names], cwd=ROOT, text=True)
    if out.strip():
        raise RuntimeError('Required handoff evidence is not yet committed: ' + out[:800])


def base_env(recovery):
    return dict(os.environ, STRASSEN_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(recovery / 'source/src'),
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def combined(original, recovery, control):
    state = deepcopy(ctl.read(original / 'progress.json'))
    state.update(title='MLSys shapes and automatic actual-model continuation', campaign_id=recovery.name,
                 cohort_path=str(recovery.relative_to(ROOT)))
    # The superseded release/report belong to the recovery, not to both queues.
    state['stages'] = [s for s in state['stages'] if s['id'] not in ('release', 'report')]
    pending = [dict(id=s['id'], label=s.get('label', s['id']), kind=s.get('kind', 'measurement'),
                    state='not_started', expected=s.get('expected'), completed=0, succeeded=0, failed=0, measured=0,
                    detail='Automatically queued after final shape verification.', evidence=[])
               for s in ctl.read(recovery / 'plan.json')['stages']]
    phase = control['state']
    if phase in ('pause_pending', 'paused', 'takeover_intent', 'retired', 'recovery_started', 'completed'):
        receipt = original / (FINAL + '-started.json')
        if receipt.exists():
            run = Path(ctl.read(receipt)['run'])
            rows, _ = ctl.read_new_rows(run / 'results.jsonl', 0)
            if (run / 'artifacts/results.jsonl').exists():
                rows, _ = ctl.read_new_rows(run / 'artifacts/results.jsonl', 0)
            item = next(s for s in state['stages'] if s['id'] == FINAL)
            item.update(ctl.counts(rows))
            if control.get('final_verified'):
                item.update(state='succeeded', detail='Sealed, committed final phase verified by local handoff.')
                state['selected_results'] = [r for r in state.get('selected_results', []) if r.get('stage') != FINAL]
                state['selected_results'].extend(ctl.headline_results(rows, FINAL, run))
            last = next((r.get('utc') for r in reversed(rows) if r.get('utc')), None)
            if last:
                item['worker_last_data_utc'] = last
                state['worker_last_data_utc'] = last
    current = ctl.read(recovery / 'progress.json') if (recovery / 'progress.json').exists() else {}
    if (phase in ('recovery_started', 'completed') or (recovery/'controller-started.json').exists()) and current.get('stages'):
        for key in ('state', 'stage', 'detail', 'next_step', 'worker_last_data_utc',
                    'allocation_released', 'remote_execution_uncertain'):
            if key in current:
                state[key] = current[key]
        pending = current['stages']
        state['selected_results'].extend(current.get('selected_results', []))
        old_audit = next(s for s in state['stages'] if s['id'] == FINAL + '-audit')
        audit = next((s for s in pending if s['id'] == 'original-final-audit'), {})
        old_audit.update(state=audit.get('state', 'not_started'), detail='Executed by continuation: original-final-audit.')
    else:
        state['next_step'] = 'Automatic handoff on this allocation: repaired model tools → Qwen → Mistral → Gemma → release.'
    for stage in pending:
        stage = deepcopy(stage)
        stage['id'] = 'recovery/' + stage['id']
        state['stages'].append(stage)
    if phase == 'failed':
        state.update(state='attention', detail=control.get('error', 'Handoff failed; inspect local evidence.'))
    state.update(updated_utc=ctl.utc(), heartbeat_utc=ctl.utc(), handoff_state=phase)
    return state


def start_recovery_once(recovery):
    """No retry after an ambiguous Popen; the controller startup receipt is durable."""
    plan = ctl.read(recovery/'plan.json')
    receipt = recovery/'controller-process.json'
    startup = recovery/'controller-started.json'
    if receipt.exists() or startup.exists():
        pid = ctl.read(receipt if receipt.exists() else startup)['pid']
        candidate = process(pid)
        expected = str(recovery/'source/tools/run_mlsys_unattended_v001.py')+' run --cohort '+str(recovery)
        if candidate and expected in candidate['command']:
            return candidate
        if (recovery/'controller-completion.json').exists():
            return None
        raise RuntimeError('Recovery startup is recorded but process is absent; no automatic duplicate')
    intent = recovery/'recovery-launch-intent.json'
    if intent.exists():
        # A dead supervisor could have spawned a child just before recording PID.
        # Give that child time to write its own startup receipt, then fail closed.
        for _ in range(20):
            time.sleep(.5)
            if startup.exists() or receipt.exists():
                return start_recovery_once(recovery)
        raise RuntimeError('Recovery launch intent has no receipt; inspect before retry')
    ctl.write(intent, dict(utc=ctl.utc(), allocation_id=plan['allocation_id']))
    with (recovery/'controller.log').open('x') as log:
        actual = subprocess.Popen([plan['controller_python'], str(recovery/'source/tools/run_mlsys_unattended_v001.py'),
            'run', '--cohort', str(recovery)], cwd=recovery/'source', env=base_env(recovery),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    ctl.write(receipt, dict(pid=actual.pid, launched_utc=ctl.utc()))
    identity = process(actual.pid)
    if identity is None:
        raise RuntimeError('Recovery process exited during startup')
    return identity


def watchdog(recovery):
    config = ctl.read(recovery/'handoff-plan.json')
    supervisor = ctl.read(recovery/'handoff-supervisor.json')
    while not inactive(supervisor):
        if ctl.read(recovery/'handoff-state.json')['state'] == 'completed':
            return 0
        time.sleep(2)
    state = ctl.read(recovery/'handoff-state.json')
    result = dict(utc=ctl.utc(), state_seen=state['state'], resumed=False)
    if state['state'] in PRE_TRANSFER and matches(process(config['parent']['pid']), config['parent']):
        checked_signal(config['parent'], signal.SIGCONT)
        result['resumed'] = True
    elif state['state'] in ('takeover_intent', 'retired', 'recovery_started') or state.get('takeover_fenced'):
        lock = (recovery/'handoff.lock').open('ab')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            original=Path(config['original']); parent=config['parent']
            verified=ctl.read(recovery/'handoff-verification.json')
            if not state.get('final_verified') or verified.get('remote_may_still_be_running') is not False:
                raise RuntimeError('Missing verified idle evidence for watchdog takeover')
            run_path,_=ctl.verify_phase(original, FINAL)
            require_clean([run_path, original/(FINAL+'-finished.json')])
            if forbidden(original):
                raise RuntimeError('Original release/cleanup began; watchdog refuses takeover')
            if not inactive(parent):
                observed=process(parent['pid'])
                if not matches(observed,parent) or 'T' not in observed['state']:
                    raise RuntimeError('Original parent is not the verified stopped process')
                checked_signal(parent,signal.SIGKILL)
                for _ in range(100):
                    if inactive(parent): break
                    time.sleep(.1)
                else: raise RuntimeError('Watchdog could not retire parent')
            actual=start_recovery_once(recovery)
            state.update(state='recovery_started', watchdog_adopted=True)
            ctl.atomic(recovery/'handoff-state.json',state)
            while actual and not inactive(actual):
                ctl.atomic(recovery/'combined-progress.json',combined(original,recovery,state))
                time.sleep(5)
            if not (recovery/'controller-completion.json').exists():
                raise RuntimeError('Adopted recovery ended without completion')
            state.update(state='completed',updated_utc=ctl.utc())
            ctl.atomic(recovery/'handoff-state.json',state)
            ctl.atomic(recovery/'combined-progress.json',combined(original,recovery,state))
            result['adopted_recovery']=True
        except Exception as exc:
            result['error']=str(exc)
            state.update(state='failed',error='Watchdog: '+str(exc))
            ctl.atomic(recovery/'handoff-state.json',state)
            ctl.atomic(recovery/'combined-progress.json',combined(Path(config['original']),recovery,state))
    ctl.write(recovery/'handoff-watchdog-result.json',result)
    ctl.commit([recovery/'handoff-watchdog-result.json'],'Archive local handoff watchdog outcome')
    return 0


def run(recovery):
    config = ctl.read(recovery / 'handoff-plan.json'); original = Path(config['original'])
    parent = config['parent']; plan = ctl.read(recovery / 'plan.json')
    lock = (recovery / 'handoff.lock').open('ab'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ctl.archive.verify_frozen(recovery)
    ctl.write(recovery / 'handoff-supervisor.json', process(os.getpid()))
    control = dict(state='observing', started_utc=ctl.utc(), final_verified=False)
    journal = (recovery / 'handoff-events.jsonl').open('a', buffering=1)
    def publish():
        control['updated_utc'] = ctl.utc(); ctl.atomic(recovery / 'handoff-state.json', control)
        ctl.atomic(recovery / 'combined-progress.json', combined(original, recovery, control))
    def transition(state, **fields):
        control.update(state=state, **fields); publish()
        journal.write(json.dumps(dict(event=state, utc=ctl.utc(), **fields)) + '\n')
    publish()
    with (recovery / 'handoff-watchdog.log').open('x') as log:
        guard = subprocess.Popen([sys.executable, __file__, 'watchdog', '--recovery', str(recovery)],
                                 env=base_env(recovery), stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    ctl.write(recovery / 'handoff-watchdog.json', dict(pid=guard.pid))
    sleep_guard = None
    if Path('/usr/bin/caffeinate').exists():
        sleep_guard = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        started = time.monotonic()
        while not (original / (FINAL + '-started.json')).exists():
            if forbidden(original) or not matches(process(parent['pid']), parent):
                raise RuntimeError('Original campaign ended or changed before final-phase handoff; no remote action taken')
            if time.monotonic() - started > 48 * 3600:
                raise TimeoutError('Local handoff wait exceeded 48 hours')
            publish(); time.sleep(5)
        if forbidden(original):
            raise RuntimeError('Original release or cleanup already began')
        child = process(ctl.read(original / 'operations' / FINAL / 'process.json')['pid'])
        if not child or str(original) not in child['command'] or '--stage ' + FINAL not in child['command']:
            raise RuntimeError('Final phase process identity could not be established')
        transition('pause_pending', final_child=child)
        checked_signal(parent, signal.SIGSTOP)
        for _ in range(50):
            observed = process(parent['pid'])
            if matches(observed, parent) and 'T' in observed['state']:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original parent did not enter stopped state')
        if forbidden(original):
            raise RuntimeError('Release race detected after parent stop')
        transition('paused')
        wait_start = time.monotonic()
        while not inactive(child):
            if guard.poll() is not None:
                raise RuntimeError('Local fail-safe watchdog stopped')
            if time.monotonic() - wait_start > 10000:
                raise TimeoutError('Final phase transport exceeded handoff wait budget')
            publish(); time.sleep(5)
        run_path, summary = ctl.verify_phase(original, FINAL)
        original_progress = ctl.read(original / 'progress.json')
        required = [s for s in original_progress['stages'] if s['id'].startswith('MAIN-') and s['id'] not in (FINAL, FINAL+'-audit')]
        if len(required) != 48 or any(s['state'] != 'succeeded' for s in required):
            raise RuntimeError('Earlier main-study stages are incomplete or unsuccessful')
        rows, _ = ctl.read_new_rows(run_path / 'artifacts/results.jsonl', 0)
        groups = ctl.read(run_path / 'artifacts/planned_cases.json')
        if ctl.counts(rows)['completed'] != sum(len(g['inputs']) * len(g['arms']) for g in groups):
            raise RuntimeError('Final phase count differs from frozen plan')
        require_clean([run_path, original / (FINAL+'-started.json'), original / (FINAL+'-finished.json')])
        transition('paused', final_verified=True, final_run=str(run_path))
        ctl.write(recovery / 'handoff-verification.json', dict(verified_utc=ctl.utc(), original=str(original),
                  parent=parent, child=child, final_run=str(run_path), remote_may_still_be_running=False,
                  count=ctl.counts(rows), child_exit_status='inactive or zombie; exact exit code not available',
                  original_controller_status='Will be superseded, not marked normally completed'))
        ctl.commit([recovery/'handoff-verification.json', recovery/'handoff-events.jsonl'], 'Verify completed shape evidence before same-allocation model handoff')
        observed = process(parent['pid'])
        if forbidden(original) or not matches(observed, parent) or 'T' not in observed['state']:
            raise RuntimeError('Original controller ownership or stopped state changed before retirement')
        transition('takeover_intent')
        # SIGKILL is limited to the stopped orchestration parent. Its phase child
        # is already inactive, archived and committed. It prevents final cleanup
        # from releasing the allocation during the replacement controller run.
        checked_signal(parent, signal.SIGKILL)
        for _ in range(100):
            if inactive(parent):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original controller did not retire; recovery not launched')
        transition('retired')
        ctl.write(recovery / 'handoff-retired.json', dict(utc=ctl.utc(), parent=parent, original_controller='superseded',
                  allocation_id=plan['allocation_id'], allocation_released=False))
        ctl.commit([recovery/'handoff-retired.json', recovery/'handoff-events.jsonl'], 'Record retired shape controller and retained allocation')
        actual = start_recovery_once(recovery)
        transition('recovery_started', recovery_pid=actual['pid'] if actual else None)
        ctl.commit([recovery/'controller-process.json', recovery/'handoff-events.jsonl', recovery/'recovery-launch-intent.json'], 'Launch actual-model continuation after shape completion')
        while actual and not inactive(actual):
            publish(); time.sleep(5)
        if not (recovery/'controller-completion.json').exists():
            raise RuntimeError('Recovery controller ended without a completion receipt; no blind retry')
        transition('completed', recovery_completion=ctl.read(recovery/'controller-completion.json')['status'])
    except BaseException as exc:
        fenced = control['state'] in ('takeover_intent', 'retired', 'recovery_started', 'completed')
        if not fenced and control['state'] in PRE_TRANSFER and matches(process(parent['pid']), parent):
            checked_signal(parent, signal.SIGCONT)
        transition('failed', error=str(exc), takeover_fenced=fenced, traceback=traceback.format_exc())
        return 1
    finally:
        journal.close()
        if sleep_guard is not None:
            sleep_guard.terminate()
        ctl.write(recovery/'handoff-completion.json', control)
        ctl.commit([recovery/'handoff-completion.json', recovery/'handoff-events.jsonl', recovery/'combined-progress.json'],
                   'Archive local campaign handoff '+control['state'])
    return 0


def launch(original, recovery):
    ctl.archive.verify_frozen(recovery); plan = ctl.read(recovery/'plan.json'); old = ctl.read(original/'plan.json')
    if any(plan[k] != old[k] for k in ('session', 'allocation_id', 'remote_identity', 'private_root')):
        raise ValueError('Continuation must retain original allocation and private root')
    if forbidden(original):
        raise RuntimeError('Original campaign already entered terminal cleanup')
    parent = process(ctl.read(original/'controller-process.json')['pid'])
    expected = str(original/'source/tools/run_mlsys_unattended_v001.py')+' run --cohort '+str(original)
    if not parent or expected not in parent['command']:
        raise RuntimeError('Unexpected original controller identity')
    ctl.write(recovery/'handoff-plan.json', dict(original=str(original), recovery=str(recovery), parent=parent,
        armed_utc=ctl.utc(), gate=FINAL, allocation_id=plan['allocation_id'], no_remote_access_before_gate=True))
    ctl.commit([recovery/'handoff-plan.json'], 'Arm automatic same-allocation model continuation after shape study')
    with (recovery/'handoff.log').open('x') as log:
        child = subprocess.Popen([plan['controller_python'], str(recovery/'source/tools/supervise_mlsys_handoff_v001.py'),
            'run', '--recovery', str(recovery)], cwd=recovery/'source', env=base_env(recovery),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    ctl.write(recovery/'handoff-process.json', dict(pid=child.pid, launched_utc=ctl.utc()))
    ctl.commit([recovery/'handoff-process.json'], 'Record detached local handoff supervisor')
    print(json.dumps(dict(status='launched', pid=child.pid, recovery=str(recovery))))


def dashboard(original,recovery):
    import urllib.request
    feed=recovery/'combined-progress.json'
    if not feed.exists():
        raise RuntimeError('Combined feed is not ready')
    plan=ctl.read(recovery/'plan.json')
    previous=process(ctl.read(original/'controller-process.json')['dashboard_pid'])
    if not previous or str(original/'source/status/server_v011.py') not in previous['command'] or '--port 8767' not in previous['command']:
        raise RuntimeError('Unexpected original dashboard process')
    command=[plan['controller_python'],str(recovery/'source/status/server_v011.py'),'--port','8767','--campaign-state',str(feed)]
    ctl.write(recovery/'dashboard-switch.json',dict(utc=ctl.utc(),previous=previous,command=command))
    checked_signal(previous,signal.SIGTERM)
    for _ in range(100):
        if inactive(previous): break
        time.sleep(.1)
    else: raise RuntimeError('Old dashboard did not stop')
    with (recovery/'dashboard.log').open('x') as log:
        child=subprocess.Popen(command,cwd=recovery/'source',env=base_env(recovery),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=log,start_new_session=True)
    ctl.write(recovery/'dashboard-process.json',dict(pid=child.pid,utc=ctl.utc(),url='http://127.0.0.1:8767/'))
    error=None
    for _ in range(40):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8767/api/progress',timeout=2) as response:
                payload=json.load(response)
            if recovery.name not in json.dumps(payload):
                raise RuntimeError('New dashboard does not expose recovery feed')
            error=None;break
        except Exception as exc:
            error=exc;time.sleep(.25)
    if error: raise error
    ctl.write(recovery/'dashboard-verified.json',dict(utc=ctl.utc(),campaign_id=recovery.name,queued_models=['qwen','mistral','gemma']))
    ctl.commit([recovery/'dashboard-switch.json',recovery/'dashboard-process.json',recovery/'dashboard-verified.json'],
               'Show queued actual-model continuation on existing live dashboard')
    print(json.dumps(dict(status='verified',url='http://127.0.0.1:8767/')))


def main():
    p=argparse.ArgumentParser(); p.add_argument('action', choices=('launch','run','watchdog','dashboard'))
    p.add_argument('--recovery', type=Path, required=True); p.add_argument('--original', type=Path)
    args=p.parse_args(); recovery=args.recovery.resolve()
    if not recovery.is_relative_to(ROOT/'runs'):
        raise ValueError('Recovery must be a project run')
    if args.action in ('launch','dashboard'):
        if args.original is None:
            p.error('--original required for launch/dashboard')
        return (launch if args.action=='launch' else dashboard)(args.original.resolve(), recovery)
    return run(recovery) if args.action=='run' else watchdog(recovery)

if __name__=='__main__':
    raise SystemExit(main())
