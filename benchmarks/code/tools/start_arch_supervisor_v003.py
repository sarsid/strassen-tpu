"""Freeze or restart the authorized architecture supervisor without duplicate jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
from arch_resilience_v001 import ROOT, BASE, check_contract
from run_large_real_v002 import atomic, commit, read, write, utc
from run_region_cohort_v001 import sha


def initialize(folder):
    check_contract(ROOT)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'tools', 'runtime', 'configs'], cwd=ROOT, check=True)
    folder.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with (folder / 'source.tar').open('xb') as stream:
        subprocess.run(['git', 'archive', revision, '--', 'src', 'tools', 'runtime', 'configs', 'status', 'docs', 'AGENTS.md'], cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(folder / 'source.tar') as archive:
        archive.extractall(folder / 'source', filter='data')
    write(folder / 'frozen.json', dict(source_commit=revision, archive_sha256=sha(folder / 'source.tar'),
        source_sha256={str(p.relative_to(folder / 'source')): sha(p) for p in (folder / 'source').rglob('*') if p.is_file()}))
    write(folder / 'supervisor-state.json', dict(hardware='v6e', runtime=None, pending_request=None,
        current_cohort=None, finished=False, action='initializing', created_utc=utc(), source_commit=revision))
    write(folder / 'history.json', dict(cohorts=[dict(path=str(BASE), hardware='v6e'),
        dict(path=str(ROOT / 'runs/20260927-arch-v6e-continuation-v001'), hardware='v6e')], recoveries={}))
    write(folder / 'progress.json', dict(schema_version=1, campaign_id=folder.name, state='preparing',
        title='Automatic v6e then v5e recovery', heartbeat_utc=utc(), updated_utc=utc(), stages=[], selected_results=[],
        detail='Frozen automatic recovery controller; validating checkpoints before allocation.'))
    commit([folder], 'Freeze persistent automatic architecture-study supervisor')


def alive(pid, expected):
    command = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,command='], capture_output=True, text=True).stdout.strip()
    return bool(command and not command.startswith('Z') and expected in command)


def start(folder, replace_dashboard_pid=None):
    startlock = (folder / 'start.lock').open('ab')
    fcntl.flock(startlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime_lock = (folder / 'supervisor.lock').open('ab')
    try:
        fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='already_running', folder=str(folder))))
        return
    fcntl.flock(runtime_lock, fcntl.LOCK_UN)
    if read(folder / 'supervisor-state.json').get('finished'):
        print(json.dumps(dict(status='completed', folder=str(folder))))
        return
    active = read(folder / 'active-source.json') if (folder / 'active-source.json').exists() else {}
    snapshot = (folder / active.get('snapshot', '.')).resolve()
    if not snapshot.is_relative_to(folder.resolve()):
        raise ValueError('Active supervisor snapshot escaped its folder')
    frozen = read(snapshot / 'frozen.json')
    source = snapshot / 'source'
    entrypoint = active.get('entrypoint', 'supervise_arch_studies_v003.py')
    if entrypoint != Path(entrypoint).name or not entrypoint.startswith('supervise_arch_studies_v'):
        raise ValueError('Invalid supervisor entrypoint')
    for name, expected in frozen['source_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Supervisor frozen source changed: ' + name)
    env = dict(os.environ, STRASSEN_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(source / 'src'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    python = str(ROOT.parent / '.venv-reconcile/bin/python')
    with (folder / 'supervisor.log').open('a') as log:
        child = subprocess.Popen([python, str(source / 'tools' / entrypoint), '--folder', str(folder)],
            cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    atomic(folder / 'supervisor-process.json', dict(pid=child.pid, launched_utc=utc(), source_commit=frozen['source_commit'], snapshot=str(snapshot), entrypoint=entrypoint))
    if replace_dashboard_pid:
        if not alive(replace_dashboard_pid, 'status/server_v011.py --port 8788'):
            raise ValueError('Requested old dashboard PID did not match')
        os.kill(replace_dashboard_pid, signal.SIGTERM)
        time.sleep(1)
    dashboard = read(folder / 'dashboard-process.json') if (folder / 'dashboard-process.json').exists() else {}
    if not dashboard or not alive(dashboard['pid'], '--campaign-state ' + str(folder / 'progress.json')):
        with (folder / 'dashboard.log').open('a') as log:
            server = subprocess.Popen([python, str(source / 'status/server_v011.py'), '--port', '8788',
                '--campaign-state', str(folder / 'progress.json')], cwd=source, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        atomic(folder / 'dashboard-process.json', dict(pid=server.pid, launched_utc=utc(), url='http://127.0.0.1:8788/'))
    print(json.dumps(dict(status='started', supervisor_pid=child.pid, dashboard='http://127.0.0.1:8788/')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['initialize', 'start'])
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--replace-dashboard-pid', type=int)
    a = p.parse_args()
    folder = a.folder.resolve()
    if not folder.is_relative_to(ROOT / 'runs'):
        raise ValueError('Supervisor directory must be inside project runs')
    if a.action == 'initialize':
        initialize(folder)
    else:
        start(folder, a.replace_dashboard_pid)
