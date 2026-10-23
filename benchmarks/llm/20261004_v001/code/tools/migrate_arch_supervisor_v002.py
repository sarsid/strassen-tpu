"""Replace only the local supervisor using a new immutable source snapshot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from arch_resilience_v001 import ROOT, check_contract
from run_large_real_v002 import atomic, commit, read, utc, write
from run_region_cohort_v001 import sha
from start_arch_supervisor_v003 import alive


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--entrypoint', required=True, choices=['supervise_arch_studies_v004.py'])
    a = p.parse_args()
    folder = a.folder.resolve()
    if not folder.is_relative_to(ROOT / 'runs') or a.tag != Path(a.tag).name:
        raise ValueError('Invalid migration location')
    startlock = (folder / 'start.lock').open('ab')
    fcntl.flock(startlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = read(folder / 'supervisor-process.json')
    state = read(folder / 'supervisor-state.json')
    if state['hardware'] != 'v6e':
        raise ValueError('This scope change requires a running v6e study')
    if not state.get('current_cohort'):
        raise ValueError('Migration requires an existing detached measurement controller')
    cohort = Path(state['current_cohort'])
    worker = read(cohort / 'controller-process.json')
    expected_worker = str(cohort / 'source/tools/run_large_real_v003.py')
    if not alive(worker['pid'], expected_worker):
        raise ValueError('Existing measurement controller is not running')
    if not alive(prior['pid'], '--folder ' + str(folder)):
        raise ValueError('Prior supervisor process does not match')
    snapshot = folder / 'source-updates' / a.tag
    snapshot.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with (snapshot / 'source.tar').open('xb') as stream:
        subprocess.run(['git', 'archive', revision, '--', 'src', 'tools', 'runtime', 'configs', 'status', 'docs', 'AGENTS.md'],
            cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(snapshot / 'source.tar') as archive:
        archive.extractall(snapshot / 'source', filter='data')
    check_contract(snapshot / 'source')
    if not (snapshot / 'source/tools' / a.entrypoint).exists():
        raise ValueError('Requested supervisor entrypoint is not committed')
    write(snapshot / 'frozen.json', dict(source_commit=revision, archive_sha256=sha(snapshot / 'source.tar'),
        source_sha256={str(p.relative_to(snapshot / 'source')): sha(p) for p in (snapshot / 'source').rglob('*') if p.is_file()}))
    write(snapshot / 'migration-started.json', dict(utc=utc(), previous_supervisor=prior,
        controller_pid=worker['pid'], cohort=str(cohort), runtime=state['runtime'],
        previous_active_source=read(folder / 'active-source.json') if (folder / 'active-source.json').exists() else None,
        intent='Replace local supervisor only; preserve the running measurement controller, allocation, and frozen kernels.'))
    commit([snapshot], 'Freeze v6e-only stop scope before supervisor handover')
    os.kill(prior['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while alive(prior['pid'], '--folder ' + str(folder)) and time.monotonic() < deadline:
        time.sleep(0.2)
    with (folder / 'supervisor.lock').open('ab') as lock:
        deadline = time.monotonic() + 20
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Prior supervisor lock still held; do not launch a duplicate')
                time.sleep(0.2)
    after = read(folder / 'supervisor-state.json')
    if after['current_cohort'] != str(cohort) or after['runtime'] != state['runtime']:
        raise RuntimeError('Measurement cohort or allocation changed; reconcile before handover')

    if not alive(worker['pid'], expected_worker):
        raise RuntimeError('Measurement controller exited during handover; retain state for reconciliation')
    active = dict(snapshot=str(snapshot.relative_to(folder)), entrypoint=a.entrypoint,
        source_commit=revision, activated_utc=utc())
    atomic(folder / 'active-source.json', active)
    fcntl.flock(startlock, fcntl.LOCK_UN)
    subprocess.run([sys.executable, str(ROOT / 'tools/start_arch_supervisor_v003.py'), 'start', '--folder', str(folder)], check=True)
    write(snapshot / 'migration-completion.json', dict(utc=utc(), active_source=active,
        new_supervisor=read(folder / 'supervisor-process.json'), controller_pid=worker['pid'],
        controller_still_running=alive(worker['pid'], expected_worker), allocation_unchanged=True))
    commit([folder / 'active-source.json', folder / 'supervisor-process.json', snapshot / 'migration-completion.json'],
        'Activate v6e-only supervisor without interrupting measurements')


if __name__ == '__main__':
    main()
