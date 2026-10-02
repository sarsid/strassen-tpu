"""Finish a prepared handover after the previous supervisor's lock closes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from arch_resilience_v001 import ROOT, check_contract
from run_large_real_v002 import atomic, commit, read, utc, write
from run_region_cohort_v001 import sha
from start_arch_supervisor_v003 import alive

p = argparse.ArgumentParser()
p.add_argument('--folder', type=Path, required=True)
p.add_argument('--tag', required=True)
a = p.parse_args()
folder = a.folder.resolve()
snapshot = folder / 'source-updates' / a.tag
if not folder.is_relative_to(ROOT / 'runs') or a.tag != Path(a.tag).name:
    raise ValueError('Invalid migration location')
started = read(snapshot / 'migration-started.json')
state = read(folder / 'supervisor-state.json')
if alive(started['previous_supervisor']['pid'], '--folder ' + str(folder)):
    raise ValueError('Previous supervisor is still running')
if state['current_cohort'] != started['cohort'] or state['runtime'] != started['runtime']:
    raise ValueError('Runtime or measurement cohort changed during handover')
expected_worker = str(Path(started['cohort']) / 'source/tools/run_large_real_v003.py')
if not alive(started['controller_pid'], expected_worker):
    raise ValueError('Measurement controller is not running')
frozen = read(snapshot / 'frozen.json')
for name, expected in frozen['source_sha256'].items():
    if sha(snapshot / 'source' / name) != expected:
        raise ValueError('Prepared snapshot changed: ' + name)
check_contract(snapshot / 'source')
startlock = (folder / 'start.lock').open('ab')
fcntl.flock(startlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
with (folder / 'supervisor.lock').open('ab') as lock:
    deadline = time.monotonic() + 20
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Supervisor lock remains held; do not start a duplicate')
            time.sleep(0.2)
active = dict(snapshot=str(snapshot.relative_to(folder)), entrypoint='supervise_arch_studies_v003.py',
    source_commit=frozen['source_commit'], activated_utc=utc())
atomic(folder / 'active-source.json', active)
fcntl.flock(startlock, fcntl.LOCK_UN)
subprocess.run([sys.executable, str(ROOT / 'tools/start_arch_supervisor_v003.py'), 'start', '--folder', str(folder)], check=True)
write(snapshot / 'migration-completion.json', dict(utc=utc(), active_source=active,
    new_supervisor=read(folder / 'supervisor-process.json'), controller_pid=started['controller_pid'],
    controller_still_running=alive(started['controller_pid'], expected_worker), allocation_unchanged=True,
    recovery='Initial handover stopped at the still-closing file lock; rechecked old-process absence and waited for exclusive ownership.'))
commit([folder / 'active-source.json', folder / 'supervisor-process.json', snapshot / 'migration-completion.json'],
    'Complete verified supervisor handover after lock release')
