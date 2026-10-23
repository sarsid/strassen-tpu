"""Read worker completion and test the existing execution lock without launching work."""
import argparse
import fcntl
import json
from pathlib import Path

base = Path('/content/Strassen_MM_Focus')
p = argparse.ArgumentParser()
p.add_argument('--run-id', required=True)
a = p.parse_args()
if '/' in a.run_id or a.run_id in ('.', '..'):
    raise ValueError('Invalid run ID')
run = base / 'runs' / a.run_id
def read(name):
    path = run / name
    return json.loads(path.read_text()) if path.exists() else None
free = True
if (base / 'active.lock').exists():
    with (base / 'active.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            free = False
print(json.dumps(dict(kind='supervisor_probe', run_id=a.run_id, run_exists=run.exists(),
    execution_lock_free=free, completion=read('completion.json'), archive_ready=read('archive-ready.json'))), flush=True)
