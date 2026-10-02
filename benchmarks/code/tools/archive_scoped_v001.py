"""Freeze committed inputs and archive one command without staging other work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout_seconds <= 0:
        parser.error('a command and positive timeout are required')
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        parser.error('label must contain only letters, numbers, hyphens or underscores')
    for name in args.source:
        candidate = (ROOT / name).resolve()
        if not candidate.is_relative_to(ROOT) or candidate == ROOT:
            parser.error('source paths must be inside the project')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *args.source], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + args.label + '-' + uuid.uuid4().hex[:6]
    run = ROOT / 'runs' / run_id
    run.mkdir()
    source = run / 'source'
    source.mkdir()
    archive = run / 'source.tar'
    with archive.open('xb') as handle:
        subprocess.run(['git', 'archive', revision, '--', *args.source], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter='data')
    save(run / 'source-manifest.json', {'source_commit': revision, 'archive_sha256': digest(archive),
         'files': {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}})
    save(run / 'execution.json', {'run_id': run_id, 'label': args.label, 'source_commit': revision,
         'command': command, 'scope': 'local preparation or correctness; never TPU performance',
         'started_utc': datetime.now(timezone.utc).isoformat(), 'timeout_seconds': args.timeout_seconds})
    print('EXECUTION_DIR=' + str(run), flush=True)
    env = dict(os.environ, PYTHONPATH=str(source / 'src'), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               STRASSEN_EXECUTION_DIR=str(run), STRASSEN_PROJECT_ROOT=str(ROOT), JAX_PLATFORMS='cpu')
    started = time.monotonic()
    code = -1
    problem = None
    try:
        with (run / 'execution.log').open('x') as handle:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            save(run / 'process.json', {'pid': child.pid})
            try:
                code = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                problem = 'TimeoutExpired'
    except Exception as error:
        problem = type(error).__name__
    save(run / 'completion.json', {'status': 'completed' if code == 0 else 'failed', 'exit_code': code,
         'error_type': problem, 'elapsed_seconds': time.monotonic() - started,
         'finished_utc': datetime.now(timezone.utc).isoformat()})
    save(run / 'artifact-manifest.json', {'files': {str(p.relative_to(run)): {'sha256': digest(p), 'bytes': p.stat().st_size}
         for p in sorted(run.rglob('*')) if p.is_file()}})
    relative = str(run.relative_to(ROOT))
    subprocess.run(['git', 'add', '--', relative], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Archive ' + run_id, '--', relative], cwd=ROOT, check=True)
    print(json.dumps({'run_id': run_id, 'exit_code': code, 'error_type': problem}), flush=True)
    return 0 if code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
