#!/usr/bin/env python3
"""Launch one frozen MM phase asynchronously inside the existing Colab VM.

The launcher returns promptly. A detached, bounded worker owns the benchmark
process, preserves failure evidence, and writes an exclusive completion archive.
No allocation, installation, JAX initialization or runtime reset occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

BASE = Path('/content/Strassen_MM_Focus')
RUNS = BASE / 'runs'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            value.update(data)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def checked_local_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or not path.resolve().is_relative_to(BASE):
        raise argparse.ArgumentTypeError('Path must be beneath /content/Strassen_MM_Focus')
    return path


def checked_module(value):
    if not re.fullmatch(r'strassen_mm\.[a-z][a-z0-9_]*_v[0-9]{3}', value):
        raise argparse.ArgumentTypeError('Use a versioned strassen_mm module')
    return value


def checked_runner_args(values):
    reserved = {'--campaign', '--phase', '--output-dir', '--expected-identity',
                '--allocation-id', '--max-wall-seconds'}
    for value in values:
        if '\x00' in value or value.split('=', 1)[0] in reserved:
            raise ValueError('Runner arguments cannot override frozen execution fields')
    return list(values)


def checked_run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise argparse.ArgumentTypeError('Use a short alphanumeric run ID')
    return value


def extract_source(archive, destination):
    with tarfile.open(archive, 'r:*') as package:
        members = package.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Archive contains an unsafe member path')
            if not (member.isdir() or member.isfile()):
                raise ValueError('Source archive must contain only regular files and directories')
        package.extractall(destination, members=members, filter='data')
    # Git archives place src/ at the archive root. Prefer that exact location;
    # archived test fixtures and old run snapshots are not candidate roots.
    marker = Path('src/strassen_mm/benchmark_v001.py')
    if (destination / marker).is_file():
        return destination
    # Also accept a single enclosing directory, as produced by tar(source/).
    # Never recurse into runtime validation fixtures, old runs or results.
    excluded = {'runtime', 'runs', 'results', 'tests', '.git', '__pycache__'}
    matches = [child for child in destination.iterdir()
               if child.is_dir() and child.name not in excluded
               and not child.name.startswith('validation_')
               and (child / marker).is_file()]
    if len(matches) != 1:
        raise ValueError('Expected source at archive root or one unique enclosing directory')
    return matches[0]


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=30)


def worker(run):
    run = checked_local_path(run)
    plan = json.loads((run / 'launch.json').read_text())
    source = Path(plan['source_root'])
    if not source.resolve().is_relative_to(run / 'source'):
        raise ValueError('Frozen source root escaped the run directory')
    environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(source / 'src'))
    status = {'run_id': run.name, 'phase': plan['phase'], 'started_utc': utc_now(),
              'allocation_id': plan['allocation_id'], 'status': 'running'}
    write_json(run / 'worker-start.json', {**status, 'worker_pid': os.getpid()})
    child = None
    lock_stream = None
    started = time.monotonic()
    try:
        import fcntl
        lock_stream = (BASE / 'active.lock').open('ab')
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another campaign worker holds the TPU execution lock') from None
        # Detect accidental edits between upload, launch and worker execution.
        if digest(run / 'input-source.tar.gz') != plan['archive_sha256']:
            raise RuntimeError('Source archive hash changed')
        with (run / 'benchmark.log').open('x') as log:
            child = subprocess.Popen(plan['command'], cwd=source, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(run / 'benchmark-start.json', {'pid': child.pid, 'started_utc': utc_now(),
                                                    'command': plan['command']})
            try:
                code = child.wait(timeout=plan['timeout_seconds'])
            except subprocess.TimeoutExpired:
                stop_child(child)
                status.update(status='timed_out', process_exit_code=child.returncode,
                              error='Outer worker timeout; process group terminated')
            else:
                status.update(status='completed' if code == 0 else 'failed', process_exit_code=code)
                if code == 0 and not (run / 'artifacts' / 'summary.json').exists():
                    status.update(status='failed', error='Process exited zero without summary.json')
    except BaseException as error:
        if child is not None:
            stop_child(child)
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        status.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json(run / 'completion.json', status)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            manifest = {str(path.relative_to(run)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                        for path in sorted(run.rglob('*')) if path.is_file()
                        and '__pycache__' not in path.parts and path.suffix != '.pyc'}
            write_json(run / 'run-manifest.json', manifest)
            archive = RUNS / (run.name + '.tar.gz')
            partial = RUNS / (run.name + '.tar.gz.partial')
            if archive.exists():
                raise FileExistsError('Completion archive already exists')
            with partial.open('xb') as stream:
                with tarfile.open(fileobj=stream, mode='w:gz') as package:
                    package.add(run, arcname=run.name)
            # This worker alone owns its unique archive path. No existing file is replaced.
            os.link(partial, archive)
            partial.unlink()
            write_json(run / 'archive-ready.json', {'created_utc': utc_now(),
                       'archive': str(archive), 'bytes': archive.stat().st_size,
                       'sha256': digest(archive), 'status': status['status']})
        except BaseException as error:
            write_json(run / 'archive-error.json', {'error_type': type(error).__name__,
                                                  'error': str(error), 'created_utc': utc_now()})
        if lock_stream is not None:
            lock_stream.close()
    return 0 if status['status'] == 'completed' else 1


def launch(args):
    if not args.archive.is_file() or not args.expected_identity.is_file():
        raise FileNotFoundError('Uploaded source archive and expected identity must exist')
    actual = digest(args.archive)
    if actual != args.archive_sha256:
        raise ValueError('Uploaded archive SHA256 differs from frozen local source')
    identity = json.loads(args.expected_identity.read_text())
    identity = identity.get('identity', identity)
    if identity.get('colab_endpoint') != args.allocation_id:
        raise ValueError('Expected identity does not match allocation ID')
    relative = PurePosixPath(args.campaign_relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Campaign path must be relative to frozen source root')
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / args.run_id
    run.mkdir(exist_ok=False)
    (run / 'source').mkdir()
    # An exclusive run directory prevents a later rerun from replacing evidence.
    with (run / 'input-source.tar.gz').open('xb') as target, args.archive.open('rb') as source_stream:
        shutil.copyfileobj(source_stream, target)
    with (run / 'expected-identity.json').open('xb') as target:
        target.write(args.expected_identity.read_bytes())
    source = extract_source(run / 'input-source.tar.gz', run / 'source')
    campaign = source.joinpath(*relative.parts)
    if not campaign.is_file():
        raise FileNotFoundError('Campaign configuration not found in frozen source')
    if not (source / 'src' / Path(*args.benchmark_module.split('.'))).with_suffix('.py').is_file():
        raise FileNotFoundError('Benchmark module absent from frozen source')
    command = [sys.executable, '-u', '-m', args.benchmark_module,
               '--campaign', str(campaign), '--phase', args.phase,
               '--output-dir', str(run / 'artifacts'),
               '--expected-identity', str(run / 'expected-identity.json'),
               '--allocation-id', args.allocation_id,
               '--max-wall-seconds', str(max(1, args.timeout_seconds - 300))]
    command.extend(checked_runner_args(args.runner_arg))
    source_text = globals().get('__source_text__')
    if source_text is None:
        source_text = Path(__file__).read_text()
    launcher = run / 'launch_phase_v004.py'
    with launcher.open('x') as stream:
        stream.write(source_text)
    write_json(run / 'launch.json', {'run_id': args.run_id, 'phase': args.phase,
               'created_utc': utc_now(), 'source_root': str(source),
               'archive_sha256': actual, 'allocation_id': args.allocation_id,
               'command': command, 'timeout_seconds': args.timeout_seconds,
               'campaign_sha256': digest(campaign),
               'expected_identity_sha256': digest(run / 'expected-identity.json'),
               'launcher_sha256': digest(launcher)})
    with (run / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(launcher), '--worker-dir', str(run)],
                                 cwd=run, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(run / 'launcher-started.json', {'worker_pid': child.pid, 'launched_utc': utc_now()})
    print(json.dumps({'kind': 'phase_launched', 'run_id': args.run_id, 'phase': args.phase,
                      'worker_pid': child.pid, 'run_directory': str(run),
                      'expected_archive': str(RUNS / (run.name + '.tar.gz'))}), flush=True)
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker-dir':
        return worker(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=checked_local_path)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--run-id', required=True, type=checked_run_id)
    parser.add_argument('--phase', required=True, type=checked_run_id)
    parser.add_argument('--campaign-relative', required=True)
    parser.add_argument('--benchmark-module', type=checked_module, default='strassen_mm.benchmark_v001')
    parser.add_argument('--runner-arg', action='append', default=[])
    parser.add_argument('--expected-identity', required=True, type=checked_local_path)
    parser.add_argument('--allocation-id', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=28800)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}', args.archive_sha256):
        parser.error('--archive-sha256 must be a lowercase SHA256 hex digest')
    if args.timeout_seconds < 60 or args.timeout_seconds > 86400:
        parser.error('--timeout-seconds must be between 60 and 86400')
    return launch(args)


if __name__ == '__main__':
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
