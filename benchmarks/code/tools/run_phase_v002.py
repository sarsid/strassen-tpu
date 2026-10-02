"""Execute a frozen phase with byte-exact streaming and bounded recovery.

Version 002 preserves timed-out controller calls, conservatively records launch
uncertainty, and bounds both overall monitoring and completion-archive waits.
Source and final evidence are committed through the existing archive manager.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import archive_v002 as archive
from events_v001 import emit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_object(stdout, kind):
    streams = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get('kind') == 'remote_stream':
            streams.append(value['text'])
    for line in ''.join(streams).splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get('kind') == kind:
            return value
    raise RuntimeError('Controller returned no ' + kind + ' object; inspect retained log')


def captured_output(value):
    """TimeoutExpired can contain bytes even when subprocess text=True."""
    if value is None:
        return {'text': '', 'base64': ''}
    raw = value if isinstance(value, bytes) else value.encode('utf-8')
    return {'text': raw.decode('utf-8', errors='replace'),
            'base64': base64.b64encode(raw).decode('ascii')}


def valid_record(value):
    return isinstance(value, dict) and bool(value) and not value.get('unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'N1', 'N2', 'N3', 'N4'])
    parser.add_argument('--session', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--expected-identity', required=True)
    parser.add_argument('--controller-python', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    parser.add_argument('--archive-grace-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 60 <= args.timeout_seconds <= 86400:
        parser.error('--timeout-seconds must be between 60 and 86400')
    if not 30 <= args.archive_grace_seconds <= 3600:
        parser.error('--archive-grace-seconds must be between 30 and 3600')
    run = archive.begin(args.phase + '-v5e-v002', scope='tpu_benchmark')
    print('EXECUTION_DIR=' + str(run), flush=True)
    source = run / 'source'
    controller = [args.controller_python, str(source / 'runtime/colab_control_v001.py')]
    common = ['--session', args.session, '--expect-endpoint', args.endpoint]
    calls = run / 'controller'
    calls.mkdir()
    counter = 0
    orchestration_started = time.monotonic()
    grace_seconds = max(900, args.archive_grace_seconds + 300)
    deadline = orchestration_started + args.timeout_seconds + grace_seconds

    def remaining_time():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Overall phase monitoring budget exhausted; remote allocation retained for recovery')
        return remaining

    def bounded_sleep(seconds):
        time.sleep(min(seconds, remaining_time()))

    def call(command, **kwargs):
        nonlocal counter
        remaining = remaining_time()
        counter += 1
        path = calls / f'{counter:05d}.json'
        started = archive.utc()
        kwargs['timeout'] = min(kwargs.get('timeout', remaining), remaining)
        record = {'started_utc': started, 'command': command,
                  'argv': controller + command, 'timeout_seconds': kwargs['timeout']}
        try:
            child = subprocess.run(controller + command, capture_output=True, text=True, **kwargs)
        except subprocess.TimeoutExpired as error:
            stdout, stderr = captured_output(error.stdout), captured_output(error.stderr)
            record.update(finished_utc=archive.utc(), returncode=None,
                          error={'type': 'TimeoutExpired', 'timeout_seconds': error.timeout},
                          stdout=stdout['text'], stderr=stderr['text'],
                          stdout_base64=stdout['base64'], stderr_base64=stderr['base64'])
            archive.exclusive_json(path, record)
            raise
        except BaseException as error:
            record.update(finished_utc=archive.utc(), returncode=None,
                          error={'type': type(error).__name__, 'message': str(error)},
                          stdout='', stderr='')
            archive.exclusive_json(path, record)
            raise
        record.update(finished_utc=archive.utc(), returncode=child.returncode,
                      stdout=child.stdout, stderr=child.stderr)
        archive.exclusive_json(path, record)
        if child.returncode:
            raise RuntimeError(f'Controller operation failed: {path.name}')
        return child.stdout

    def exec_remote(filename, script_args):
        command = ['exec-file', *common, '--file', str(source / 'runtime' / filename),
                   '--timeout', '120']
        command.extend('--script-arg=' + str(arg) for arg in script_args)
        return call(command, timeout=180)

    details, status = {}, 'failed'
    launch_attempted = False
    launch_acknowledged = False
    last_progress = None
    completion_seen_at = None
    try:
        remote_source = '/content/Strassen_MM_Focus/' + run.name + '-source.tar'
        call(['upload', *common, '--local', str(run / 'source.tar'), '--remote', remote_source], timeout=240)
        # A timeout or lost response after this point cannot prove no worker started.
        launch_attempted = True
        response = exec_remote('launch_phase_v002.py', [
            '--archive', remote_source, '--archive-sha256', digest(run / 'source.tar'),
            '--run-id', run.name, '--phase', args.phase,
            '--campaign-relative', 'configs/campaign_v1.json',
            '--expected-identity', args.expected_identity, '--allocation-id', args.endpoint,
            '--timeout-seconds', args.timeout_seconds])
        launch = remote_object(response, 'phase_launched')
        archive.exclusive_json(run / 'remote-launch.json', launch)
        launch_acknowledged = True
        emit('campaign', name='campaign', state=args.phase + ' running on v5e',
             detail='Frozen source committed; same allocation enforced. ' + run.name)
        offsets = {'log': 0, 'results': 0, 'worker_log': 0}
        local_paths = {'log': run / 'benchmark.log', 'results': run / 'results.jsonl',
                       'worker_log': run / 'worker.log'}
        for path in local_paths.values():
            path.touch(exist_ok=False)
        consecutive_errors = 0
        while True:
            remaining_time()
            try:
                response = exec_remote('poll_phase_v002.py', [
                    '--run-id', run.name, '--log-offset', offsets['log'],
                    '--result-offset', offsets['results'], '--worker-log-offset', offsets['worker_log'],
                    '--max-bytes', '65536'])
                progress = remote_object(response, 'phase_progress')
                if progress.get('transport') != 'base64_bytes_v002':
                    raise RuntimeError('Unexpected polling transport version')
                last_progress = {key: progress.get(key) for key in (
                    'run_id', 'phase', 'worker_alive', 'benchmark_alive',
                    'completion', 'archive_ready', 'archive_error', 'summary')}
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                print(f'{args.phase}: progress query failed ({consecutive_errors}); same allocation retained', flush=True)
                if consecutive_errors >= 5:
                    raise
                bounded_sleep(15)
                continue
            for key, path in local_paths.items():
                chunk = progress[key]
                if chunk.get('error'):
                    raise RuntimeError(key + ': ' + chunk['error'])
                data = base64.b64decode(chunk['data_base64'], validate=True)
                if chunk['offset'] != offsets[key] or chunk['next_offset'] != offsets[key] + len(data):
                    raise RuntimeError(key + ': remote byte offsets do not match the received chunk')
                if data:
                    with path.open('ab') as stream:
                        stream.write(data)
                    if key == 'log':
                        # Display decoding never changes the exact archived byte stream.
                        print(data.decode('utf-8', errors='replace'), end='', flush=True)
                offsets[key] = chunk['next_offset']
            if valid_record(progress.get('archive_error')):
                raise RuntimeError('Remote archive failed; preserve allocation for recovery')
            if valid_record(progress.get('archive_ready')):
                ready = progress['archive_ready']
                package = run / 'remote-run.tar.gz'
                call(['download', *common, '--remote', ready['archive'], '--local', str(package)], timeout=300)
                if digest(package) != ready['sha256']:
                    raise RuntimeError('Downloaded archive SHA256 mismatch')
                archive.exclusive_json(run / 'remote-archive.json', ready)
                with tarfile.open(package) as packed:
                    packed.extractall(run / 'remote', filter='data')
                remote = run / 'remote' / run.name
                if (remote / 'artifacts').exists():
                    shutil.copytree(remote / 'artifacts', run / 'artifacts')
                for key, name in [('log', 'benchmark.log'), ('results', 'artifacts/results.jsonl'),
                                  ('worker_log', 'worker.log')]:
                    canonical = remote / name
                    if canonical.exists():
                        data = canonical.read_bytes()
                        prefix = local_paths[key].read_bytes()
                        if not data.startswith(prefix):
                            raise RuntimeError('Live copy differs from canonical remote artifact: ' + name)
                        with local_paths[key].open('ab') as stream:
                            stream.write(data[len(prefix):])
                # Final status comes from the verified canonical archive, not a transient poll.
                completion = json.loads((remote / 'completion.json').read_text())
                summary_path = remote / 'artifacts' / 'summary.json'
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
                details.update(remote_completion=completion, summary=summary,
                               archive_sha256=ready['sha256'], allocation_id=args.endpoint,
                               remote_may_still_be_running=False)
                status = completion['status']
                break
            has_completion = valid_record(progress.get('completion'))
            if has_completion:
                if completion_seen_at is None:
                    completion_seen_at = time.monotonic()
                if time.monotonic() - completion_seen_at >= args.archive_grace_seconds:
                    raise TimeoutError('Worker completion has no archive after bounded grace; remote evidence retained')
            elif progress.get('worker_alive') is False:
                raise RuntimeError('Worker exited without completion; remote evidence retained')
            bounded_sleep(1 if progress['results'].get('more') else 15)
    except BaseException as error:
        details['error'] = {'type': type(error).__name__, 'message': str(error)}
        details['remote_may_still_be_running'] = launch_attempted
        if last_progress is not None:
            details['last_remote_progress'] = last_progress
        print(json.dumps(details['error']), flush=True)
    details.update(launch_attempted=launch_attempted, launch_acknowledged=launch_acknowledged,
                   orchestration_wall_seconds=time.monotonic() - orchestration_started,
                   orchestration_budget_seconds=args.timeout_seconds + grace_seconds,
                   archive_grace_seconds=args.archive_grace_seconds)
    revision = archive.finish(run, status, **details)
    emit('campaign', name='campaign', state=args.phase + ' ' + status,
         detail='Execution archived and committed as ' + revision[:7] + '. ' +
                json.dumps((details.get('summary') or {}).get('case_status_counts', {})))
    print(json.dumps({'run_id': run.name, 'status': status, 'commit': revision, **details}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
