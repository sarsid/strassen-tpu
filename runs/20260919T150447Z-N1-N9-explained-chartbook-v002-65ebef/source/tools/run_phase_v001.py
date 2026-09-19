"""Execute one phase from a frozen Git snapshot on an existing TPU allocation.

Remote progress is copied append-only. Final remote evidence is downloaded,
verified and committed before another phase is allowed to begin.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'N1', 'N2', 'N3', 'N4'])
    parser.add_argument('--session', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--expected-identity', required=True)
    parser.add_argument('--controller-python', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    args = parser.parse_args()
    run = archive.begin(args.phase + '-v5e-v001', scope='tpu_benchmark')
    print('EXECUTION_DIR=' + str(run), flush=True)
    source = run / 'source'
    controller = [args.controller_python, str(source / 'runtime/colab_control_v001.py')]
    common = ['--session', args.session, '--expect-endpoint', args.endpoint]
    calls = run / 'controller'
    calls.mkdir()
    counter = 0

    def call(command, **kwargs):
        nonlocal counter
        counter += 1
        path = calls / f'{counter:05d}.json'
        started = archive.utc()
        child = subprocess.run(controller + command, capture_output=True, text=True, **kwargs)
        archive.exclusive_json(path, {'started_utc': started, 'finished_utc': archive.utc(),
            'command': command, 'returncode': child.returncode,
            'stdout': child.stdout, 'stderr': child.stderr})
        if child.returncode:
            raise RuntimeError(f'Controller operation failed: {path.name}')
        return child.stdout

    def exec_remote(filename, script_args):
        command = ['exec-file', *common, '--file', str(source / 'runtime' / filename),
                   '--timeout', '120']
        command.extend('--script-arg=' + str(arg) for arg in script_args)
        return call(command, timeout=180)

    details, status = {}, 'failed'
    launched = False
    try:
        remote_source = '/content/Strassen_MM_Focus/' + run.name + '-source.tar'
        call(['upload', *common, '--local', str(run / 'source.tar'), '--remote', remote_source], timeout=240)
        response = exec_remote('launch_phase_v002.py', [
            '--archive', remote_source, '--archive-sha256', digest(run / 'source.tar'),
            '--run-id', run.name, '--phase', args.phase,
            '--campaign-relative', 'configs/campaign_v1.json',
            '--expected-identity', args.expected_identity, '--allocation-id', args.endpoint,
            '--timeout-seconds', args.timeout_seconds])
        launch = remote_object(response, 'phase_launched')
        archive.exclusive_json(run / 'remote-launch.json', launch)
        launched = True
        emit('campaign', name='campaign', state=args.phase + ' running on v5e',
             detail='Frozen source committed; same allocation enforced. ' + run.name)
        offsets = {'log': 0, 'results': 0, 'worker_log': 0}
        local_paths = {'log': run / 'benchmark.log', 'results': run / 'results.jsonl',
                       'worker_log': run / 'worker.log'}
        for path in local_paths.values():
            path.touch(exist_ok=False)
        consecutive_errors = 0
        while True:
            try:
                response = exec_remote('poll_phase_v001.py', [
                    '--run-id', run.name, '--log-offset', offsets['log'],
                    '--result-offset', offsets['results'], '--worker-log-offset', offsets['worker_log'],
                    '--max-bytes', '65536'])
                progress = remote_object(response, 'phase_progress')
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                print(f'{args.phase}: progress query failed ({consecutive_errors}); same allocation retained', flush=True)
                if consecutive_errors >= 5:
                    raise
                time.sleep(15)
                continue
            for key, path in local_paths.items():
                chunk = progress[key]
                if chunk.get('error'):
                    raise RuntimeError(key + ': ' + chunk['error'])
                if chunk.get('text'):
                    with path.open('a') as stream:
                        stream.write(chunk['text'])
                    if key == 'log':
                        print(chunk['text'], end='', flush=True)
                offsets[key] = chunk['next_offset']
            if progress.get('archive_error'):
                raise RuntimeError('Remote archive failed; preserve allocation for recovery')
            if progress.get('archive_ready'):
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
                details.update(remote_completion=progress['completion'], summary=progress['summary'],
                               archive_sha256=ready['sha256'], allocation_id=args.endpoint)
                status = progress['completion']['status']
                break
            if progress.get('worker_alive') is False and not progress.get('completion'):
                raise RuntimeError('Worker exited without completion; remote evidence retained')
            time.sleep(1 if progress['results'].get('more') else 15)
    except BaseException as error:
        details['error'] = {'type': type(error).__name__, 'message': str(error)}
        details['remote_may_still_be_running'] = launched
        print(json.dumps(details['error']), flush=True)
    revision = archive.finish(run, status, **details)
    emit('campaign', name='campaign', state=args.phase + ' ' + status,
         detail='Execution archived and committed as ' + revision[:7] + '. ' +
                json.dumps(details.get('summary', {}).get('case_status_counts', {})))
    print(json.dumps({'run_id': run.name, 'status': status, 'commit': revision, **details}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
