"""Archive setup of an already allocated v5e, including downloaded evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import archive_v002 as archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--controller-python', required=True)
    args = parser.parse_args()
    run = archive.begin('setup-v5e-v001', scope='runtime_setup')
    print('EXECUTION_DIR=' + str(run), flush=True)
    source = run / 'source'
    controller = [args.controller_python, str(source / 'runtime/colab_control_v001.py')]
    common = ['--session', args.session, '--expect-endpoint', args.endpoint]
    remote = '/content/Strassen_MM_Focus/' + run.name
    command = controller + ['exec-file', *common, '--file', str(source / 'runtime/setup_v001.py'),
                            '--timeout', '1500', '--script-arg=--output-dir',
                            '--script-arg=' + remote, '--script-arg=--expected-endpoint',
                            '--script-arg=' + args.endpoint]
    archive.exclusive_json(run / 'commands.json', {'setup': command, 'remote_output': remote})
    status, details = 'failed', {}
    try:
        with (run / 'execution.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        details['setup_exit_code'] = result.returncode
        package = run / 'remote-setup.tar.gz'
        with (run / 'download.log').open('x') as log:
            downloaded = subprocess.run(controller + ['download', *common,
                '--remote', remote + '.tar.gz', '--local', str(package)],
                stdout=log, stderr=subprocess.STDOUT)
        details['download_exit_code'] = downloaded.returncode
        if downloaded.returncode == 0:
            details['archive_sha256'] = hashlib.sha256(package.read_bytes()).hexdigest()
            with tarfile.open(package) as packed:
                packed.extractall(run / 'artifacts', filter='data')
            identity_path = run / 'artifacts' / run.name / 'identity.json'
            if identity_path.exists():
                details['identity_path'] = str(identity_path.relative_to(archive.ROOT))
            if result.returncode == 0:
                status = 'completed'
    except Exception as error:
        details['error'] = {'type': type(error).__name__, 'message': str(error)}
    revision = archive.finish(run, status, **details)
    print(json.dumps({'run_id': run.name, 'status': status, 'commit': revision, **details}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
