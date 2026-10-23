"""Set up one owned TPU from an immutable archive_scoped_v001 execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--hardware', choices=['v5e', 'v6e'], required=True)
    p.add_argument('--controller-python', required=True)
    a = p.parse_args()
    run = Path(os.environ['STRASSEN_EXECUTION_DIR']).resolve()
    source = Path(__file__).resolve().parents[1]
    if source != run / 'source':
        raise ValueError('Execute the committed source inside an archive')
    artifacts = run / 'artifacts'
    artifacts.mkdir()
    version = 'v003' if a.hardware == 'v6e' else 'v002'
    setup = 'setup_v003.py' if a.hardware == 'v6e' else 'setup_v001.py'
    controller = [a.controller_python, str(source / 'runtime' / ('colab_control_' + version + '.py'))]
    common = ['--session', a.session, '--expect-endpoint', a.endpoint]
    remote = '/content/Strassen_MM_Focus/' + run.name
    command = controller + ['exec-file', *common, '--file', str(source / 'runtime' / setup),
        '--timeout', '1500', '--script-arg=--output-dir', '--script-arg=' + remote,
        '--script-arg=--expected-endpoint', '--script-arg=' + a.endpoint]
    (artifacts / 'commands.json').write_text(json.dumps({'setup': command, 'remote_output': remote}, indent=2))
    with (artifacts / 'setup.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1620)
    package = artifacts / 'remote-setup.tar.gz'
    with (artifacts / 'download.log').open('x') as log:
        download = subprocess.run(controller + ['download', *common, '--remote', remote + '.tar.gz',
            '--local', str(package)], stdout=log, stderr=subprocess.STDOUT, timeout=240)
    details = dict(setup_exit_code=result.returncode, download_exit_code=download.returncode,
                   hardware=a.hardware, endpoint=a.endpoint, remote_identity=remote + '/identity.json')
    if download.returncode == 0:
        details['archive_sha256'] = hashlib.sha256(package.read_bytes()).hexdigest()
        with tarfile.open(package) as packed:
            packed.extractall(artifacts, filter='data')
    identity = artifacts / run.name / 'identity.json'
    details['completed'] = result.returncode == 0 and download.returncode == 0 and identity.is_file()
    (artifacts / 'summary.json').write_text(json.dumps(details, indent=2))
    print(json.dumps(details), flush=True)
    return 0 if details['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
