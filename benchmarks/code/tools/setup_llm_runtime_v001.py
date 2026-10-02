"""Run pinned setup and retrieve its evidence inside a scoped execution archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--endpoint', required=True)
    args = parser.parse_args()
    run = Path(os.environ['STRASSEN_EXECUTION_DIR'])
    source = Path(__file__).resolve().parents[1]
    controller = [sys.executable, str(source / 'runtime/colab_control_v001.py')]
    common = ['--session', args.session, '--expect-endpoint', args.endpoint]
    remote = '/content/Strassen_MM_Focus/' + run.name
    command = controller + ['exec-file', *common, '--file', str(source / 'runtime/setup_v001.py'),
        '--timeout', '1500', '--script-arg=--output-dir', '--script-arg=' + remote,
        '--script-arg=--expected-endpoint', '--script-arg=' + args.endpoint]
    with (run / 'setup-command.json').open('x') as stream:
        json.dump({'argv': command, 'remote_identity': remote + '/identity.json'}, stream, indent=2)
    with (run / 'setup.log').open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1560)
    package = run / 'remote-setup.tar.gz'
    with (run / 'download.log').open('x') as stream:
        download = subprocess.run(controller + ['download', *common, '--remote', remote + '.tar.gz',
            '--local', str(package)], stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    if download.returncode:
        raise RuntimeError('Setup archive retrieval failed; inspect preserved download.log')
    with tarfile.open(package) as archive:
        archive.extractall(run / 'artifacts', filter='data')
    artifacts = run / 'artifacts' / run.name
    manifest = json.loads((artifacts / 'artifact-manifest.json').read_text())
    for name, entry in manifest.items():
        path = artifacts / name
        if not path.resolve().is_relative_to(artifacts.resolve()):
            raise ValueError('Invalid manifest path')
        if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Setup artifact checksum mismatch: ' + name)
    status = json.loads((artifacts / 'status.json').read_text())
    if result.returncode or status['status'] != 'completed':
        raise RuntimeError('Setup failed; retrieved evidence retained')
    identity = json.loads((artifacts / 'identity.json').read_text())
    if identity['colab_endpoint'] != args.endpoint or identity['backend'] != 'tpu' or identity['device_count'] != 1:
        raise ValueError('Setup identity mismatch')
    print(json.dumps({'status': 'completed', 'remote_identity': remote + '/identity.json',
        'local_identity': str(artifacts / 'identity.json'), 'archive_sha256': hashlib.sha256(package.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
