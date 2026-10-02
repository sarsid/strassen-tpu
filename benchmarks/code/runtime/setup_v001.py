#!/usr/bin/env python3
"""Pin and probe a single existing Colab v5e runtime without importing JAX here.

Run once per allocation, supplying a fresh output directory. No allocation,
restart, deletion, or experiment launch occurs. Every outcome is archived.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile

PINS = {'jax': '0.7.2', 'jaxlib': '0.7.2', 'libtpu': '0.0.21.1'}
PACKAGES = tuple(PINS) + ('numpy', 'ml_dtypes', 'scipy', 'requests')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def sanitize(value):
    value = re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1[REDACTED]@', str(value))
    return re.sub(r'(?i)((?:token|authorization|secret)\s*[:=]\s*)[^\s,]+', r'\1[REDACTED]', value)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-endpoint', required=True)
    args = parser.parse_args()
    base = Path('/content/Strassen_MM_Focus')
    output = args.output_dir
    if not output.is_absolute() or not output.is_relative_to(base) or '..' in output.parts or output == base:
        raise ValueError('Use a unique absolute directory beneath /content/Strassen_MM_Focus')
    base.mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    source = globals().get('__source_text__')
    if source is None:
        source = Path(__file__).read_text()
    with (output / 'setup_v001.py').open('x') as stream:
        stream.write(source)
    identity = {'captured_utc': utc_now(), 'colab_endpoint': args.expected_endpoint,
                'hostname': platform.node(), 'platform': platform.platform(),
                'python': platform.python_version(), 'python_executable': sys.executable,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'setup_sha256': hashlib.sha256(source.encode()).hexdigest()}
    write_json(output / 'before.json', {**identity, 'versions': versions()})
    status = {'started_utc': utc_now(), 'status': 'running', 'pins': PINS}
    success = False
    try:
        command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install', '--no-input']
        command += [name + '==' + version for name, version in PINS.items()]
        write_json(output / 'install-command.json', {'argv': command})
        install = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=1200)
        with (output / 'install.log').open('x') as stream:
            stream.write(sanitize(install.stdout))
        print(sanitize(install.stdout), flush=True)
        if install.returncode:
            raise RuntimeError('Package installation failed; inspect install.log')
        observed = versions()
        write_json(output / 'versions.json', observed)
        if any(observed[name] != version for name, version in PINS.items()):
            raise RuntimeError('Installed versions did not match pins')
        probe_source = '''import jax, json
items = []
for device in jax.devices():
    item = {'kind': device.device_kind, 'id': device.id, 'platform': device.platform,
            'process_index': device.process_index}
    for name in ('coords', 'core_on_chip', 'slice_index', 'local_hardware_id'):
        value = getattr(device, name, None)
        if value is not None: item[name] = value
    items.append(item)
value = {'backend': jax.default_backend(), 'device_count': len(items), 'devices': items}
print(json.dumps(value), flush=True)
assert value['backend'] == 'tpu' and len(items) == 1
assert items[0]['kind'] == 'TPU v5 lite', items[0]['kind']
'''
        with (output / 'probe.py').open('x') as stream:
            stream.write(probe_source)
        probe = subprocess.run([sys.executable, str(output / 'probe.py')],
                               capture_output=True, text=True, timeout=240)
        with (output / 'probe.stdout.log').open('x') as stream:
            stream.write(sanitize(probe.stdout))
        with (output / 'probe.stderr.log').open('x') as stream:
            stream.write(sanitize(probe.stderr))
        print(sanitize(probe.stdout), flush=True)
        if probe.returncode:
            print(sanitize(probe.stderr), flush=True)
            raise RuntimeError('TPU probe failed; inspect probe logs')
        hardware = json.loads(probe.stdout.strip().splitlines()[-1])
        write_json(output / 'identity.json', {**identity, 'versions': observed, **hardware})
        status.update(status='completed', identity_fields=['colab_endpoint', 'hostname', 'boot_id',
                                                         'versions', 'backend', 'device_count', 'devices'])
        success = True
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, error=sanitize(str(error)))
    finally:
        status['finished_utc'] = utc_now()
        write_json(output / 'status.json', status)
        manifest = {str(path.relative_to(output)): {'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(output.rglob('*')) if path.is_file()}
        write_json(output / 'artifact-manifest.json', manifest)
        archive = output.with_name(output.name + '.tar.gz')
        with archive.open('xb') as destination:
            with tarfile.open(fileobj=destination, mode='w:gz') as package:
                package.add(output, arcname=output.name)
        print(json.dumps({'kind': 'setup_finished', 'status': status['status'],
                          'output_dir': str(output), 'archive': str(archive),
                          'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}), flush=True)
    if not success:
        raise RuntimeError('Runtime setup failed; preserved outputs identify the cause')


if __name__ == '__main__':
    main()
