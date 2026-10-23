"""Smaller-tile retry adapter: one exploratory phase with 30 timing rounds.

No kernel or benchmark timing logic changes. Setup identity has fewer fields
than a measured-run identity, so validate its shared fields explicitly; screen
then captures the complete identity used without relaxation by confirmation.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate_provisioning(expected, allocation_id, hostname, boot_id, versions):
    current = {'colab_endpoint': allocation_id, 'hostname': hostname,
               'boot_id': boot_id, 'versions': versions}
    mismatch = {key: {'expected': expected.get(key), 'actual': value}
                for key, value in current.items() if expected.get(key) != value}
    if mismatch:
        raise ValueError('Provisioning identity mismatch: ' + json.dumps(mismatch, sort_keys=True))
    return current


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('SOPT-screen', 'SOPT-confirm'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-identity', required=True, type=Path)
    parser.add_argument('--allocation-id', required=True)
    parser.add_argument('--max-wall-seconds', required=True, type=float)
    parser.add_argument('--prior-run', type=Path)
    args = parser.parse_args()
    phase = args.phase.split('-')[1]
    if (phase == 'confirm') != bool(args.prior_run):
        parser.error('Confirmation requires prior-run; screening must not supply it')
    prior = json.loads(args.expected_identity.read_text())
    prior = prior.get('identity', prior)
    provisioning = 'runtime_flags' not in prior
    if provisioning:
        if phase != 'screen':
            raise ValueError('Confirmation must use complete measured-run identity')
        versions = {name: importlib.metadata.version(name) for name in prior['versions']}
        actual = validate_provisioning(prior, args.allocation_id, socket.gethostname(),
                   Path('/proc/sys/kernel/random/boot_id').read_text().strip(), versions)
        write(args.output_dir.parent / 'provisioning-before-benchmark.json',
              {'matched': True, 'actual': actual, 'expected_identity': str(args.expected_identity)})
    command = [sys.executable, str(ROOT / 'tools/benchmark_strassen_optimized_v001.py'),
               '--phase', phase, '--output-dir', str(args.output_dir),
               '--allocation-id', args.allocation_id, '--max-wall-seconds', str(args.max_wall_seconds)]
    command += ['--repeats', '30']
    if not provisioning:
        command += ['--expected-identity', str(args.expected_identity)]
    if phase == 'screen':
        command += ['--shapes-json', str(args.campaign)]
    else:
        command += ['--prior-run', str(args.prior_run)]
    code = subprocess.call(command)
    if provisioning and (args.output_dir / 'environment.json').is_file():
        environment = json.loads((args.output_dir / 'environment.json').read_text())
        current = environment['identity']
        fields = ('colab_endpoint', 'hostname', 'boot_id', 'versions', 'devices')
        mismatches = {key: {'expected': prior.get(key), 'actual': current.get(key)}
                      for key in fields if prior.get(key) != current.get(key)}
        matched = not mismatches and environment['qualified_single_v5e']
        write(args.output_dir.parent / 'provisioning-after-benchmark.json',
              {'matched': matched, 'mismatches': mismatches,
               'qualified_single_v5e': environment['qualified_single_v5e']})
        if not matched:
            raise RuntimeError('Hardware identity failed qualification; reject performance comparisons')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
