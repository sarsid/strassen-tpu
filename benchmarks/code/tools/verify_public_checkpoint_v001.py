"""Verify a portable benchmarks checkpoint without credentials or accelerator access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def sha(path):
    with path.open('rb') as f:
        return digest(f)


def read(path):
    return json.loads(path.read_text())


def verify(root):
    manifest = read(root / 'MANIFEST.json')
    assert manifest['export_complete'] is True
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store'}
    assert actual == set(manifest['files']) | {'MANIFEST.json'}, 'Unexpected or missing files'
    for name, expected in manifest['files'].items():
        p = root / name
        assert not p.is_symlink() and p.stat().st_size == expected['bytes'], name
        assert sha(p) == expected['sha256'], name
    index = read(root / 'results/v6e/phase-index.json')
    identities = {}
    member_count = 0
    for phase in index['phases']:
        path = root / phase['bundle']
        assert sha(path) == phase['bundle_sha256'], phase['phase']
        with tarfile.open(path) as archive:
            names = archive.getnames()
            assert len(names) == len(set(names))
            assert set(names) == set(phase['member_sha256'])
            for member in archive:
                parts = PurePosixPath(member.name)
                assert member.isfile() and not parts.is_absolute() and '..' not in parts.parts
                with archive.extractfile(member) as stream:
                    assert digest(stream) == phase['member_sha256'][member.name], member.name
                member_count += 1
            env = json.load(archive.extractfile('artifacts/environment.json'))
            if phase['kind'] == 'comparison':
                unit = phase['phase'].rsplit('-', 1)[0]
                identities.setdefault(unit, []).append(env['identity'])
    for unit, values in identities.items():
        assert len(values) == 2 and values[0] == values[1], unit
    checkpoint = read(root / 'checkpoint.json')
    assert len(identities) == checkpoint['v6e']['completed_shapes'] == 113
    assert len(checkpoint['v6e']['remaining_units']) == 55
    for item in manifest['compressed_reports']:
        with gzip.open(root / item['file'], 'rb') as stream:
            assert digest(stream) == item['uncompressed_sha256'], item['file']
    return dict(passed=True, files=len(manifest['files']), v6e_complete_pairs=len(identities),
                v6e_phase_members=member_count, full_study_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), sort_keys=True))
