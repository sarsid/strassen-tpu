"""Rebuild the saved v6e report from the portable raw measurements; no TPU calls."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile


def read(path):
    return json.loads(path.read_text())


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Optional JSON validation receipt, outside the checkpoint')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from verify import verify
    checks = verify(root)
    sys.path.insert(0, str(root / 'code/tools'))
    from report_arch_study_v002 import build
    import numpy as np
    index = read(root / 'results/v6e/phase-index.json')
    saved_report = root / 'results/v6e/report'
    with tempfile.TemporaryDirectory(prefix='strassen-report-replay-') as tmp:
        tmp = Path(tmp)
        cohort = tmp / 'view'
        config = 'configs/arch_v6e_168_v001'
        shutil.copytree(root / 'code' / config, cohort / 'source' / config)
        for phase in index['phases']:
            if phase['kind'] != 'comparison':
                continue
            run = tmp / 'phases' / phase['phase']
            # Only report inputs are needed; compiled diagnostics remain in the bundle.
            wanted = {'artifacts/results.jsonl', 'artifacts/selections.json',
                      'artifacts/environment.json', 'artifacts/planned_cases.json'}
            with tarfile.open(root / phase['bundle']) as archive:
                for member in archive:
                    if member.name not in wanted:
                        continue
                    path = run / member.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open('xb') as dest:
                        shutil.copyfileobj(source, dest)
            (cohort / (phase['phase'] + '-finished.json')).write_text(
                json.dumps(dict(status='completed', run=str(run))))
        out = tmp / 'report'
        exit_code = build(cohort, out)
        assert exit_code == 1, 'A partial checkpoint must not claim full completion'
        with gzip.open(saved_report / 'results.json.gz', 'rt') as stream:
            saved = json.load(stream)
        rebuilt = read(out / 'results.json')
        # Receipt paths/hashes change in the temporary portable view. Every
        # scientific field, including all paired bootstrap intervals, must match.
        saved.pop('input_sha256')
        rebuilt.pop('input_sha256')
        assert saved == rebuilt, 'Scientific report differs'
        for name in ('candidate_decisions.json',):
            with (out / name).open('rb') as a, gzip.open(saved_report / (name + '.gz'), 'rb') as b:
                assert digest(a) == digest(b), name
        assert (out / 'tuner_policy.json').read_bytes() == (saved_report / 'tuner_policy.json').read_bytes()
        checks.update(report_replay='passed', scientific_fields='identical',
                      candidate_ledger='byte_identical', policy='byte_identical',
                      precision_groups=len(rebuilt['results']), numpy_version=np.__version__)
    if args.output:
        path = args.output.resolve()
        assert not path.is_relative_to(root), 'Write validation outside immutable checkpoint'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, sort_keys=True))


if __name__ == '__main__':
    main()
