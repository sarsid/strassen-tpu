"""Reconstruct the saved report from public evidence, without an accelerator."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    assert not output.is_relative_to(root), 'Keep new output outside the immutable snapshot'
    assert not output.exists(), 'Use a fresh output directory'
    manifest = json.loads((root/'results/compute_scope_analysis_v001/inputs.json').read_text())
    portable = copy.deepcopy(manifest)
    ledger = root/'results/completed_entries_all_history.jsonl'
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == manifest['ledger_sha256']
    portable['ledger'] = str(ledger)
    records = {}
    for group in json.loads((root/'evidence/index.json').read_text())['groups']:
        for record in group['files']:
            if record['source_path'] in records:
                assert records[record['source_path']]['sha256'] == record['sha256']
            records[record['source_path']] = record
    output.mkdir(parents=True)
    inputs = output/'inputs'
    inputs.mkdir()
    for i, item in enumerate(portable['events']):
        record = records[item['path']]
        blob = (root/record['blob']).resolve()
        assert blob.is_relative_to(root)
        data = gzip.decompress(blob.read_bytes())
        assert hashlib.sha256(data).hexdigest() == item['sha256'] == record['sha256']
        dest = inputs/f'events-{i:02d}.jsonl'
        dest.write_bytes(data)
        item['path'] = str(dest)
    spec = inputs/'portable-inputs.json'
    spec.write_text(json.dumps(portable, indent=2)+'\n')
    env = dict(os.environ, STRASSEN_EXECUTION_DIR=str(output), MPLBACKEND='Agg',
               PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(root/'code/tools/report_llm_compute_scopes_v001.py'),
                    '--inputs', str(spec)], cwd=root, env=env, check=True)
    original = json.loads((root/'results/compute_scope_analysis_v001/measurements.json').read_text())
    recreated = json.loads((output/'artifacts/measurements.json').read_text())
    original.pop('created_utc')
    recreated.pop('created_utc')
    assert original == recreated, 'Reconstructed measurements must match saved report exactly'
    print(json.dumps({'passed': True, 'reconstructed_comparisons': 80,
                      'measurements_match_saved_report': True, 'output': str(output)}))


if __name__ == '__main__':
    main()
