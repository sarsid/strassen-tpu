"""Verify saved complete tasks and report them without inventing missing tasks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
from accuracy_analysis_v003 import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify(checkpoints, inputs, source_archive, admission):
    manifest = json.loads((checkpoints/'manifest.json').read_text())
    contract = json.loads((checkpoints/'execution-contract.json').read_text())
    original = json.loads((inputs/'manifest.json').read_text())
    if sha(source_archive) != contract['bundle_sha256']:
        raise ValueError('Changed scientific source bundle')
    with tarfile.open(source_archive) as archive:
        for name, expected in contract['source_sha256'].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError('Changed scientific source file')
    contract_hash = digest(contract)
    if contract_hash != manifest['execution_contract_sha256'] or manifest['admission_status'] != 'passed':
        raise ValueError('Invalid execution contract or admission')
    if admission['status'] != 'passed' or not admission['all_logits_finite']:
        raise ValueError('Missing passed runtime admission')
    if sha(inputs/'inputs.jsonl') != original['inputs_sha256']:
        raise ValueError('Changed input file')
    for key in ('examples', 'inputs_sha256', 'counts', 'checkpoint_revision', 'model'):
        if manifest[key] != original[key]:
            raise ValueError('Changed input manifest: '+key)
    rows = [json.loads(line) for line in (inputs/'inputs.jsonl').read_text().splitlines()]
    for row in rows:
        if digest({k:v for k,v in row.items() if k not in ('gold','input_sha256')}) != row['input_sha256']:
            raise ValueError('Changed input row')
    collected = {'native': [], 'strassen': []}
    counts = {}
    included = []
    for task, total in original['counts'].items():
        task_rows = [r for r in rows if r['task'] == task]
        paths = [checkpoints/'batches'/f'{task}-{first//8:03d}.json' for first in range(0, total, 8)]
        present = [p for p in paths if p.exists()]
        if not present:
            continue
        if len(present) != len(paths):
            raise ValueError('This snapshot requires whole tasks: '+task)
        counts[task] = total
        for p, first in zip(paths, range(0, total, 8), strict=True):
            batch = json.loads(p.read_text()); chunk = task_rows[first:first+8]
            if batch['task'] != task or batch['example_ids'] != [r['example_id'] for r in chunk] or batch['execution_contract_sha256'] != contract_hash:
                raise ValueError('Batch order, task or contract mismatch')
            for arm in collected:
                for row, metrics in zip(chunk, batch['results'][arm], strict=True):
                    common = {k:row.get(k) for k in ('task','example_id','subject','gold','input_sha256')}
                    collected[arm].append(dict(common, arm=arm, checkpoint_revision=manifest['checkpoint_revision'],
                        protocol_sha256=manifest['protocol_sha256'], execution_contract_sha256=contract_hash, **metrics))
            included.append(p)
    if set(checkpoints.glob('batches/*.json')) != set(included):
        raise ValueError('Unexpected checkpoint files')
    if not counts:
        raise ValueError('No complete tasks')
    return manifest, contract, collected, counts, included


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); admission=json.loads(a.admission.read_text())
    manifest, contract, collected, counts, files=verify(a.checkpoints,a.inputs,a.source_archive,admission)
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'batches').mkdir()
    for name in ('manifest.json','execution-contract.json'):
        shutil.copyfile(a.checkpoints/name,a.output/name)
    shutil.copyfile(a.admission,a.output/'runtime-admission.json')
    for path in files: shutil.copyfile(path,a.output/'batches'/path.name)
    derived=dict(manifest, examples=[x for x in manifest['examples'] if x['task'] in counts], counts=counts,
        analysis_scope='Complete saved tasks only; original full manifest retained separately',
        missing_tasks=sorted(set(manifest['counts'])-set(counts)))
    save(a.output/'analysis-manifest.json',derived)
    for arm,rows in collected.items():
        with (a.output/(arm+'.jsonl')).open('x') as f:
            for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    report=analyze(derived,a.output/'native.jsonl',a.output/'strassen.jsonl')
    save(a.output/'summary.json',dict(report,missing_tasks=derived['missing_tasks'],evaluation_scope=derived['analysis_scope']))
    save(a.output/'verification.json',dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        completed_tasks=counts,missing_tasks=derived['missing_tasks'],model_complete=False,
        original_manifest_sha256=sha(a.checkpoints/'manifest.json'),
        source_bundle_sha256=sha(a.source_archive),execution_contract_sha256=digest(contract),
        runtime_admission_passed=True,input_and_source_hashes_verified=True))
    save(a.output/'artifact-manifest.json',dict(files={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(tasks=report['tasks'],missing_tasks=derived['missing_tasks']),indent=2))


if __name__=='__main__': main()
