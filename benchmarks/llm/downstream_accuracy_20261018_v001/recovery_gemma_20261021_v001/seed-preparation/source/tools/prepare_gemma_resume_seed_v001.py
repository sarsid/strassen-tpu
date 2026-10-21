"""Preserve both complete Qwen models and the exact saved Gemma MMLU prefix."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from resume_accuracy_gemma_remote_v001 import COMPLETED, digest, inspect_seed, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior-run', type=Path, required=True)
    p.add_argument('--output-run', type=Path, required=True)
    a = p.parse_args()
    source = a.output_run / 'source.tar.gz'
    source_sha = sha(source)
    if source_sha != 'ff715440fe67ce95f16b118f8694cbcf2f8cd9e0df8f1ea37c812d5c425c1495':
        raise ValueError('Expected unchanged scientific source')
    checkpoints = a.prior_run / 'checkpoints/gemma4_31b'
    manifest = json.loads((checkpoints / 'manifest.json').read_text())
    contract = json.loads((checkpoints / 'execution-contract.json').read_text())
    with tarfile.open(source) as archive:
        original = json.loads(archive.extractfile('plans/llm_accuracy_20261018_v001/subset_gemma4_31b_v002/manifest.json').read())
        raw_inputs = archive.extractfile('plans/llm_accuracy_20261018_v001/subset_gemma4_31b_v002/inputs.jsonl').read()
        if hashlib.sha256(raw_inputs).hexdigest() != original['inputs_sha256']:
            raise ValueError('Input archive mismatch')
        for name, expected in contract['source_sha256'].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError('Changed scientific source')
    for key in ('examples', 'inputs_sha256', 'counts', 'checkpoint_revision', 'model'):
        if manifest[key] != original[key]:
            raise ValueError('Changed original manifest field: ' + key)
    if digest(contract) != manifest['execution_contract_sha256'] or contract['bundle_sha256'] != source_sha:
        raise ValueError('Changed execution contract')
    files = {}
    for model, expected in COMPLETED.items():
        archive = a.prior_run / (model + '.tar.gz')
        receipt = json.loads(archive.with_suffix('.verified.json').read_text())
        if sha(archive) != expected or receipt['sha256'] != expected or receipt['bytes'] != archive.stat().st_size:
            raise ValueError('Completed model archive is not verified')
        files['completed/' + archive.name] = archive.read_bytes()
    for path in sorted(checkpoints.rglob('*')):
        if path.is_file():
            files['artifacts/gemma4_31b/' + str(path.relative_to(checkpoints))] = path.read_bytes()
    evidence = dict(
        original_run=a.prior_run.name,
        original_admission_receipt_available=False,
        evidence_available='Saved manifest records admission_status=passed; exact contract and paired batch files are present',
        limitation='Detailed original runtime admission and preflight numeric receipts were not downloaded before provider loss',
        original_manifest_sha256=sha(checkpoints / 'manifest.json'),
        original_execution_contract_sha256=digest(contract),
        fresh_preflight_and_admission_required=True,
        reuse_requires_exact_regenerated_contract=True,
    )
    files['artifacts/gemma4_31b/provenance/original-runtime-evidence.json'] = (json.dumps(evidence, indent=2) + '\n').encode()
    receipt = dict(prepared_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
        completed_models=list(COMPLETED), completed_task_evaluations=6, saved_mmlu_pairs=104,
        original_admission_receipt_available=False,
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    files['seed-manifest.json'] = (json.dumps(receipt, indent=2) + '\n').encode()
    output = a.output_run / 'resume-seed.tar.gz'
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    inspect_seed(output, sha(output), source_sha)
    final = dict(receipt, sha256=sha(output), bytes=output.stat().st_size)
    with (a.output_run / 'seed-receipt.json').open('x') as f:
        json.dump(final, f, indent=2)
        f.write('\n')
    print(json.dumps({key: final[key] for key in ('sha256', 'bytes', 'completed_models', 'completed_task_evaluations', 'saved_mmlu_pairs', 'original_admission_receipt_available')}))


if __name__ == '__main__':
    main()
