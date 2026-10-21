"""Preserve both complete Qwen models and the all saved Gemma batches and their original provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from resume_accuracy_gemma_remote_v002 import COMPLETED, digest, inspect_seed, sha


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
    # Recover the old evidence gap verbatim; the new admission belongs only to v011.
    with tarfile.open(a.prior_run / 'resume-seed.tar.gz') as prior_seed:
        name = 'artifacts/gemma4_31b/provenance/original-runtime-evidence.json'
        files[name] = prior_seed.extractfile(name).read()
        old_batches = {m.name.rsplit('/', 1)[1]: hashlib.sha256(prior_seed.extractfile(m).read()).hexdigest()
                       for m in prior_seed.getmembers() if '/batches/' in m.name}
    batches = {}
    for path in sorted((checkpoints / 'batches').glob('*.json')):
        verified = a.prior_run / ('verified-gemma-gsm-first16-v001' if path.name.startswith('gsm8k')
                                   else 'verified-gemma-mmlu-ppl-v001') / 'batches' / path.name
        if sha(path) != sha(verified):
            raise ValueError('Checkpoint differs from independently verified results')
        inherited = path.name in old_batches
        if inherited and sha(path) != old_batches[path.name]:
            raise ValueError('Original inherited batch changed')
        batches[path.name] = dict(sha256=sha(path), execution_run='20261020-accuracy-study-v010'
                                  if inherited else a.prior_run.name)
    admission = checkpoints / 'provenance/admission-20261021T065510Z-20a85c.json'
    lineage = dict(prior_run=a.prior_run.name, batches=batches,
        v011_admission=dict(file=admission.name, sha256=sha(admission)),
        v011_preflight_sha256=sha(checkpoints / 'provenance/query-block-preflight-0.json'),
        original_v010_admission_receipt_available=False,
        qualification='Fresh v011 receipts apply only to new v011 batches; original v010 receipt gap remains disclosed',
        recovery_requires_fresh_preflight_admission_and_contract=True)
    files['artifacts/gemma4_31b/provenance/resume-lineage-v002.json'] = (json.dumps(lineage, indent=2) + '\n').encode()
    receipt = dict(prepared_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
        completed_models=list(COMPLETED), completed_task_evaluations=8,
        saved_pairs={'mmlu':114, 'ppl':32, 'gsm8k':16}, original_admission_receipt_available=False,
        v011_admission_receipt_available=True,
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
    print(json.dumps({key: final[key] for key in ('sha256', 'bytes', 'completed_models', 'completed_task_evaluations', 'saved_pairs', 'original_admission_receipt_available')}))


if __name__ == '__main__':
    main()
