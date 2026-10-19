"""Verify a complete model archive and reproduce the frozen paired analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from accuracy_analysis_v003 import analyze, read_rows, strict_gsm8k
from accuracy_checkpoint_report_v001 import digest, save, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-archive', 'archive-receipt', 'source-archive', 'inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.archive_receipt.read_text())
    if sha(args.model_archive) != receipt['sha256'] or args.model_archive.stat().st_size != receipt['bytes']:
        raise ValueError('Model archive differs from remote receipt')
    if args.output.exists():
        raise ValueError('Output must be a new immutable directory')
    with tarfile.open(args.model_archive) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        roots = {PurePosixPath(name).parts[0] for name in names if name}
        if len(roots) != 1 or len(set(names)) != len(names):
            raise ValueError('Ambiguous archive root or duplicate member')
        if sum(member.size for member in members) > 128 * 1024**2:
            raise ValueError('Unexpectedly large results archive')
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member')
            if member.isfile() and path.suffix not in ('.json', '.jsonl'):
                raise ValueError('Unexpected non-result file')
        args.output.mkdir(parents=True)
        raw = args.output / 'raw'
        raw.mkdir()
        archive.extractall(raw, filter='data')
    model = raw / next(iter(roots))
    admissions = list(model.glob('admission-*.json'))
    if len(admissions) != 1:
        raise ValueError('Expected exactly one admission for this archived execution')
    admission = json.loads(admissions[0].read_text())
    manifest, contract, collected, counts, batches = verify(model, args.inputs, args.source_archive, admission)
    if counts != {'mmlu': 114, 'gsm8k': 32, 'ppl': 32} or counts != manifest['counts']:
        raise ValueError('Model is missing a complete registered task')
    if sha(Path(__file__).with_name('accuracy_analysis_v003.py')) != contract['source_sha256']['tools/accuracy_analysis_v003.py']:
        raise ValueError('Reporting analysis differs from frozen scientific analysis')
    paired = {}
    for arm in ('native', 'strassen'):
        actual = read_rows(model / (arm + '.jsonl'))
        reconstructed = {(row['task'], row['example_id']): row for row in collected[arm]}
        if actual != reconstructed:
            raise ValueError('Aggregated rows differ from saved paired batches')
        paired[arm] = actual
    report = analyze(manifest, model / 'native.jsonl', model / 'strassen.jsonl')
    original_report = json.loads((model / 'summary.json').read_text())
    for key, value in report.items():
        if original_report.get(key) != value:
            raise ValueError('Reproduced analysis differs from remote summary: ' + key)
    gsm_pairs = []
    for spec in manifest['examples']:
        if spec['task'] != 'gsm8k':
            continue
        key = ('gsm8k', spec['example_id'])
        item = {'example_id': spec['example_id'], 'gold': spec['gold']}
        for arm in ('native', 'strassen'):
            row = paired[arm][key]
            if row['extracted_answer'] != strict_gsm8k(row['generated_text']):
                raise ValueError('Stored extracted answer differs from frozen extraction')
            if len(row['generated_token_ids']) != row['generated_tokens'] or not 0 < row['generated_tokens'] <= contract['max_new_tokens']:
                raise ValueError('Invalid generation length')
            item[arm] = {k: row[k] for k in ('prediction', 'extracted_answer', 'stop_reason', 'generated_tokens')}
            item[arm]['correct'] = row['prediction'] == spec['gold']
        gsm_pairs.append(item)
    detail = dict(scope='Descriptive stopping/extraction audit; registered scores and criteria are unchanged',
        n=len(gsm_pairs),
        different_extracted_answers=sum(row['native']['extracted_answer'] != row['strassen']['extracted_answer'] for row in gsm_pairs),
        max_new_tokens_failures={arm: sum(row[arm]['stop_reason'] == 'max_new_tokens' for row in gsm_pairs) for arm in ('native', 'strassen')},
        pairs=gsm_pairs)
    shutil.copyfile(args.model_archive, args.output / 'model.tar.gz')
    shutil.copyfile(args.archive_receipt, args.output / 'archive-receipt.json')
    save(args.output / 'summary.json', report)
    save(args.output / 'gsm8k-stopping-audit.json', detail)
    save(args.output / 'verification.json', dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        model=manifest['model'], checkpoint_revision=manifest['checkpoint_revision'], model_complete=True,
        completed_tasks=counts, paired_rows_per_arm=sum(counts.values()), batch_files=len(batches),
        model_archive_sha256=sha(args.model_archive), source_bundle_sha256=sha(args.source_archive),
        execution_contract_sha256=digest(contract), original_inputs_sha256=sha(args.inputs / 'inputs.jsonl'),
        runtime_admission_passed=True, input_and_source_hashes_verified=True,
        aggregated_rows_match_batches=True, frozen_analysis_matches_remote_summary=True))
    save(args.output / 'artifact-manifest.json', dict(files={str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(model=manifest['model'], tasks=report['tasks'],
        gsm8k_stopping_audit={k: v for k, v in detail.items() if k != 'pairs'}), indent=2))


if __name__ == '__main__':
    main()
