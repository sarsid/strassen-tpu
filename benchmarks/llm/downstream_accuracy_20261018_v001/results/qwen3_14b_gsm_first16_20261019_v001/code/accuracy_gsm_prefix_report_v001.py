"""Preserve a completed GSM8K batch prefix; no full-task or equivalence claim."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from accuracy_analysis_v003 import strict_gsm8k


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoints', 'inputs', 'source-archive', 'runtime-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batches', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.batches < 4:
        raise ValueError('Only a partial prefix of one to three complete batches')
    manifest = json.loads((args.checkpoints / 'manifest.json').read_text())
    contract = json.loads((args.checkpoints / 'execution-contract.json').read_text())
    original = json.loads((args.inputs / 'manifest.json').read_text())
    contract_hash = digest(contract)
    if sha(args.source_archive) != contract['bundle_sha256']:
        raise ValueError('Changed scientific bundle')
    with tarfile.open(args.source_archive) as archive:
        for name, expected in contract['source_sha256'].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError('Changed scientific source: ' + name)
    if manifest['execution_contract_sha256'] != contract_hash or manifest['admission_status'] != 'passed':
        raise ValueError('Invalid execution contract or admission')
    if contract['batch_size'] != 8 or contract['max_new_tokens'] != 1024 or contract['do_sample']:
        raise ValueError('Changed batch or decoding settings')
    for name in ('model', 'checkpoint_revision', 'inputs_sha256', 'counts', 'examples'):
        if manifest[name] != original[name]:
            raise ValueError('Changed original input manifest: ' + name)
    if sha(args.inputs / 'inputs.jsonl') != original['inputs_sha256']:
        raise ValueError('Changed original input file')
    inputs = [json.loads(line) for line in (args.inputs / 'inputs.jsonl').read_text().splitlines()]
    for row in inputs:
        if digest({k: v for k, v in row.items() if k not in ('gold', 'input_sha256')}) != row['input_sha256']:
            raise ValueError('Changed input row')
    admission = None
    for line in args.runtime_log.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('kind') == 'runtime_admission':
            admission = event
    if not admission or admission['status'] != 'passed' or not admission['all_logits_finite']:
        raise ValueError('No passed fresh runtime admission')
    selected = [row for row in inputs if row['task'] == 'gsm8k'][:8 * args.batches]
    paired, paths = [], []
    for index in range(args.batches):
        path = args.checkpoints / 'batches' / f'gsm8k-{index:03d}.json'
        batch = json.loads(path.read_text())
        chunk = selected[index * 8:(index + 1) * 8]
        if batch['task'] != 'gsm8k' or batch['example_ids'] != [row['example_id'] for row in chunk] or batch['execution_contract_sha256'] != contract_hash:
            raise ValueError('Changed task, example order or execution contract')
        if set(batch['results']) != {'native', 'strassen'} or any(len(rows) != 8 for rows in batch['results'].values()):
            raise ValueError('Missing or unequal paired rows')
        for position, row in enumerate(chunk):
            item = {'example_id': row['example_id'], 'gold': row['gold'], 'input_sha256': row['input_sha256']}
            for arm in ('native', 'strassen'):
                result = batch['results'][arm][position]
                extracted = strict_gsm8k(result['generated_text'])
                prediction = None if result['stop_reason'] == 'max_new_tokens' else extracted
                if result['prediction'] != prediction or result['extracted_answer'] != extracted:
                    raise ValueError('Strict extraction mismatch')
                if len(result['generated_token_ids']) != result['generated_tokens'] or not 0 < result['generated_tokens'] <= 1024:
                    raise ValueError('Invalid generation length')
                item[arm] = {k: result[k] for k in ('prediction', 'extracted_answer', 'stop_reason', 'generated_tokens')}
                item[arm]['correct'] = prediction == row['gold']
            paired.append(item)
        paths.append(path)
    summary = dict(model=manifest['model'], checkpoint_revision=manifest['checkpoint_revision'],
        scope='Interim ordered prefix; not the full registered 32-example GSM8K subset',
        completed_examples=len(paired), planned_examples=32, task_complete=False,
        preservation_assessment='Deferred until the complete registered subset; no interim equivalence verdict',
        correct={arm: sum(row[arm]['correct'] for row in paired) for arm in ('native', 'strassen')},
        max_new_tokens_failures={arm: sum(row[arm]['stop_reason'] == 'max_new_tokens' for row in paired) for arm in ('native', 'strassen')},
        different_scored_predictions=sum(row['native']['prediction'] != row['strassen']['prediction'] for row in paired),
        different_extracted_answers=sum(row['native']['extracted_answer'] != row['strassen']['extracted_answer'] for row in paired),
        baseline_correct_strassen_incorrect=sum(row['native']['correct'] and not row['strassen']['correct'] for row in paired),
        baseline_incorrect_strassen_correct=sum(not row['native']['correct'] and row['strassen']['correct'] for row in paired),
        pairs=paired)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'batches').mkdir()
    for name in ('manifest.json', 'execution-contract.json'):
        shutil.copyfile(args.checkpoints / name, args.output / name)
    for path in paths:
        shutil.copyfile(path, args.output / 'batches' / path.name)
    save(args.output / 'runtime-admission.json', admission)
    save(args.output / 'summary.json', summary)
    save(args.output / 'verification.json', dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        execution_contract_sha256=contract_hash, source_bundle_sha256=sha(args.source_archive),
        original_inputs_sha256=sha(args.inputs / 'inputs.jsonl'), runtime_log_sha256=sha(args.runtime_log),
        input_and_source_hashes_verified=True, paired_rows_verified=True, fresh_runtime_admission_passed=True,
        original_full_manifest_retained=True, full_gsm8k_task_complete=False))
    save(args.output / 'artifact-manifest.json', dict(files={str(path.relative_to(args.output)): sha(path)
        for path in sorted(args.output.rglob('*')) if path.is_file()}))
    print(json.dumps({k: v for k, v in summary.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
