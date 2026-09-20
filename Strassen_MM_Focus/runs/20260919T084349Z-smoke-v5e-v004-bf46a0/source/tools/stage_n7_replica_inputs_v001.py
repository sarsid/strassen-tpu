"""Verify and copy the sealed original N7 inputs into a new uploadable directory."""
import argparse
import json
from pathlib import Path
import shutil
from strassen_mm import selector_v002 as s
from strassen_mm import benchmark_n7_replica_v002 as replica


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.canonical_root.resolve()
    if not (root / '.git').is_dir():
        raise ValueError('Canonical repository required')
    evaluation = root / 'runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5'
    screen = root / 'runs/20260919T065405Z-N7-screen-v5e-v004-6ef489'
    endpoint = 'tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q'

    def outer_file(run, relative):
        file = run / relative
        entry = s.load(run / 'artifact-manifest.json')[relative]
        if entry != {'bytes': file.stat().st_size, 'sha256': s.digest(file)}:
            raise ValueError('Outer artifact seal mismatch: ' + relative)
        return file

    for run in (evaluation, screen):
        completion = s.load(outer_file(run, 'completion.json'))
        if (completion['status'] != 'completed' or completion['allocation_id'] != endpoint
                or completion['remote_may_still_be_running'] is not False):
            raise ValueError('Original execution is not complete on the expected cohort')
        outer_file(run, 'artifacts/artifact_manifest.json')
        s.verify_seal(run / 'artifacts')
    original, environment, rows, provenance = replica.verify_completed_run(evaluation, 'N7-evaluate', 128)
    choices = outer_file(screen, 'artifacts/selections.json')
    rule = outer_file(evaluation, 'artifacts/frozen_selector_input.json')
    expected = {
        choices: '5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37',
        rule: '18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77',
        original / 'results.jsonl': '757c77c09ad1016415ee4930305d6d0229af3f128ef802db9433f113cbe88153',
    }
    for path, digest in expected.items():
        if s.digest(path) != digest:
            raise ValueError('Original registered input hash mismatch: ' + str(path))
    preserved = replica.verify_original_policy(original, environment, s.load(choices), s.load(rule), choices, rule)
    source_hashes = replica.verify_reused_source(original)
    target = root / 'data/n7_replication_v001'
    target.mkdir(exist_ok=False)
    shutil.copytree(original, target / 'original_evaluation')
    shutil.copyfile(choices, target / 'selections.json')
    shutil.copyfile(rule, target / 'selector.json')
    s.verify_seal(target / 'original_evaluation')
    if s.digest(target / 'selections.json') != s.digest(choices) or s.digest(target / 'selector.json') != s.digest(rule):
        raise ValueError('Copied policy bytes differ')
    with (target / 'provenance.json').open('x') as stream:
        json.dump({'original_evaluation': provenance, 'selection_source': str(choices),
            'selector_source': str(rule), 'preserved_policy': preserved, 'source_hashes': source_hashes,
            'policy': 'Exact bytes; no refit, retuning or identity rewrite'}, stream, indent=2, sort_keys=True)
    with (target / 'copy_manifest.json').open('x') as stream:
        json.dump({str(path.relative_to(target)): {'bytes': path.stat().st_size, 'sha256': s.digest(path)}
            for path in sorted(target.rglob('*')) if path.is_file() and path.name != 'copy_manifest.json'},
            stream, indent=2, sort_keys=True)
    print(json.dumps({'copied_to': str(target), 'original_results': len(rows), 'preserved_policy': preserved}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
