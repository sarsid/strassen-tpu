"""Audit and export only the historical 168-shape v5e matrix study.

No TPU calls or measurements. Original phases remain untouched. Compressed
phase bundles contain byte-identical artifacts; the summary is derived.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

METHODS = ('native_default', 'native', 'cubic', 'one_level', 'two_level')
METRICS = ('relative_l2', 'max_abs_error', 'rmse', 'mean_abs_error',
           'p50_abs_error', 'p99_abs_error', 'normwise_error')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('STRASSEN_PROJECT_ROOT', Path(__file__).resolve().parents[1])))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    if not out.is_relative_to(root / 'results/v5e'):
        raise ValueError('Use a dedicated results/v5e directory')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'phases').mkdir()
    (out / 'provenance').mkdir()
    cohort = root / 'runs/20260921-mlsys-main-v5e-v001'
    frozen = read(cohort / 'frozen.json')
    assert sha(cohort / 'source.tar') == frozen['archive_sha256']
    for relative, expected in frozen['source_sha256'].items():
        assert sha(cohort / 'source' / relative) == expected, relative
    auditor_path = cohort / 'source/tools/audit_mlsys_shapes_v001.py'
    spec = importlib.util.spec_from_file_location('frozen_v5e_auditor', auditor_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    for name in ('frozen.json', 'cohort.json', 'plan.json'):
        shutil.copyfile(cohort / name, out / 'provenance' / name)
    with (cohort / 'source.tar').open('rb') as source, (out / 'provenance/source.tar.gz').open('xb') as dest:
        with gzip.GzipFile(filename='', fileobj=dest, mode='wb', mtime=0) as z:
            shutil.copyfileobj(source, z)
    cfgdir = cohort / 'source/configs/mlsys_shapes_v001'
    shutil.copytree(cfgdir, out / 'config')
    shutil.copyfile(cohort / 'source/plans/mlsys_shapes_v001/PROTOCOL.md', out / 'PROTOCOL.md')
    shutil.copyfile(Path(__file__), out / 'provenance/export_v5e_results_v001.py')
    manifest_shapes = read(cfgdir / 'shapes.json')['shapes']
    phases = ['MAIN-smoke'] + [f'MAIN-{i:02d}-{stage}' for i in range(1, 13) for stage in ('screen', 'confirm')]
    rows = {}
    records = []
    identities = {}
    totals = {stage: Counter() for stage in ('smoke', 'screen', 'confirm')}
    for phase in phases:
        receipt_path = cohort / (phase + '-finished.json')
        receipt = read(receipt_path)
        run = Path(receipt['run'])
        assert run.resolve().is_relative_to(cohort / 'phases')
        assert receipt['status'] == 'completed'
        assert sha(run / 'completion.json') == receipt['completion_sha256']
        completion = read(run / 'completion.json')
        assert completion['status'] == 'completed' and completion['remote_may_still_be_running'] is False
        original_manifest = read(run / 'artifact-manifest.json')['sha256']
        artifact_files = sorted(f for f in (run / 'artifacts').rglob('*') if f.is_file())
        for f in artifact_files:
            assert not f.is_symlink()
            assert sha(f) == original_manifest[str(f.relative_to(run))], str(f)
        audit = auditor.audit(run / 'artifacts')
        assert audit['passed']
        if audit['stage'] == 'confirm':
            assert audit['interval_replay'] == 'passed'
        write(out / 'audits' / (phase + '.json'), audit)
        totals[audit['stage']].update(audit['status_counts'])
        artifact = run / 'artifacts'
        environment = read(artifact / 'environment.json')
        identity = environment['identity']
        assert environment['qualified_single_v5e']
        identity_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identities[identity_key] = identity
        bundle = out / 'phases' / (phase + '.tar.gz')
        members = {str(f.relative_to(run)): f for f in artifact_files}
        members.update({'completion.json': run / 'completion.json',
                        'original-artifact-manifest.json': run / 'artifact-manifest.json',
                        'execution.json': run / 'execution.json',
                        'receipt.json': receipt_path})
        if (run / 'benchmark.log').exists():
            members['benchmark.log'] = run / 'benchmark.log'
        member_hashes = {name: sha(f) for name, f in members.items()}
        with bundle.open('xb') as stream:
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w') as archive:
                    for name, f in sorted(members.items()):
                        info = archive.gettarinfo(str(f), arcname=name)
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        with f.open('rb') as data:
                            archive.addfile(info, data)
        with tarfile.open(bundle) as archive:
            assert set(archive.getnames()) == set(members)
            for member in archive:
                assert member.isfile()
                with archive.extractfile(member) as data:
                    digest = hashlib.file_digest(data, 'sha256').hexdigest()
                assert digest == member_hashes[member.name], member.name
        records.append(dict(phase=phase, original_run=str(run.relative_to(root)),
                            bundle=str(bundle.relative_to(out)), bundle_sha256=sha(bundle),
                            member_sha256=member_hashes, original_receipt_sha256=sha(receipt_path),
                            audit=str((out / 'audits' / (phase + '.json')).relative_to(out))))
        if audit['stage'] == 'confirm':
            samples, cases = defaultdict(dict), defaultdict(list)
            with (artifact / 'results.jsonl').open() as f:
                for line in f:
                    event = json.loads(line)
                    if event['event'] not in ('sample', 'case_result'):
                        continue
                    key = event['group_id'], event['arm_id']
                    if event['event'] == 'sample':
                        pair = event['seed'], event['round']
                        assert pair not in samples[key]
                        samples[key][pair] = event['elapsed_ms']
                    else:
                        cases[key].append(event)
            stats = read(artifact / 'confirmation_statistics.json')['by_shape']
            for group in read(artifact / 'planned_cases.json'):
                sid = group['shape']['id']
                assert sid not in rows
                arms = {a['arm_id']: a for a in group['arms']}
                methods = {}
                for method in METHODS:
                    info = stats[sid]['headline'][method]
                    arm = group['headline_candidates'][method]
                    assert arm == info['candidate_id']
                    key = group['group_id'], arm
                    measured = list(samples[key].values())
                    observed = cases[key]
                    assert len(measured) == 90 and len(observed) == 3
                    comparison = info['comparison_vs_tuned_native']
                    assert math.isclose(statistics.mean(measured), comparison['candidate_mean_ms'], rel_tol=1e-12)
                    metadata = observed[0]['kernel_metadata']
                    assert metadata['input_dtype'] == 'bfloat16' and metadata['output_dtype'] == 'float32'
                    methods[method] = dict(configuration=arms[arm], kernel_metadata=metadata,
                        mean_ms=statistics.mean(measured), sample_count=len(measured),
                        fresh_seeds=sorted({r['seed'] for r in observed}),
                        eligible=info['valid_numerical_comparison_vs_tuned_native'],
                        speedup_vs_tuned_native=1 / comparison['candidate_over_reference_latency_ratio'],
                        speedup_ci95=[1 / comparison['ci95'][1], 1 / comparison['ci95'][0]],
                        original_latency_comparison=comparison,
                        errors_worst_across_inputs={name: max(r['correctness'][name] for r in observed) for name in METRICS},
                        reference_scopes=sorted({r['correctness']['reference_scope'] for r in observed}),
                        case_statuses=[r['status'] for r in observed])
                rows[sid] = dict(shape=group['shape'], methods=methods, identity_key=identity_key,
                                 evidence_bundle=str(bundle.relative_to(out)), group_id=group['group_id'])
        print(json.dumps(dict(phase=phase, passed=True, checks=audit['checks'], outcomes=audit['outcomes'])), flush=True)
    assert len(manifest_shapes) == len(rows) == 168
    assert set(rows) == {s['id'] for s in manifest_shapes}
    assert len({(s['m'], s['k'], s['n']) for s in manifest_shapes}) == 168
    assert len(identities) == 1
    ordered = [rows[s['id']] for s in manifest_shapes]
    aggregate = {}
    for method in METHODS:
        counts = Counter()
        for row in ordered:
            m = row['methods'][method]
            lo, hi = m['speedup_ci95']
            counts['ineligible' if not m['eligible'] else 'win' if lo > 1 else 'loss' if hi < 1 else 'inconclusive'] += 1
        aggregate[method] = dict(counts)
    result = dict(schema_version=1, study='historical_v5e_168_fp32_v001', completed=True,
        source_cohort=str(cohort.relative_to(root)), original_source_commit=frozen['baseline_commit'],
        shapes=168, output_dtype='float32', input_dtype='bfloat16', accumulation_dtype='float32',
        timing_scope='Complete resident call including padding/cropping; excludes compilation and host transfer.',
        case_status_counts={k: dict(v) for k, v in totals.items()}, comparison_counts_vs_tuned_native=aggregate,
        identities=identities, results=ordered,
        limitations=['Historical implementation and search; not the new joint tuner.',
                     'No BF16-output arm in this study.',
                     'Development Gaussian shapes, not unseen-shape or LLM-quality validation.',
                     'Sampled error maxima do not bound unsampled entries or arbitrary input values.',
                     'Original campaign-wide controller state is not used as proof of MAIN completion.',
                     'LLM phases, other v5e studies and v6e results are excluded.'])
    write(out / 'summary.json', result)
    write(out / 'provenance/phase-index.json', dict(phases=records, source_archive_uncompressed_sha256=frozen['archive_sha256']))
    lines = ['# Historical v5e: 168-shape matrix study', '',
        'This is the completed September 21 study, exported without new TPU measurements. '
        'All five families use BF16 inputs and FP32 accumulation/output. Original screen selections are retained; '
        'confirmation does not reselect a winner. This archive contains no v6e or LLM measurements.', '',
        'All 25 main-study phase audits passed, including deterministic confidence-interval replay for all 12 confirmation phases. '
        'Every copied bundle member was checked against its original bytes. Numerical references are not rerun by this archive audit.', '',
        '## Coverage', '', f'Outcome counts: `{result["case_status_counts"]}`.', '',
        '| Family | Wins vs tuned Native | Inconclusive | Losses | Ineligible |',
        '|---|---:|---:|---:|---:|']
    for method, c in aggregate.items():
        lines.append(f'| {method} | {c.get("win",0)} | {c.get("inconclusive",0)} | {c.get("loss",0)} | {c.get("ineligible",0)} |')
    lines += ['', 'Wins use pointwise paired 95% intervals, conditional on screening; no multiplicity adjustment.', '',
        '## Confirmed times by shape', '',
        '| M × K × N | Default Native ms | Tuned Native ms | Cubic ms | S1 ms | S2 ms | S1 speedup | S2 speedup |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in ordered:
        methods = row['methods'];s = row['shape']
        fields = [' × '.join(str(s[k]) for k in ('m','k','n'))]
        fields += [f'{methods[k]["mean_ms"]:.6f}' for k in METHODS]
        fields += [f'{methods[k]["speedup_vs_tuned_native"]:.4f}×' + ('' if methods[k]['eligible'] else ' ineligible') for k in ('one_level','two_level')]
        lines.append('| ' + ' | '.join(fields) + ' |')
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    files = {str(f.relative_to(out)): dict(sha256=sha(f), bytes=f.stat().st_size) for f in sorted(out.rglob('*')) if f.is_file()}
    write(out / 'EXPORT_MANIFEST.json', dict(completed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        exporter_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files=files, member_hashes='provenance/phase-index.json', phase_audits=25, confirmation_interval_replays=12,
        note='This manifest covers machine-exported evidence. Human README and gap analysis are committed alongside it.'))
    print(json.dumps(dict(completed=True, output=str(out), shapes=168, files=len(files),
                          bytes=sum(v['bytes'] for v in files.values()), case_status_counts=result['case_status_counts'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
