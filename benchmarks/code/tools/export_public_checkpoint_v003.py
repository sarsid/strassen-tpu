"""Export the completed architecture study, retaining canonical evidence once.

The full local history is untouched. No credentials, allocation or TPU APIs are
used. Source is read from a committed Git tree; verified raw artifacts remain
byte-identical, with every member covered by a portable checksum index.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from report_arch_interim_v001 import summarize


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def bundle(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = {name: sha(f) for name, f in files.items()}
    with path.open('xb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as archive:
                for name, f in sorted(files.items()):
                    assert not f.is_symlink()
                    info = archive.gettarinfo(str(f), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with f.open('rb') as source:
                        archive.addfile(info, source)
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--input-view', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, report, out = a.root.resolve(), a.report.resolve(), a.output.resolve()
    assert out.name == 'benchmarks' and not out.exists()
    out.mkdir()
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', source_commit], cwd=root, text=True).splitlines()
    code_roots = {'src', 'tools', 'configs', 'runtime', 'docs', 'plans', 'tests'}
    root_names = {'AGENTS.md', 'README.md', 'TODO.md', 'strassen_optimized.py',
                  'START_HERE_v001.md', 'START_HERE_v002.md', 'START_HERE_v003.md'}
    code_paths = [s for s in paths if s.split('/')[0] in code_roots or s in root_names
                  or (s.startswith('status/') and s.endswith(('.py', '.html', '.md')))]
    code = out / 'code'
    code.mkdir()
    proc = subprocess.Popen(['git', 'archive', source_commit, '--', *code_paths], cwd=root, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
        for member in archive:
            if not member.isfile():
                continue
            dest = code / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    assert proc.wait() == 0
    shutil.copyfile(code / 'tools/verify_public_checkpoint_v002.py', out / 'verify.py')
    shutil.copyfile(code / 'tools/replay_public_checkpoint_v002.py', out / 'replay.py')
    v6 = root / 'results/v6e/168_shapes_joint_fp32_bf16_20260927_v001'
    complete = read(v6 / 'COMPLETE.json')
    assert complete['completed'] and len(set(complete['completed_units'])) == 168
    assert sha(report / 'results.json') == complete['results_sha256']
    for name in ('provenance', 'qualification', 'recovery'):
        shutil.copytree(v6 / name, out / 'results/v6e' / name)
    for name in ('README.md', 'PROTOCOL.md'):
        shutil.copyfile(v6 / name, out / 'results/v6e' / ('ORIGINAL_README.md' if name == 'README.md' else name))
    shutil.copyfile(v6 / 'COMPLETE.json', out / 'results/v6e/COMPLETE.json')
    for name in ('168_shapes_fp32_20260921_v001', '168_shapes_legacy_fp32_bf16_20260927_v001'):
        shutil.copytree(root / 'results/v5e' / name, out / 'results/v5e' / name)
    supervisor = root / 'runs/20260927-arch-supervised-v001'
    for name in ('history.json', 'active-source.json', 'supervisor-state.json', 'progress.json', 'events.jsonl', 'completion.json'):
        if (supervisor / name).exists():
            dest = out / 'provenance/supervisor' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(supervisor / name, dest)
    shutil.copytree(supervisor / 'operator-pause-20260929T205856Z', out / 'provenance/operator-pause')
    shutil.copytree(supervisor / 'final-verification-20261002-v001', out / 'provenance/final-verification')
    shutil.copytree(supervisor / 'operations/20261002T141705-release-e20ccf', out / 'provenance/final-release')
    shutil.copyfile(root / 'status/work_progress_v001.jsonl', out / 'provenance/work_progress_v001.jsonl')
    phase_records = []
    receipts = sorted(a.input_view.glob('*-finished.json'))
    assert len(receipts) == 336
    cohorts = {Path(read(p)['provenance']['cohort']) for p in receipts}
    for cohort in sorted(cohorts):
        receipts.extend(sorted(cohort.glob('arch-smoke-finished.json')))
    for receipt in receipts:
        r = read(receipt)
        assert r['status'] == 'completed'
        run = Path(r['run'])
        assert sha(run / 'completion.json') == r['completion_sha256']
        phase = receipt.name.removesuffix('-finished.json')
        kind = 'qualification' if phase == 'arch-smoke' else 'comparison'
        cohort = Path(r.get('provenance', {}).get('cohort', receipt.parent))
        manifest_path = run / 'artifact-manifest.json'
        if manifest_path.exists():
            original = read(manifest_path)['sha256']
        else:
            # The lost-acknowledgment recovery uses the remote worker's manifest,
            # verified against the checksum-protected recovered archive.
            proof_path = Path(r['provenance']['evidence'])
            proof = read(proof_path)
            recovered = proof_path.parent / 'remote-run.tar.gz'
            assert sha(recovered) == proof['archive_sha256']
            manifest_path = run / 'run-manifest.json'
            with tarfile.open(recovered) as archive:
                assert archive.extractfile(proof['run_id'] + '/run-manifest.json').read() == manifest_path.read_bytes()
            original = {name: value['sha256'] for name, value in read(manifest_path).items()}

        files = {str(f.relative_to(run)): f for f in (run / 'artifacts').rglob('*') if f.is_file()}
        assert files
        for name, f in files.items():
            assert name in original and sha(f) == original[name], str(f)
        files['receipt.json'] = receipt
        files['original-artifact-manifest.json'] = manifest_path
        for name in ('completion.json', 'execution.json', 'benchmark.log', 'worker.log', 'remote-launch.json', 'remote-archive.json'):
            if (run / name).exists():
                files[name] = run / name
        name = (cohort.name + '-' if kind == 'qualification' else '') + phase
        path = out / 'results/v6e/phases' / (name + '.tar.gz')
        hashes = bundle(path, files)
        phase_records.append(dict(phase=phase, kind=kind, cohort=cohort.name,
            original_run=str(run.relative_to(root)), original_receipt=str(receipt),
            bundle=str(path.relative_to(out)), bundle_sha256=sha(path), member_sha256=hashes))
        print(json.dumps(dict(exported=phase, kind=kind, bytes=path.stat().st_size)), flush=True)
    write(out / 'results/v6e/phase-index.json', dict(phases=phase_records,
        rule='Canonical artifacts retained once per phase; original byte hashes verified. Whole screen/confirm pairs only; retries never pooled.'))
    compressed = []
    report_out = out / 'results/v6e/report'
    report_out.mkdir()
    for f in sorted(report.iterdir()):
        if f.name in ('candidate_decisions.json', 'results.json'):
            dest = report_out / (f.name + '.gz')
            with f.open('rb') as src, dest.open('xb') as stream:
                with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as z:
                    shutil.copyfileobj(src, z)
            compressed.append(dict(file=str(dest.relative_to(out)), uncompressed_sha256=sha(f), uncompressed_bytes=f.stat().st_size))
        else:
            shutil.copyfile(f, report_out / f.name)
    data = read(report / 'results.json')
    assert data['completed'] and len(data['results']) == 336 and not data['missing']
    summary = dict(completed=True, verified_shapes=168, planned_shapes=168,
                   by_output=summarize(data['results']), case_status_counts=data['case_status_counts'])
    write(report_out / 'summary.json', summary)
    lines = ['# Final v6e results: 168 shapes', '',
        'All 168 geometries are complete in FP32 and BF16 output. Measurements, tuning choices and numerical errors are retained separately for each output precision.', '',
        'Wins and losses require the paired pointwise 95% interval to lie entirely above or below 1. Intervals crossing 1 are unresolved. There is no multiple-comparison adjustment; these Gaussian inputs and shapes are development data.', '',
        '| Output | Method | Wins vs tuned Native | Losses | Unresolved |',
        '|---|---|---:|---:|---:|']
    for dtype, group in summary['by_output'].items():
        for role in ('cubic', 's1', 's2'):
            counts = group['methods'][role]['vs_tuned_native']
            lines.append('| ' + ' | '.join([dtype, role, *(str(counts.get(k, 0)) for k in ('win', 'loss', 'inconclusive'))]) + ' |')
    lines += ['', '## Numerical error', '',
        'Each entry summarizes the worst of three fresh seeds per shape, relative to FP64 on exact BF16 operands. The reference uses the full output or 128 × 128 sampled entries over all K; finiteness is checked over the full output. Numerical eligibility does not establish LLM prediction quality.', '',
        '| Output | Method | Median relative L2 % | Maximum relative L2 % |',
        '|---|---|---:|---:|']
    for dtype, group in summary['by_output'].items():
        for role in ('native', 'cubic', 's1', 's2'):
            m = group['methods'][role]
            lines.append('| ' + ' | '.join([dtype, role, *(f'{100*m[k]:.6f}' for k in ('relative_l2_median', 'relative_l2_max'))]) + ' |')
    lines += ['', '[Full timings and decisions](RESULTS.md) · [Tuner design](TUNER_DESIGN.md).', '']
    (report_out / 'SUMMARY.md').write_text('\n'.join(lines))
    units = sorted({r['phase'].rsplit('-', 1)[0] for r in phase_records if r['kind'] == 'comparison'})
    remaining = [f'arch-{i:03d}' for i in range(1, 169) if f'arch-{i:03d}' not in units]
    checkpoint = dict(schema_version=1, saved_utc=datetime.now(timezone.utc).isoformat(),
        state='completed', reason='All 168 v6e shapes verified and committed; owned TPU released and recovery monitor deleted.',
        source_commit=source_commit, source_scope='Committed research source; historical modules retained without authorizing new depth-3/4 runs.',
        parent_repository='https://github.com/sarsid/strassen-tpu',
        v6e=dict(planned_shapes=168, completed_shapes=168, precision_groups=336,
                 completed_units=units, remaining_units=remaining, interrupted_unit=None,
                 report='results/v6e/report/SUMMARY.md', full_study_complete=True),
        historical_v5e=dict(completed_shapes=168, output='float32', matched_to_new_protocol=False),
        new_v5e=dict(completed_shapes=0, state='deferred_not_queued', policy='Original v5e kernels/tuning plus timed BF16 output; no v6e tuning transplant.'),
        automation_state='DELETED', owned_tpu=None, resume_requires_reauthentication=False, automatic_followups=[],
        omitted_duplicates=['per-phase remote copies of artifacts already retained', 'per-phase nested source archives (one frozen source archive per cohort retained)',
                            'older interim reports superseded by the final 168-shape report', 'ephemeral process lock files and local Python environments'])
    write(out / 'checkpoint.json', checkpoint)
    shutil.copyfile(code / 'docs/PUBLIC_BENCHMARK_FINAL_20261002_v001.md', out / 'README.md')
    (out / 'RESUME.md').write_text('# Experiment complete\n\nAll 168 v6e shapes are saved. The TPU was released, the supervisor finished, and the recovery monitor was deleted. Nothing remains to resume. LLM fusion or new v5e measurements require fresh instructions.\n')
    (out / 'results/v6e/README.md').write_text(
        '# v6e final results: 168/168 shapes\n\n'
        'Start with [the summary](report/SUMMARY.md), [timings and decisions](report/RESULTS.md), '
        'and [tuner design](report/TUNER_DESIGN.md). Both output contracts have 168 completed geometries.\n\n'
        'The final report is complete, the TPU was released, and no experiments are queued. '
        'See [checkpoint.json](../../checkpoint.json). `ORIGINAL_README.md` is historical.\n\n'
        '`phases/` retains canonical raw evidence with member hashes in `phase-index.json`. '
        '`report/results.json.gz` and `report/candidate_decisions.json.gz` decompress to the original JSON bytes. '
        'Recovery evidence is excluded from statistics unless its whole comparison was verified. '
        'Only one copy of each phase\'s artifacts is kept; cohort source archives remain in `provenance/`.\n')
    files = {str(f.relative_to(out)): dict(bytes=f.stat().st_size, sha256=sha(f))
             for f in sorted(out.rglob('*')) if f.is_file()}
    assert all(x['bytes'] < 100_000_000 for x in files.values()), 'A file is too large for normal GitHub storage'
    write(out / 'MANIFEST.json', dict(schema_version=1, export_complete=True, full_v6e_study_complete=True,
        source_commit=source_commit, files=files, compressed_reports=compressed))
    print(json.dumps(dict(output=str(out), files=len(files), bytes=sum(x['bytes'] for x in files.values()),
                          v6e_complete_shapes=168, phase_bundles=len(phase_records))), flush=True)


if __name__ == '__main__':
    main()
