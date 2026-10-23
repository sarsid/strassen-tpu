"""Export the paused architecture study, retaining canonical evidence once.

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
    shutil.copyfile(code / 'tools/verify_public_checkpoint_v001.py', out / 'verify.py')
    shutil.copyfile(code / 'tools/replay_public_checkpoint_v001.py', out / 'replay.py')
    v6 = root / 'results/v6e/168_shapes_joint_fp32_bf16_20260927_v001'
    for name in ('provenance', 'qualification', 'recovery'):
        shutil.copytree(v6 / name, out / 'results/v6e' / name)
    for name in ('README.md', 'PROTOCOL.md'):
        shutil.copyfile(v6 / name, out / 'results/v6e' / ('ORIGINAL_README.md' if name == 'README.md' else name))
    for name in ('168_shapes_fp32_20260921_v001', '168_shapes_legacy_fp32_bf16_20260927_v001'):
        shutil.copytree(root / 'results/v5e' / name, out / 'results/v5e' / name)
    supervisor = root / 'runs/20260927-arch-supervised-v001'
    for name in ('history.json', 'active-source.json', 'supervisor-state.json', 'progress.json', 'events.jsonl'):
        if (supervisor / name).exists():
            dest = out / 'provenance/supervisor' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(supervisor / name, dest)
    shutil.copytree(supervisor / 'operator-pause-20260929T205856Z', out / 'provenance/operator-pause')
    shutil.copyfile(root / 'status/work_progress_v001.jsonl', out / 'provenance/work_progress_v001.jsonl')
    phase_records = []
    receipts = sorted(report.with_name(report.name + '-input-view').glob('*-finished.json'))
    assert len(receipts) == 226
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
        original = read(run / 'artifact-manifest.json')['sha256']
        files = {str(f.relative_to(run)): f for f in (run / 'artifacts').rglob('*') if f.is_file()}
        assert files
        for name, f in files.items():
            assert name in original and sha(f) == original[name], str(f)
        files['receipt.json'] = receipt
        files['original-artifact-manifest.json'] = run / 'artifact-manifest.json'
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
    summary = read(report / 'interim_summary.json')
    assert summary['verified_shapes'] == 113 and not data['completed']
    units = sorted({r['phase'].rsplit('-', 1)[0] for r in phase_records if r['kind'] == 'comparison'})
    remaining = [f'arch-{i:03d}' for i in range(1, 169) if f'arch-{i:03d}' not in units]
    checkpoint = dict(schema_version=1, saved_utc=datetime.now(timezone.utc).isoformat(),
        state='paused_by_user', reason='Colab credits exhausted; new account/authentication deferred by user.',
        source_commit=source_commit, source_scope='Committed research source; historical modules retained without authorizing new depth-3/4 runs.',
        parent_repository='https://github.com/sarsid/strassen-tpu',
        v6e=dict(planned_shapes=168, completed_shapes=113, precision_groups=226,
                 completed_units=units, remaining_units=remaining, interrupted_unit='arch-113',
                 report='results/v6e/report/SUMMARY.md', full_study_complete=False),
        historical_v5e=dict(completed_shapes=168, output='float32', matched_to_new_protocol=False),
        new_v5e=dict(completed_shapes=0, state='not_started', policy='Original v5e kernels/tuning plus timed BF16 output; no v6e tuning transplant.'),
        automation_state='PAUSED', owned_tpu=None, resume_requires_reauthentication=True,
        omitted_duplicates=['per-phase remote copies of artifacts already retained', 'per-phase nested source archives (one frozen source archive per cohort retained)',
                            'older interim reports superseded by the 113-shape report', 'ephemeral process lock files and local Python environments'])
    write(out / 'checkpoint.json', checkpoint)
    shutil.copyfile(code / 'docs/PUBLIC_BENCHMARK_CHECKPOINT_20260929_v001.md', out / 'README.md')
    shutil.copyfile(code / 'docs/PUBLIC_BENCHMARK_RESUME_20260929_v001.md', out / 'RESUME.md')
    (out / 'results/v6e/README.md').write_text(
        '# v6e checkpoint: paused at 113/168 shapes\n\n'
        'Start with [the summary](report/SUMMARY.md), [timings and decisions](report/RESULTS.md), '
        'and [tuner design](report/TUNER_DESIGN.md). Both output contracts have 113 completed geometries.\n\n'
        'The report is an immutable export from the last completed analysis. Its dated heading '
        'saying "still running" describes report generation, not current status: the study is **paused**. '
        'See [checkpoint.json](../../checkpoint.json). `ORIGINAL_README.md` is historical.\n\n'
        '`phases/` retains canonical raw evidence with member hashes in `phase-index.json`. '
        '`report/results.json.gz` and `report/candidate_decisions.json.gz` decompress to the original JSON bytes. '
        'Recovery evidence is excluded from statistics unless its whole comparison was verified. '
        'Only one copy of each phase\'s artifacts is kept; cohort source archives remain in `provenance/`.\n')
    files = {str(f.relative_to(out)): dict(bytes=f.stat().st_size, sha256=sha(f))
             for f in sorted(out.rglob('*')) if f.is_file()}
    assert all(x['bytes'] < 100_000_000 for x in files.values()), 'A file is too large for normal GitHub storage'
    write(out / 'MANIFEST.json', dict(schema_version=1, export_complete=True, full_v6e_study_complete=False,
        source_commit=source_commit, files=files, compressed_reports=compressed))
    print(json.dumps(dict(output=str(out), files=len(files), bytes=sum(x['bytes'] for x in files.values()),
                          v6e_complete_shapes=113, phase_bundles=len(phase_records))), flush=True)


if __name__ == '__main__':
    main()
