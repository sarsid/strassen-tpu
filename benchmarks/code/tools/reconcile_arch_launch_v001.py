"""Verify a completed remote archive after a lost launch acknowledgment.

Recovery is an additional observation. The failed controller archive is never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from run_region_cohort_v001 import read, sha, write, utc


def verify(folder):
    proof = read(folder / 'reconciliation.json')
    package = folder / 'remote-run.tar.gz'
    if sha(package) != proof['archive_sha256']:
        raise ValueError('Recovered package changed')
    run = folder / 'remote' / proof['run_id']
    manifest = read(run / 'run-manifest.json')
    # Compare the extracted manifest itself to the checksum-protected package.
    with tarfile.open(package) as packed:
        if packed.extractfile(proof['run_id'] + '/run-manifest.json').read() != (run / 'run-manifest.json').read_bytes():
            raise ValueError('Recovered manifest changed')
    for relative, expected in manifest.items():
        file = (run / relative).resolve()
        if not file.is_relative_to(run.resolve()) or sha(file) != expected['sha256']:
            raise ValueError('Recovered artifact mismatch: ' + relative)
    done = read(run / 'completion.json')
    launch = read(run / 'launch.json')
    for key in ('run_id', 'phase', 'allocation_id'):
        if done[key] != proof[key] or launch[key] != proof[key]:
            raise ValueError('Recovered execution identity mismatch')
    if done['status'] != 'completed' or done['process_exit_code'] != 0:
        raise ValueError('Recovered worker did not complete')
    if launch['archive_sha256'] != proof['source_archive_sha256'] or sha(run / 'input-source.tar.gz') != proof['source_archive_sha256']:
        raise ValueError('Recovered source differs from failed launch')
    summary = read(run / 'artifacts/summary.json')
    if not summary['completed']:
        raise ValueError('Incomplete recovered measurements')
    planned = read(run / 'artifacts/planned_cases.json')
    rows = [json.loads(line) for line in (run / 'artifacts/results.jsonl').read_text().splitlines()]
    cases = [r for r in rows if r.get('event') == 'case_result']
    expected = sum(len(g['inputs']) * len(g['arms']) for g in planned)
    keys = {(r['group_id'], r['seed'], r['arm_id']) for r in cases}
    if len(keys) != expected or len(cases) != expected:
        raise ValueError('Recovered terminal case count mismatch')
    return run, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--prior-cohort', type=Path, required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--archive-sha256', required=True)
    a = p.parse_args()
    old = read(a.prior_cohort / (a.phase + '-finished.json'))
    failed = Path(old['run'])
    launch = read(failed / 'execution.json')
    if sha(a.folder / 'remote-run.tar.gz') != a.archive_sha256:
        raise ValueError('Download differs from remote archive-ready checksum')
    target = a.folder / 'remote'
    target.mkdir(exist_ok=False)
    with tarfile.open(a.folder / 'remote-run.tar.gz') as packed:
        for member in packed.getmembers():
            if Path(member.name).parts[0] != old['run_id']:
                raise ValueError('Wrong run in recovery package')
        packed.extractall(target, filter='data')
    write(a.folder / 'reconciliation.json', dict(
        observed_utc=utc(), run_id=old['run_id'], phase=a.phase,
        allocation_id=launch['allocation_id'], source_archive_sha256=launch['source_archive_sha256'],
        archive_sha256=a.archive_sha256, original_failed_run=str(failed),
        original_completion_sha256=sha(failed / 'completion.json'),
        cause='Colab connection lost during launch acknowledgment; remote worker completed successfully.',
        underlying_disconnect_cause='Not established by available logs.',
        remote_execution_uncertain=False, no_measurements_reexecuted=True))
    run, summary = verify(a.folder)
    write(a.folder / 'verified.json', dict(run=str(run), summary=summary,
        original_failure_preserved=True, verified_utc=utc()))
    print(json.dumps(dict(recovered=True, run=str(run), cases=summary['case_status_counts'])))


if __name__ == '__main__':
    main()
