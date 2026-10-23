"""Join independently verified whole comparisons across runtime attempts."""
import argparse
import json
from pathlib import Path
import shutil
from arch_resilience_v001 import BASE, collect, units
from run_region_cohort_v001 import read, write, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--hardware', choices=['v5e', 'v6e'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    chosen = collect(read(a.history), a.hardware)
    if set(chosen) != set(units(a.hardware)):
        raise ValueError('Study coverage is incomplete')
    view = a.output_dir.parent / (a.output_dir.name + '-input-view')
    config = view / 'source/configs'
    config.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASE / 'source/configs/arch_v6e_168_v001', config / 'arch_v6e_168_v001')
    proofs = {}
    for pair in chosen.values():
        for item in pair.values():
            run = Path(item['run'])
            write(view / (item['phase'] + '-finished.json'), dict(status='completed', run=str(run),
                completion_sha256=sha(run / 'completion.json'), provenance=item))
            proofs[item['phase']] = item
    if a.hardware == 'v6e':
        from report_arch_study_v002 import build
        code = build(view, a.output_dir)
        if code:
            raise ValueError('Final v6e report is incomplete')
    else:
        import report_mlsys_multiallocation_v001 as report
        # Every referenced phase was verified above; the view intentionally
        # references immutable runs in several original cohort directories.
        def checked(_cohort, receipt):
            item = proofs[receipt.name.removesuffix('-finished.json')]
            run = Path(item['run'])
            return run, read(run / 'artifacts/summary.json'), 'completed'
        report.checked_phase = checked
        for suffix, dtype in [('fp32', 'float32'), ('bf16', 'bfloat16')]:
            report.SCOPES['arch_v5e_168_' + suffix + '_v001'] = 'Historical v5e, ' + dtype + ' output'
        result = report.report(view, a.output_dir)
        result['completed'] = len(result['coverage']) == 2 and all(
            c['confirmed_shapes'] == c['expected_shapes'] == 168 for c in result['coverage'].values()) and result['confirmed_method_rows'] == 1680
        if not result['completed']:
            raise ValueError('Final v5e report is incomplete')
        write(a.output_dir / 'results.json', result)
    write(a.output_dir / 'recovery_provenance.json', dict(units=chosen,
        rule='Whole comparisons only; each screen/confirm pair shares one device identity. Never pool retry samples.'))


if __name__ == '__main__':
    main()
