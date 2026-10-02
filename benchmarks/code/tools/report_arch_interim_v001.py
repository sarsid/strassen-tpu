"""Read-only, dated interim summary of verified whole v6e comparisons."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

from arch_resilience_v001 import BASE, collect
from report_arch_study_v002 import build
from report_v6e_suite_v001 import classify
from run_region_cohort_v001 import read, write, sha


ROLES = ('native_default', 'native', 'cubic', 's1', 's2')


def summarize(rows):
    result = {}
    for dtype in ('float32', 'bfloat16'):
        group = [r for r in rows if r['shape']['output_dtype'] == dtype]
        methods = {}
        for role in ROLES:
            arms = [r['methods'].get(role, {}) for r in group]
            errors = [a['errors']['relative_l2'] for a in arms
                      if a.get('eligible') and 'relative_l2' in a.get('errors', {})]
            methods[role] = dict(
                vs_tuned_native=dict(Counter(classify(a) for a in arms)),
                vs_cubic=dict(Counter(classify(a, 'cubic') for a in arms)),
                eligible=sum(bool(a.get('eligible')) for a in arms),
                relative_l2_median=statistics.median(errors) if errors else None,
                relative_l2_max=max(errors) if errors else None)
        policy = Counter()
        for r in group:
            decision = r['recommendation']
            candidate = decision.get('candidate') or {}
            label = ('s' + str(candidate['depth'])) if candidate.get('depth') else candidate.get('family', 'unavailable')
            policy[label] += 1
        result[dtype] = dict(shapes=len(group), methods=methods,
                            frozen_policy_recommendations=dict(policy))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    history = read(args.history)
    chosen = collect(history, 'v6e')
    if not chosen:
        raise ValueError('No verified complete comparisons')
    print(json.dumps(dict(verified_shapes=len(chosen))), flush=True)
    view = args.output_dir.parent / (args.output_dir.name + '-input-view')
    configs = view / 'source/configs'
    configs.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASE / 'source/configs/arch_v6e_168_v001', configs / 'arch_v6e_168_v001')
    for pair in chosen.values():
        for item in pair.values():
            run = Path(item['run'])
            write(view / (item['phase'] + '-finished.json'), dict(status='completed', run=str(run),
                  completion_sha256=sha(run / 'completion.json'), provenance=item))
    # This frozen report intentionally returns 1 for partial coverage. It does
    # not reselect from confirmation, alter experiments, or issue TPU calls.
    code = build(view, args.output_dir)
    data = read(args.output_dir / 'results.json')
    assert code == (0 if data['completed'] else 1)
    assert len(data['results']) == 2 * len(chosen)
    assert all(r['terminal_outcomes'] == r['expected_outcomes'] for r in data['results'])
    summary = dict(interim=True, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                   verified_shapes=len(chosen), planned_shapes=168,
                   by_output=summarize(data['results']),
                   case_status_counts=data['case_status_counts'])
    write(args.output_dir / 'interim_summary.json', summary)
    write(args.output_dir / 'recovery_provenance.json', dict(history=history, units=chosen,
          rule='Whole comparisons only; each screen/confirm pair shares one device identity. Never pool retry samples.'))
    lines = ['# Interim v6e results', '', f"Verified {len(chosen)}/168 shapes at {summary['finished_utc']}.", '',
             'Results use three fresh inputs and 90 paired timing rounds per arm. Wins/losses require the paired pointwise 95% interval to lie entirely above/below 1; intervals crossing 1 are unresolved. No multiple-comparison adjustment. All shapes and Gaussian inputs are development data.', '',
             'The execution order is not a random sample of the grid. These counts are not an estimate of the final 168-shape win rate. v5e has a separate study.', '',
             '| Output | Method | Wins vs tuned Native | Losses | Unresolved | Unavailable/ineligible |',
             '|---|---|---:|---:|---:|---:|']
    for dtype, group in summary['by_output'].items():
        for role in ('cubic', 's1', 's2'):
            c = group['methods'][role]['vs_tuned_native']
            lines.append('| ' + dtype + ' | ' + role + ' | ' + ' | '.join(str(c.get(k, 0)) for k in
                         ('win', 'loss', 'inconclusive', 'unavailable_or_ineligible')) + ' |')
    lines += ['', '## Numerical error', '',
              'Relative L2 values below are the median and maximum across shapes of each shape’s worst of three seeds. Only eligible headline arms are summarized. Reference is FP64 on exact BF16 operands, either full output or 128 × 128 sampled output entries across all K; output finiteness is checked in full.', '',
              '| Output | Method | Median relative L2 % | Maximum relative L2 % |', '|---|---|---:|---:|']
    for dtype, group in summary['by_output'].items():
        for role in ROLES:
            m = group['methods'][role]
            vals = ['—' if m[k] is None else f'{100*m[k]:.6f}' for k in ('relative_l2_median', 'relative_l2_max')]
            lines.append('| ' + ' | '.join([dtype, role, *vals]) + ' |')
    lines += ['', 'The full five-method timings, frozen tuner decisions and counterfactual comparisons are in RESULTS.md. All confirmation arms and error metrics are retained in results.json; all screening decisions are in candidate_decisions.json. This snapshot does not change the running campaign or its selection rules.', '']
    (args.output_dir / 'SUMMARY.md').write_text('\n'.join(lines))
    report = args.output_dir / 'RESULTS.md'
    report.write_text(f'> INTERIM SNAPSHOT: {len(chosen)}/168 shapes; the experiment is still running.\n\n' + report.read_text())
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
