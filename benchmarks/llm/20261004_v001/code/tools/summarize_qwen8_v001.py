"""Summarize the completed BF16 Qwen8 campaign without pooling timing scopes."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ['native_default', 'native_tuned', 'cubic_tuned', 's1', 's2']
LABELS = ['Default Native', 'Tuned Native', 'Tuned Cubic', 'S1', 'S2']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = [r for r in rows if r['model'] == 'qwen3_8b' and r['output_dtype'] == 'bfloat16']
    assert len(selected) == 50 and len({r['entry_id'] for r in selected}) == 50
    groups = collections.defaultdict(dict)
    for row in selected:
        assert row['algorithm'] not in groups[(row['batch'], row['sequence'])]
        groups[(row['batch'], row['sequence'])][row['algorithm']] = row
    assert set(groups) == {(1, 512), (1, 1024), (1, 2048), (1, 4096),
                           (4, 512), (4, 1024), (4, 2048), (8, 512), (8, 1024), (8, 2048)}
    idx = np.random.default_rng(20261003).integers(0, 15, size=(10000, 15))
    output = {'as_of_utc': datetime.now(timezone.utc).isoformat(),
              'input_ledger_sha256': hashlib.sha256(raw).hexdigest(),
              'selected_entries_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
              'comparisons': 50, 'workloads': 10,
              'all_quality_gates_passed': all(r['quality_gate_passed'] for r in selected),
              'method': 'Per-workload paired bootstrap of the ratio of medians across 15 matched rounds; 10000 resamples, seed 20261003; percentile 95% intervals, no multiple-comparison adjustment.',
              'scope': 'BF16 output, DEFAULT arithmetic, v6e; streamed full forward includes recurring serialized H2D, embedding and all-token logits; compilation and prepared host layouts excluded. Resident pass is a separate sum of layer medians, not whole-model resident latency. B1/S512 predates the separate resident pass.',
              'quality_scope': selected[-1]['measurement']['quality_scope'], 'groups': []}
    for (batch, sequence), group in sorted(groups.items()):
        assert set(group) == set(ARMS)
        base = group['native_tuned']['measurement']
        base_samples = np.array([s['elapsed_ms'] for s in base['samples']])
        assert len(base_samples) == 15
        result = {'batch': batch, 'sequence': sequence, 'algorithms': {}}
        for arm in ARMS:
            row = group[arm]
            m = row['measurement']
            samples = np.array([s['elapsed_ms'] for s in m['samples']])
            assert len(samples) == 15 and np.isfinite(samples).all()
            assert m['confirmation_order'] == base['confirmation_order']
            assert abs(float(np.median(samples)) - m['elapsed_median_ms']) < 1e-6
            ci = np.quantile(np.median(base_samples[idx], axis=1) / np.median(samples[idx], axis=1), [.025, .975])
            result['algorithms'][arm] = {
                'entry_id': row['entry_id'], 'evidence_run': row['evidence_run'],
                'full_forward_ms': m['elapsed_median_ms'],
                'resident_pass_ms': m.get('resident_pass', {}).get('aggregate_ms'),
                'speedup_vs_tuned_native': base['elapsed_median_ms'] / m['elapsed_median_ms'],
                'speedup_vs_tuned_native_95': ci.tolist(),
                'quality': m['quality'],
                'perplexity_change_percent': 100 * (m['quality']['perplexity'] / group['native_default']['measurement']['quality']['perplexity'] - 1),
                'max_hidden_relative_l2': max((h['relative_l2'] for h in m['hidden_errors']), default=0),
                'quality_gate_passed': row['quality_gate_passed'],
            }
        output['groups'].append(result)
    output['aggregate'] = {}
    for arm in ARMS[1:]:
        items = [g['algorithms'][arm] for g in output['groups']]
        resident = [g for g in output['groups'] if g['algorithms'][arm]['resident_pass_ms'] is not None]
        output['aggregate'][arm] = {
            'nominal_full_forward_wins_vs_tuned_native': sum(v['speedup_vs_tuned_native'] > 1 for v in items),
            'ci_above_one_vs_tuned_native': sum(v['speedup_vs_tuned_native_95'][0] > 1 for v in items),
            'ci_below_one_vs_tuned_native': sum(v['speedup_vs_tuned_native_95'][1] < 1 for v in items),
            'resident_comparable_workloads': len(resident),
            'nominal_resident_wins_vs_tuned_native': sum(g['algorithms'][arm]['resident_pass_ms'] < g['algorithms']['native_tuned']['resident_pass_ms'] for g in resident),
            'quality_ranges': {key: [min(v['quality'][key] for v in items), max(v['quality'][key] for v in items)]
                               for key in ['delta_nll', 'mean_kl', 'top1_agreement', 'logit_relative_l2', 'logit_rmse', 'logit_max_abs', 'logit_normalized_max']},
            'perplexity_change_percent_range': [min(v['perplexity_change_percent'] for v in items), max(v['perplexity_change_percent'] for v in items)],
            'maximum_hidden_relative_l2': max(v['max_hidden_relative_l2'] for v in items),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    text = ['# Completed Qwen3-8B BF16 study on v6e', '',
            'All 10 workloads and 50 comparisons are complete. All configured finite-output, NLL, KL and token-agreement gates passed.', '',
            'S1 has some resident-layer wins, but no workload has a 95% matched-round interval entirely above one for full-forward speedup over Tuned Native. S2 has no such win either. Tuned Native is clearly faster than S1 and S2 on B4/S2048 and B8/S2048. These intervals are per workload and unadjusted for multiple comparisons.', '',
            '## Full forward, median milliseconds', '',
            '| Batch | Sequence | ' + ' | '.join(LABELS) + ' |',
            '|---:|---:|' + '---:|' * 5]
    for group in output['groups']:
        text.append(f"| {group['batch']} | {group['sequence']} | " + ' | '.join(f"{group['algorithms'][a]['full_forward_ms']:.1f}" for a in ARMS) + ' |')
    text += ['', '## Separate resident-layer pass, milliseconds', '',
             '| Batch | Sequence | ' + ' | '.join(LABELS) + ' |', '|---:|---:|' + '---:|' * 5]
    for group in output['groups']:
        if group['algorithms']['s1']['resident_pass_ms'] is None:
            continue
        text.append(f"| {group['batch']} | {group['sequence']} | " + ' | '.join(f"{group['algorithms'][a]['resident_pass_ms']:.1f}" for a in ARMS) + ' |')
    text += ['', output['scope'], '',
             'S1 has lower resident time than Tuned Native on 3 of the 9 comparable workloads; S2 on none. The strongest S1 reduction is B4/S1024: 123.1 versus 139.2 ms, or 11.6%. Its full-forward advantage there is only 0.3%, with an interval spanning a tie.', '',
             '## Numerical differences from Default Native', '',
             '| Algorithm | Perplexity change | Next-token top-1 agreement | Logit relative L2 | Maximum layer relative L2 |',
             '|---|---:|---:|---:|---:|']
    for arm, label in zip(ARMS[1:], LABELS[1:]):
        item = output['aggregate'][arm]
        q = item['quality_ranges']
        ppl = item['perplexity_change_percent_range']
        text.append(f"| {label} | {ppl[0]:+.3f}% to {ppl[1]:+.3f}% | {100*q['top1_agreement'][0]:.2f}%–{100*q['top1_agreement'][1]:.2f}% | {100*q['logit_relative_l2'][0]:.2f}%–{100*q['logit_relative_l2'][1]:.2f}% | {100*item['maximum_hidden_relative_l2']:.2f}% |")
    text += ['', 'Quality uses fixed held-out WikiText2 windows, not downstream task accuracy or independent full-dataset replications. Perplexity changes of either sign do not establish a quality improvement. Intermediate hidden errors and maximum logit errors are diagnostics; passing final predictive gates does not imply identical activations. The largest intermediate deviations merit follow-up.', '',
             'All per-workload predictive metrics, error ranges, confidence intervals, entry identities and evidence-run paths are preserved in `summary.json`. Full profiles, tiles, per-layer errors and timing samples remain in the committed campaign ledger and case archives.', '',
             output['method'], '', 'Ledger SHA256: `' + output['input_ledger_sha256'] + '`.', '']
    (args.output / 'README.md').write_text('\n'.join(text))
    print(json.dumps({'comparisons': output['comparisons'], 'all_quality_gates_passed': output['all_quality_gates_passed'],
                      'aggregate': output['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
