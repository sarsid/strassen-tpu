"""Detailed numerical/prediction costs and MM-to-model consistency checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    roots = {}
    for stage in ('tune', 'resident', 'quality'):
        receipt = read(args.cohort / (args.model + '-' + stage + '-finished.json'))
        if receipt['status'] != 'completed':
            raise ValueError('Required stage did not complete: ' + stage)
        roots[stage] = Path(receipt['run']) / 'artifacts'
    quality = read(roots['quality'] / 'quality.json')
    layers = read(roots['quality'] / 'layer_errors.json')
    resident = read(roots['resident'] / 'resident_timings.json')
    setup = read(roots['resident'] / 'resident_setup.json')
    count = max(row['layer'] for row in layers) + 1
    arms = list(quality)
    if not all(quality[a]['all_finite'] for a in arms):
        raise ValueError('Nonfinite quality output requires explicit failure analysis')
    lines = ['# Detailed error and timing analysis', '',
             f'Model: {args.model}; {count} layers; {quality[arms[0]]["positions"]} held-out targets.',
             'All numerical and prediction differences below use default composed Native as reference.',
             'Passing the frozen gate is not proof of zero accuracy cost or equivalent downstream task performance.', '',
             '| Metric | ' + ' | '.join(arms) + ' |',
             '|---|' + '---:|' * len(arms)]
    metrics = [
        ('Top-1 accuracy (%)', 'candidate_top1_accuracy', 100),
        ('Top-5 accuracy (%)', 'candidate_top5_accuracy', 100),
        ('Top-1 agreement (%)', 'top1_agreement', 100),
        ('Correct to wrong (tokens)', 'correct_to_wrong', 1),
        ('Wrong to correct (tokens)', 'wrong_to_correct', 1),
        ('Perplexity', 'candidate_perplexity', 1),
        ('Mean NLL', 'candidate_nll', 1),
        ('NLL change', 'nll_delta', 1),
        ('Logit relative L2 (%)', 'relative_l2', 100),
        ('Logit RMSE', 'rmse', 1),
        ('Logit mean absolute error', 'mean_abs_error', 1),
        ('Logit maximum absolute error', 'max_abs_error', 1),
        ('Logit cosine similarity', 'logit_cosine', 1),
        ('Mean forward KL', 'mean_kl', 1),
        ('Mean reverse KL', 'mean_reverse_kl', 1),
        ('Mean Jensen-Shannon divergence', 'mean_js', 1),
        ('Mean total variation', 'mean_total_variation', 1),
        ('Mean Brier score', 'mean_brier', 1),
        ('Nonfinite positions', 'nonfinite_positions', 1),
    ]
    for label, key, scale in metrics:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{quality[a][key]*scale:.7g}' for a in arms) + ' |')
    lines.append('| Calibration ECE (15 bins) | ' + ' | '.join(f'{quality[a]["calibration"]["value"]:.7g}' for a in arms) + ' |')
    for label, key, scale in [('Accuracy change, 95% CI (pp)', 'paired_accuracy_delta', 100),
                              ('NLL change, 95% CI', 'paired_nll_delta', 1)]:
        cells = []
        for arm in arms:
            low, high = quality[arm][key]['ci95']
            cells.append(f'[{low*scale:+.7g}, {high*scale:+.7g}]')
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    for key, label in [('token_max_logit_error_quantiles', 'Per-token maximum logit error'),
                       ('nll_delta_quantiles', 'NLL delta')]:
        for quantile in quality[arms[0]][key]:
            lines.append('| ' + label + ' ' + quantile + ' | ' +
                         ' | '.join(f'{quality[a][key][quantile]:.7g}' for a in arms) + ' |')
    lines += ['', 'Intervals resample the 16 text windows, preserving token pairing. They describe one corpus and one allocation.', '',
              '## Hidden-state accumulation', '',
              'These measurements use the first held-out prompt and every layer. Local errors share Native input; accumulated errors follow each arm\'s own preceding states.', '',
              '| Arm | Worst local relative L2 | Worst accumulated relative L2 | Final accumulated relative L2 | Final max absolute error |',
              '|---|---:|---:|---:|---:|']
    for arm in arms:
        local = [r for r in layers if r['arm_id'] == arm and r['scope'] == 'local_same_native_input']
        propagated = [r for r in layers if r['arm_id'] == arm and r['scope'] == 'accumulated_own_forward_state']
        final = max(propagated, key=lambda r: r['layer'])['metrics']
        lines.append(f'| {arm} | {100*max(r["metrics"]["relative_l2"] for r in local):.5f}% | '
                     f'{100*max(r["metrics"]["relative_l2"] for r in propagated):.5f}% | '
                     f'{100*final["relative_l2"]:.5f}% | {final["max_abs_error"]:.7g} |')
    confirmations = {}
    for projection in ('gateup', 'down'):
        value = read(roots['tune'] / ('m2048_' + projection + '_confirmation_statistics.json'))
        if len(value['by_shape']) != 1:
            raise ValueError('Expected one confirmed shape per projection')
        confirmations[projection] = next(iter(value['by_shape'].values()))['headline']
    lines += ['', '## Does the MM change translate into model latency?', '',
              'The estimate multiplies the confirmed first-layer gate/up-plus-down time difference by the layer count. '
              'This is a consistency check, not a device profile: isolated calls synchronize individually and can have different cache/dispatch behavior.', '',
              '| Arm | Isolated MMs × layers: latency change (ms) | Measured full prompt change (ms) | Measured full prompt change (%) |',
              '|---|---:|---:|---:|']
    translation = {}
    timing = resident['prompt_forward_last_token']['timings']
    for arm in ('cubic', 'one_level', 'two_level'):
        comparisons = [confirmations[p][arm]['comparison_vs_tuned_native'] for p in ('gateup', 'down')]
        expected = count * sum(c['candidate_mean_ms'] - c['reference_mean_ms'] for c in comparisons)
        observed = timing[arm]['mean_ms'] - timing['native']['mean_ms']
        percent = 100 * observed / timing['native']['mean_ms']
        translation[arm] = dict(estimated_mm_delta_ms=expected, observed_model_delta_ms=observed, observed_percent=percent)
        lines.append(f'| {arm} | {expected:+.5f} | {observed:+.5f} | {percent:+.3f}% |')
    lines += ['', f'Resident device payload: {setup["loaded_device_bytes"]:,} bytes. All model weights were loaded before timers started.',
              'Compilation, checkpoint I/O and weight transfer are excluded. Prompt-forward timing includes embedding, all layers, last-token head, dispatch and final synchronization.', '',
              'Evidence:'] + ['- ' + str(path) for path in roots.values()]
    (args.output_dir / 'ERROR_DETAILS.md').write_text('\n'.join(lines) + '\n')
    (args.output_dir / 'analysis.json').write_text(json.dumps(dict(translation=translation, input_sha256=hashes), indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output_dir), model=args.model, layers=count)))


if __name__ == '__main__':
    main()
