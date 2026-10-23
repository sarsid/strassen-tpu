"""Audit the frozen eight-pair depth probe and graph all pairs, including failures."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def fmt(value, digits=3): return 'unavailable' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    cohort = parser.parse_args().cohort.resolve()
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir()
    receipt = read(cohort / 'LEVEL-poc-finished.json')
    run = Path(receipt['run'])
    assert run.resolve().is_relative_to(cohort / 'phases')
    assert sha(run / 'completion.json') == receipt['completion_sha256']
    completion = read(run / 'completion.json')
    assert completion['status'] == 'completed' and completion['remote_may_still_be_running'] is False
    checks = 0
    for name, expected in read(run / 'artifact-manifest.json')['sha256'].items():
        assert sha(run / name) == expected, name
        checks += 1
    frozen = read(cohort / 'frozen.json')
    assert sha(cohort / 'source.tar') == frozen['archive_sha256']
    for name, expected in frozen['source_sha256'].items():
        assert sha(cohort / 'source' / name) == expected, name
        checks += 1
    config = read(cohort / 'source/configs/two_level_poc_v001/campaign.json')
    shapes = read(cohort / 'source/configs/two_level_poc_v001/shapes.json')['shapes']
    groups = read(run / 'artifacts/planned_cases.json')
    rows = [json.loads(line) for line in (run / 'artifacts/results.jsonl').read_text().splitlines()]
    summary = read(run / 'artifacts/summary.json')
    assert summary['completed'] and len(summary['completed_groups']) == len(groups) == 8
    cases = [r for r in rows if r['event'] == 'case_result']
    assert len(cases) == 16
    assert dict(Counter(r['status'] for r in cases)) == summary['case_status_counts']
    env = read(run / 'artifacts/environment.json')
    assert env['qualified_single_v5e'] and env['identity']['allocation_id'] == frozen['allocation_id']
    by, samples, fingerprints = {}, {}, {}
    for r in rows:
        if r['event'] == 'case_start':
            shape_id = r['shape_id']
            if shape_id in fingerprints: assert fingerprints[shape_id] == r['input_fingerprint']
            fingerprints[shape_id] = r['input_fingerprint']
            checks += 1
    for r in cases:
        key = (r['group_id'], r['arm_id'])
        assert key not in by
        by[key] = r
        ss = sorted([s for s in rows if s['event'] == 'sample' and (s['group_id'], s['arm_id']) == key], key=lambda s: s['round'])
        samples[key] = ss
        assert len(ss) == r['timing']['sample_count']
        if ss:
            assert len(ss) == 30 and [s['round'] for s in ss] == list(range(30))
            values = np.array([s['elapsed_ms'] for s in ss])
            np.testing.assert_allclose(values.mean(), r['timing']['mean_ms'], rtol=1e-13)
            assert np.all(np.isfinite(values)) and np.all(values > 0)
        metric = r.get('correctness')
        if metric:
            gate = config['correctness']['gate']
            passed = metric['finite'] and metric['relative_l2'] <= gate['relative_l2_max'] and metric['max_abs_error'] <= gate['max_abs_atol'] + gate['max_abs_reference_rtol'] * metric['max_abs_reference']
            assert bool(passed) == metric['pass']
            assert r['eligible_for_speedup_claim'] == (r['status'] == 'ok')
        checks += 1
    comparisons = []
    for g in groups:
        gid = g['group_id']
        one, two = (by[gid, arm] for arm in config['arms'])
        sa, sb = (samples[gid, arm] for arm in config['arms'])
        c = dict(group_id=gid, shape_id=g['shape']['id'], shape_mkn=[g['shape'][d] for d in ('m', 'k', 'n')], tile=g['tile'],
                 one_level_ms=one['timing'].get('mean_ms'), two_level_ms=two['timing'].get('mean_ms'),
                 one_level_status=one['status'], two_level_status=two['status'],
                 one_level_l2=(one.get('correctness') or {}).get('relative_l2'), two_level_l2=(two.get('correctness') or {}).get('relative_l2'),
                 eligible=one['eligible_for_speedup_claim'] and two['eligible_for_speedup_claim'])
        if len(sa) == len(sb) == 30:
            assert all({a['position'], b['position']} == {0, 1} for a, b in zip(sa, sb))
            assert Counter(a['position'] for a in sa) == {0: 15, 1: 15}
            a, b = (np.array([s['elapsed_ms'] for s in ss]) for ss in (sa, sb))
            ix = np.random.default_rng(2026092131).integers(0, 30, (4000, 30))
            ratios = a[ix].mean(axis=1) / b[ix].mean(axis=1)
            c.update(speedup_two_over_one=float(a.mean() / b.mean()), ci95=np.quantile(ratios, [.025, .975]).tolist(),
                     latency_reduction_percent=float(100 * (1 - b.mean() / a.mean())))
            checks += 2
        comparisons.append(c)
    result = dict(cohort=str(cohort), run=str(run), allocation=frozen['allocation_id'], comparisons=comparisons, cases=cases,
                  audit=dict(passed=True, checks=checks, outcome_counts=summary['case_status_counts']),
                  scope='Eight matched-tile pairs, four larger synthetic shapes, one v5e, one Gaussian seed per shape. No exhaustive tuning or end-to-end LLM claim.')
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# One-level versus two-level Strassen on larger shapes', '',
             'Four larger synthetic matrix products on one TPU v5e, with two shared tiles per shape. All eight pairs are reported.', '',
             'Means of 30 synchronized rounds after five warmups; the two-arm order alternates where both compile. Full-call latency includes device padding and crop; compilation and host transfers are excluded. Both use BF16 inputs/pre-additions and FP32 accumulation/output.', '',
             '| Shape (M x K x N) | Tile (BM, BN, BK) | One level ms | Two levels ms | One / two [95% CI] | Status (one / two) |',
             '|---|---|---:|---:|---:|---|']
    for c in comparisons:
        ratio = f'{c["speedup_two_over_one"]:.3f} [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}]' if 'ci95' in c else 'unavailable'
        lines.append(f'| {c["shape_id"]}: {" x ".join(map(str,c["shape_mkn"]))} | {c["tile"]} | {fmt(c["one_level_ms"])} | {fmt(c["two_level_ms"])} | {ratio} | {c["one_level_status"]} / {c["two_level_status"]} |')
    lines += ['', 'Ratios above 1 favor two levels. Intervals are pointwise paired bootstrap intervals conditional on these 30 rounds, not independent cross-machine confidence or guarantees across inputs. Failed numerical gates disqualify speedup claims.', '',
              '## Numerical error', '', '| Shape | Tile BK | One-level relative L2 | Two-level relative L2 |', '|---|---:|---:|---:|']
    for c in comparisons:
        lines.append(f'| {c["shape_id"]} | {c["tile"][2]} | {fmt(c["one_level_l2"],6)} | {fmt(c["two_level_l2"],6)} |')
    lines += ['', 'The reference uses the exact quantized BF16 operands cast to host FP32, every K term, and 128 x 128 sampled output positions including edges. Whole-output finiteness is checked. Passing the existing relative-L2 <= 0.02 and absolute-error gates does not mean equal error or unchanged model quality.', '',
              '## What this probe can establish', '',
              '- The second level replaces seven half-tile products with 49 quarter-tile products. This reduces multiplication work by another 12.5%, but adds combinations, scratch-buffer traffic, and scheduling overhead.',
              '- This initial implementation reuses one FP32 half-output-tile scratch buffer for each outer product. The result assesses this schedule and these two tile choices, not the best possible two-level kernel.',
              '- The parent tiles are 2048 x 2048 with K panels of 512 or 1024. Two-level leaf products therefore have M=512, N=512, K=128 or 256. Increasing the full matrix alone does not enlarge these leaves.',
              '- Comparing the best observed tile per depth would be exploratory selection from two choices, not independently confirmed optimal tuning. All matched pairs are retained above.',
              '- Qwen, Mistral, and Gemma name model-derived projection dimensions. These are synthetic matrix products, not full-model inference measurements.', '',
              '## Evidence', '', f'- Allocation: `{frozen["allocation_id"]}`.', f'- Frozen source revision: `{frozen["baseline_commit"]}`.',
              f'- Phase archive: `{run}`.', f'- {checks} source/artifact/statistics/status checks passed.', f'- Outcomes: `{json.dumps(summary["case_status_counts"],sort_keys=True)}`.']
    for r in cases:
        if r['status'] != 'ok': lines.append(f'- Failure `{r["group_id"]}` / `{r["arm_id"]}`: {r.get("error_message")}')
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    os.environ['MPLCONFIGDIR'] = str(out / 'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.subplots_adjust(top=.84, bottom=.11, hspace=.48, left=.09, right=.97)
    labels = ['12288 square', 'Qwen gate/up', 'Mistral down', 'Gemma gate/up']
    overall_maximum = max(c[k] for c in comparisons for k in ('one_level_ms', 'two_level_ms') if c[k] is not None)
    for ax, tile in zip(axes, config['tiles']):
        selected = [c for c in comparisons if c['tile'] == tile]
        maximum = max([c[k] for c in selected for k in ('one_level_ms', 'two_level_ms') if c[k] is not None] or [1])
        for i, (arm, color) in enumerate([('one_level', '#3478AC'), ('two_level', '#D18B32')]):
            vals = [c[arm + '_ms'] if c[arm + '_ms'] is not None else np.nan for c in selected]
            bars = ax.bar(np.arange(4) + (i - .5) * .3, vals, .27, label=arm.replace('_', ' ').title(), color=color, zorder=3)
            for j, (bar, value, c) in enumerate(zip(bars, vals, selected)):
                if c[arm + '_status'] != 'ok': bar.set_hatch('///'); bar.set_edgecolor('#333333')
                if np.isfinite(value): ax.text(bar.get_x() + bar.get_width()/2, value + maximum * .025, f'{value:.2f}', ha='center', fontsize=10)
                else: ax.text(j + (i - .5) * .3, maximum * .04, 'failed', rotation=90, ha='center', fontsize=9)
        for j, c in enumerate(selected):
            if c['eligible']:
                ax.hlines(min(c['one_level_ms'], c['two_level_ms']), j - .34, j + .34, color='#202020', linestyle='--', linewidth=1, zorder=4)
        ax.set_title(f'Parent tile BM=2048, BN=2048, BK={tile[2]}', loc='left', pad=12)
        ax.set_xticks(range(4), labels); ax.set_ylabel('Mean latency (ms)')
        ax.set_ylim(0, overall_maximum * 1.22); ax.set_xlim(-.6, 3.6); ax.grid(axis='y', alpha=.2, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(handles=[Patch(color='#3478AC', label='One level'), Patch(color='#D18B32', label='Two levels')], loc='upper left', bbox_to_anchor=(0, 1.30), ncol=2, frameon=False)
    fig.suptitle('Does a second Strassen level help on larger matrices?', x=.09, ha='left', fontsize=17)
    fig.text(.09, .045, 'One TPU v5e • 30 rounds per arm • BF16 inputs, FP32 output\nLower is faster. Dashed lines mark the faster eligible result in each pair. Two tile choices; no full tuning sweep.', fontsize=10)
    fig.savefig(out / 'latency.png', dpi=180, facecolor='white'); plt.close(fig)
    (out / 'artifact-manifest.json').write_text(json.dumps({p.name: sha(p) for p in out.iterdir() if p.is_file()}, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), audit_checks=checks, outcomes=summary['case_status_counts'], comparisons=comparisons), indent=2))


if __name__ == '__main__': main()
