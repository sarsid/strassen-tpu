"""Compare frozen choices on the same 168 development shapes, without pooling chips."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_v6e_suite_v001 import paired

METHODS = ('native_default', 'native', 'cubic', 'one_level', 'two_level')
LABELS = {'native': 'Tuned Native', 'native_default': 'Default Native', 'cubic': 'Tuned cubic',
          'one_level': 'Strassen 1', 'two_level': 'Strassen 2'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def classification(q):
    if not q or not q.get('eligible', True):
        return 'unavailable_or_ineligible'
    lo, hi = q['ci95']
    return 'win' if lo > 1 else 'loss' if hi < 1 else 'inconclusive'


def geo(values):
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def main(root, out):
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(p):
        hashes[str(p.relative_to(root))] = digest(p)
        return json.loads(p.read_text())

    def phase(cohort, name):
        receipt = read(cohort / (name + '-finished.json'))
        run = Path(receipt['run'])
        assert run.is_relative_to(cohort / 'phases')
        assert receipt['status'] == 'completed'
        assert digest(run / 'completion.json') == receipt['completion_sha256']
        completion = read(run / 'completion.json')
        assert not completion.get('remote_may_still_be_running')
        manifest = read(run / 'artifact-manifest.json')['sha256']
        for name in ('results.jsonl', 'planned_cases.json', 'environment.json', 'summary.json'):
            p = run / 'artifacts' / name
            assert digest(p) == manifest[str(p.relative_to(run))], p
        art = run / 'artifacts'
        groups = read(art / 'planned_cases.json')
        identity = read(art / 'environment.json')['identity']
        assert read(art / 'summary.json')['completed']
        raw_path = art / 'results.jsonl'
        hashes[str(raw_path.relative_to(root))] = digest(raw_path)
        samples, cases = defaultdict(dict), defaultdict(list)
        with raw_path.open() as handle:
            for line in handle:
                r = json.loads(line)
                if r['event'] not in ('sample', 'case_result'):
                    continue
                key = r['group_id'], r['arm_id']
                if r['event'] == 'sample':
                    pair = r['seed'], r['round']
                    assert pair not in samples[key]
                    samples[key][pair] = r['elapsed_ms']
                else:
                    cases[key].append(r)
        return art, groups, identity, samples, cases

    v6path = root / 'runs/20260924-v6e-suite-recovery-v001/operations/combined-report/artifacts/report/results.json'
    v6report = read(v6path)
    assert v6report['completed'] and not v6report['missing_batches']
    v6 = {r['shape_id']: r for r in v6report['results']}
    v5 = {}
    environments = {'v5e': [], 'v6e': []}
    v5cohort = root / 'runs/20260921-mlsys-main-v5e-v001'
    for batch in range(1, 13):
        art, groups, identity, samples, cases = phase(v5cohort, f'MAIN-{batch:02d}-confirm')
        environments['v5e'].append(identity)
        stats = read(art / 'confirmation_statistics.json')['by_shape']
        manifest = read(art.parent / 'artifact-manifest.json')['sha256']
        assert digest(art / 'confirmation_statistics.json') == manifest['artifacts/confirmation_statistics.json']
        for g in groups:
            sid = g['shape']['id']
            assert sid not in v5
            methods = {}
            for f in METHODS:
                info = stats[sid]['headline'][f]
                arm = info['candidate_id']
                assert g['headline_candidates'][f] == arm
                observed = cases[g['group_id'], arm]
                raw = samples[g['group_id'], arm]
                assert len(raw) == 90 and len(observed) == 3
                eligible = info['valid_numerical_comparison_vs_tuned_native']
                q = info['comparison_vs_tuned_native']
                speed = dict(speedup=1/q['candidate_over_reference_latency_ratio'],
                             ci95=[1/q['ci95'][1], 1/q['ci95'][0]], eligible=eligible)
                metadata = observed[0]['kernel_metadata']
                errors = {k: max(c['correctness'][k] for c in observed)
                          for k in ('relative_l2', 'max_abs_error', 'rmse', 'mean_abs_error',
                                    'p99_abs_error', 'normwise_error')}
                methods[f] = dict(mean_ms=statistics.mean(raw.values()), comparison=speed,
                                  eligible=eligible, errors=errors, candidate_id=arm,
                                  tile=metadata.get('tile_bm_bn_bk'),
                                  padded_volume_ratio=metadata.get('padded_volume_ratio'))
                assert math.isclose(methods[f]['mean_ms'], q['candidate_mean_ms'], rel_tol=1e-10)
            s1, s2 = [g['headline_candidates'][f] for f in ('one_level', 'two_level')]
            s2_vs_s1 = paired(samples[g['group_id'], s1], samples[g['group_id'], s2])
            assert s2_vs_s1
            s2_vs_s1['eligible'] = methods['one_level']['eligible'] and methods['two_level']['eligible']
            v5[sid] = dict(shape_mkn=[g['shape'][k] for k in ('m', 'k', 'n')], methods=methods,
                           s2_vs_s1=s2_vs_s1, allocation_id=identity['allocation_id'],
                           evidence=str(art.relative_to(root)))

    for cohort_name, batches in [('20260923-v6e-suite-v001', range(1, 10)),
                                ('20260924-v6e-suite-recovery-v001', range(10, 13))]:
        for batch in batches:
            art, groups, identity, samples, cases = phase(root / 'runs' / cohort_name, f'suite-{batch:02d}-confirm')
            environments['v6e'].append(identity)
            for g in groups:
                r = v6[g['shape']['id']]
                assert r['provenance']['allocation_id'] == identity['allocation_id']
                arms = {role: a['arm_id'] for a in g['arms'] for role in a['headline_roles']}
                for f in METHODS:
                    raw = samples[g['group_id'], arms[f]]
                    assert len(raw) == 90
                    assert math.isclose(statistics.mean(raw.values()), r['methods'][f]['mean_ms'], rel_tol=1e-10)
                r['s2_vs_s1'] = paired(samples[g['group_id'], arms['one_level']], samples[g['group_id'], arms['two_level']])
                assert r['s2_vs_s1']
                r['s2_vs_s1']['eligible'] = r['methods']['one_level']['eligible'] and r['methods']['two_level']['eligible']
    assert set(v5) == set(v6) and len(v5) == 168
    rows = []
    for sid, r in v6.items():
        assert r['shape_mkn'] == v5[sid]['shape_mkn']
        m6 = {}
        for f in METHODS:
            x = r['methods'][f]
            m6[f] = dict(mean_ms=x['mean_ms'], comparison=x['comparisons']['native'], eligible=x['eligible'],
                         errors=x['errors'], candidate_id=x['candidate']['candidate_id'], tile=x['candidate']['tile'],
                         padded_volume_ratio=x['padded_volume_ratio'], device_mean_ms=x['device_profile']['mean_ms'],
                         device_speedup_vs_tuned=x['device_speedup_vs_tuned'],
                         comparison_vs_cubic=x['comparisons']['cubic'])
        rows.append(dict(shape_id=sid, shape_mkn=r['shape_mkn'], category=r['category'],
                         v5e=v5[sid], v6e=dict(methods=m6, s2_vs_s1=r['s2_vs_s1'], provenance=r['provenance'])))

    def aggregate(subset, chip):
        result = {'n': len(subset)}
        for f in METHODS:
            ms = [r[chip]['methods'][f] for r in subset]
            result[f] = dict(counts=dict(Counter(classification(m['comparison']) for m in ms)),
                             geomean_speedup=geo([m['comparison']['speedup'] for m in ms if m['eligible']]),
                             worst_relative_l2=max(m['errors']['relative_l2'] for m in ms),
                             median_relative_l2=statistics.median(m['errors']['relative_l2'] for m in ms))
        result['s2_vs_s1'] = dict(Counter(classification(r[chip]['s2_vs_s1']) for r in subset))
        result['fastest_mean_native_s1_s2'] = dict(Counter(min(('native', 'one_level', 'two_level'),
            key=lambda f: r[chip]['methods'][f]['mean_ms']) for r in subset))
        return result

    subsets = {'all': rows}
    for category in sorted({r['category'] for r in rows}):
        subsets[category] = [r for r in rows if r['category'] == category]
    summary = {k: {chip: aggregate(rs, chip) for chip in ('v5e', 'v6e')} for k, rs in subsets.items()}
    transitions = {f: dict(Counter(classification(r['v5e']['methods'][f]['comparison']) + ' -> ' +
                                  classification(r['v6e']['methods'][f]['comparison']) for r in rows))
                   for f in ('one_level', 'two_level')}
    cross_chip = {k: {f: geo([r['v5e']['methods'][f]['mean_ms']/r['v6e']['methods'][f]['mean_ms'] for r in rs])
                     for f in METHODS} for k, rs in subsets.items()}
    best = {chip: {f: max(rows, key=lambda r: r[chip]['methods'][f]['comparison']['speedup'])['shape_mkn']
                  for f in ('one_level', 'two_level')} for chip in ('v5e', 'v6e')}
    result = dict(matched_shapes=168, source_policy='Frozen screening choices only; no raw cross-chip pooling.',
                  summary=summary, transitions=transitions, cross_chip_latency_ratios=cross_chip,
                  best_shapes=best, rows=rows, environments=environments, input_sha256=hashes)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    colors = {'small_or_skinny': '#9aa5b1', 'large_2048_aligned': '#087f8c', 'other_large_dimensions': '#d17a22'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, f in zip(axes, ('one_level', 'two_level')):
        for category, color in colors.items():
            rs = subsets[category]
            ax.scatter([r['v5e']['methods'][f]['comparison']['speedup'] for r in rs],
                       [r['v6e']['methods'][f]['comparison']['speedup'] for r in rs], s=30, alpha=.8,
                       color=color, label=category.replace('_', ' '), linewidths=.3, edgecolors='white')
        ax.axhline(1, color='#555', lw=1); ax.axvline(1, color='#555', lw=1)
        ax.plot([.25, 1.45], [.25, 1.45], ls='--', lw=.8, color='#aaa')
        ax.set(xlim=(.25, 1.45), ylim=(.25, 1.45), xlabel='v5e speedup over its tuned Native',
               ylabel='v6e speedup over its tuned Native', title=LABELS[f])
        ax.grid(alpha=.15)
    axes[0].legend(loc='lower right', frameon=False, fontsize=8)
    fig.suptitle('Same 168 shapes: Strassen advantage shrinks on v6e\nEach dot is one confirmed shape; speedup above 1 is faster', fontsize=13)
    fig.savefig(out / 'cross_chip.png', dpi=180)
    fig.savefig(out / 'cross_chip.svg')
    plt.close(fig)

    sizes = [4096, 4097, 8191, 8192, 8193, 16384]
    by_dims = {tuple(r['shape_mkn']): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    method_colors = {'native': '#52616b', 'one_level': '#087f8c', 'two_level': '#b45c99'}
    for ax, chip in zip(axes, ('v5e', 'v6e')):
        for j, f in enumerate(('native', 'one_level', 'two_level')):
            vals = [by_dims[(d,d,d)][chip]['methods'][f]['mean_ms']/
                    by_dims[(d,d,d)][chip]['methods']['native']['mean_ms'] for d in sizes]
            bars = ax.bar(np.arange(len(sizes)) + (j-1)*.24, vals, .23, color=method_colors[f], label=LABELS[f])
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], rotation=90, padding=3, fontsize=7)
        ax.axhline(1, color='#555', lw=.8)
        ax.set(xticks=range(len(sizes)), xticklabels=[f'{d:,}' for d in sizes], ylim=(0, 3.3),
               xlabel='Square dimension (M = K = N)', ylabel='Latency / tuned Native latency (lower is better)', title=chip)
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle('Tile boundaries change the winner; size alone does not', fontsize=14)
    fig.savefig(out / 'boundary_shapes.png', dpi=180); fig.savefig(out / 'boundary_shapes.svg'); plt.close(fig)

    lines = ['# v5e versus v6e: the same 168 shapes', '',
             'All 168 development shapes match exactly by M,K,N. Complete-call latency excludes transfer and compilation. '
             'Each method is its frozen screening choice, confirmed on three fresh BF16 inputs × 30 paired rounds. '
             'Intervals are pointwise 95%, conditional on selection, without multiplicity adjustment. '
             'The v6e run spans two allocations (126 + 42 shapes); each shape keeps its own paired controls. '
             'No raw samples are pooled across chips or allocations.', '',
             'This is a comparison of two hardware/software/tuning stacks, not a hardware-only causal experiment. '
             'v5e used JAX/jaxlib 0.7.2 and libtpu 0.0.21.1; v6e used 0.11.2 and 0.0.48. '
             'v5e screened 16 tiles per custom family; v6e used a dimension-only four/six-tile shortlist from 20 tiles, '
             'with additional pipelined implementations and different VMEM budgets. The reference, BF16 preadds, FP32 output '
             'and numerical gates are comparable, but input seeds differ. No exhaustive optimum or new-shape generalization is established.', '',
             '## Clear wins over tuned Native', '', '| Region | Shapes | v5e S1 | v5e S2 | v6e S1 | v6e S2 |',
             '|---|---:|---:|---:|---:|---:|']
    for k, ss in summary.items():
        lines.append('| '+k+' | '+str(ss['v5e']['n'])+' | '+' | '.join(str(ss[c][f]['counts'].get('win',0))
                     for c,f in [('v5e','one_level'),('v5e','two_level'),('v6e','one_level'),('v6e','two_level')])+' |')
    lines += ['', 'Small/skinny means min(M,K,N) < 1024. Aligned means every dimension is a multiple of 2048. '
              'The remaining shapes form the other-large-dimensions group. These are descriptive regions, not a validated selector.', '',
              '## Direct S2 versus S1 comparison', '',
              'Speedup here is S1 time / S2 time. The same hierarchical bootstrap is applied to both chips, within each shape.', '']
    for chip in ('v5e','v6e'):
        lines.append(f'- {chip}: {summary["all"][chip]["s2_vs_s1"]}.')
    lines += ['', '## Illustrative complete-call timings (ms)', '',
              '| M × K × N | v5e Native | v5e S1 | v5e S2 | v6e Native | v6e S1 | v6e S2 |', '|---|---:|---:|---:|---:|---:|---:|']
    examples = [(d,d,d) for d in sizes] + [(4096,65536,8192),(16384,8192,16384),(2048,6144,65536),(49152,4096,4096)]
    for dims in examples:
        r=by_dims[dims]
        lines.append('| '+' × '.join(map(str,dims))+' | '+' | '.join(f'{r[c]["methods"][f]["mean_ms"]:.4f}'
                     for c in ('v5e','v6e') for f in ('native','one_level','two_level'))+' |')
    lines += ['', '## Error and implementation observations', '']
    for chip in ('v5e','v6e'):
        lines.append(f'- {chip}, worst relative L2 across the per-shape references: '+', '.join(
            f'{LABELS[f]} {summary["all"][chip][f]["worst_relative_l2"]:.6g}' for f in ('native','cubic','one_level','two_level'))+'.')
    lines += ['', 'All frozen headline comparisons in both studies pass the numerical gates. Error values are measured '
              'against exact BF16 operands using full or 128×128 sampled FP64 references across all K; they do not measure LLM prediction quality.', '',
              'The square 8191 case on v6e has only 0.0366% padded volume overhead, yet custom cubic and both Strassen paths '
              'are much slower than at 8192. The 8193 square selects 1536×1536×512 tiles and incurs 34.4% padded-volume overhead. '
              'Extra work volume alone therefore cannot explain the whole boundary cliff: layout, padding/crop materialization, '
              'selected tile geometry and scheduling are candidates for targeted diagnosis.', '',
              'The aligned rule captures 31 of 33 v6e S1 wins, but includes four losses and three inconclusive outcomes. '
              'It is useful as a first filter, not a dispatch guarantee. Two aligned counterexamples above have no padding and still lose. '
              'All 69 shapes with a dimension below 1024 have no clear S1 or S2 win on v6e.', '',
              'Google currently lists v6e at 918 BF16 TFLOP/s and 1638 GB/s HBM bandwidth, and v5e at 197 TFLOP/s and 800 GiB/s '
              '(about 859 GB/s). That is about 4.66× compute versus 1.9× bandwidth. The reduced Strassen advantage is consistent with extra additions, '
              'reconstruction and data movement becoming relatively more expensive. This is an interpretation, not a direct stall-counter attribution. '
              'Sources: [Google v6e](https://docs.cloud.google.com/tpu/docs/v6e), [Google v5e](https://docs.cloud.google.com/tpu/docs/v5e).', '',
              'Recommended focus: S1 for large aligned v6e shapes, Native fallback elsewhere, and targeted boundary/skinny-kernel work. '
              'S2 requires a demonstrated advantage over S1 before accepting its greater numerical error. Validate any shape rule on new shapes.', '',
              '![Cross-chip comparison](cross_chip.png)', '', '![Boundary examples](boundary_shapes.png)', '',
              'See comparison.json for every matched shape, candidates, confidence intervals, errors, direct S2/S1 comparisons, '
              'device-profile means, environment fingerprints and input hashes. Separate device profile samples are not pooled with complete-call timing.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(matched_shapes=168, summary=summary, transitions=transitions, best_shapes=best), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('STRASSEN_PROJECT_ROOT','.')))
    p.add_argument('--output-dir', type=Path)
    a=p.parse_args()
    out=a.output_dir or Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    main(a.root.resolve(),out.resolve())
