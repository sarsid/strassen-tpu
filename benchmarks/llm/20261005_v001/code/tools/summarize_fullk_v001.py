"""Compact tables and separate precision plots from the sealed full-K report."""
import argparse
import hashlib
import json
import os
from pathlib import Path


FAMILIES = [('Current S1', 'current', 1, '#2475b8'),
            ('Current S2', 'current', 2, '#d57822'),
            ('Parent S1', 'parent_s1', 1, '#29845a'),
            ('Blocked cubic', 'parent_cubic', 0, '#8e69a7')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    data = json.loads(args.report.read_text())
    assert data['completed'], 'Only summarize a sealed, fully attempted experiment'
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    rows = []
    lines = ['# Full-contraction results', '',
             'BF16 inputs and pre-additions, FP32 accumulation. Output precision is separated below.',
             'All candidates were fixed before measurement. Each has three inputs × 30 paired rounds.',
             'Fastest observed choices below are descriptive; they have not received an independent confirmation run.',
             'Complete-call timing includes output conversion, excludes compilation and transfer. Errors use sampled all-K FP64 references.', '',
             '| Shape (M × K × N), output | Fastest Native ms | Existing S1 ms | Full-K S1 ms | Existing S2 ms | Full-K S2 ms | Full-K parent S1 ms |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g in data['results']:
        s, methods = g['shape'], g['methods']
        def fastest(impl, depth=0, full=False):
            pool = [m for m in methods.values() if m['eligible'] and m['candidate']['implementation'] == impl
                    and m['candidate']['depth'] == depth and (not full or m['candidate']['tile'][2] == s['k'])]
            return min(pool, key=lambda m: m['mean_ms']) if pool else None
        def ms(m):
            return '—' if m is None or not m['eligible'] else f'{m["mean_ms"]:.5f}'
        lines.append('| ' + ' × '.join(str(s[k]) for k in ('m', 'k', 'n')) + ', ' + s['output_dtype'] + ' | '
                     + ' | '.join(ms(m) for m in (fastest('native'), methods.get('S1_incumbent'),
                         fastest('current', 1, True), methods.get('S2_incumbent'), fastest('current', 2, True),
                         fastest('parent_s1', 1, True))) + ' |')
    lines += ['', 'Existing settings are historical tuned choices for the three grid geometries. The parent geometry uses a fixed shorter-K control, not a historically tuned incumbent.', '',
             '| Shape (M × K × N), output | Family | Best shorter-K ms (BM, BN, BK) | Best full-K ms (BM, BN, BK) | Shorter / full | Full-K / Native speedup | Full-K rel. L2 % |',
             '|---|---|---|---|---:|---:|---:|']
    for dtype in ('float32', 'bfloat16'):
        groups = [g for g in data['results'] if g['shape']['output_dtype'] == dtype]
        fig, axes = plt.subplots(len(groups), 2, figsize=(12, 3.2 * len(groups)), squeeze=False)
        for idx, g in enumerate(groups):
            s, methods = g['shape'], g['methods']
            native = min((m for m in methods.values() if m['candidate']['implementation'] == 'native' and m['eligible']), key=lambda m: m['mean_ms'])
            default = methods['Native_default']
            for label, impl, depth, color in FAMILIES:
                pool = [m for m in methods.values() if m['candidate'].get('role') == 'panel_curve'
                        and m['candidate']['implementation'] == impl and m['candidate']['depth'] == depth and m['eligible']]
                shorter = [m for m in pool if m['candidate']['tile'][2] < s['k']]
                full = [m for m in pool if m['candidate']['tile'][2] == s['k']]
                if not shorter and not full:
                    continue
                short = min(shorter, key=lambda m: m['mean_ms']) if shorter else None
                best = min(full, key=lambda m: m['mean_ms']) if full else None
                row = dict(shape=s, family=label, shorter=short, full=best,
                           incumbent=methods.get(f'S{depth}_incumbent') if impl == 'current' else None,
                           native=native, native_default=default)
                rows.append(row)
                def desc(m):
                    return 'infeasible' if m is None else f'{m["mean_ms"]:.5f} {tuple(m["candidate"]["tile"])}'
                def num(v):
                    return '—' if v is None else f'{v:.4f}'
                lines.append('| ' + ' × '.join(str(s[k]) for k in ('m', 'k', 'n')) + ', ' + dtype
                             + f' | {label} | {desc(short)} | {desc(best)} | '
                             + num(short['mean_ms'] / best['mean_ms'] if short and best else None) + ' | '
                             + num(native['mean_ms'] / best['mean_ms'] if best else None) + ' | '
                             + num(100 * best['errors']['relative_l2'] if best else None) + ' |')
            for j, tile in enumerate(s['output_tiles']):
                ax = axes[idx, j]
                for label, impl, depth, color in FAMILIES:
                    curve = [m for m in methods.values() if m['candidate'].get('role') == 'panel_curve'
                             and m['candidate']['implementation'] == impl and m['candidate']['depth'] == depth
                             and m['candidate']['tile'][:2] == tile and m['eligible']]
                    curve.sort(key=lambda m: m['candidate']['tile'][2])
                    if curve:
                        ax.plot([m['candidate']['tile'][2] for m in curve], [m['mean_ms'] for m in curve],
                                marker='o', color=color, label=label)
                ax.axhline(native['mean_ms'], color='#333333', linestyle='--', label='Fastest observed Native')
                if default['eligible']:
                    ax.axhline(default['mean_ms'], color='#888888', linestyle=':', label='Default Native')
                ax.set_title(f'{s["m"]} × {s["k"]} × {s["n"]}; BM × BN = {tile[0]} × {tile[1]}')
                ax.set_xticks(s['panels'], [str(k) + ('\n(full K)' if k == s['k'] else '') for k in s['panels']])
                ax.set_xlabel('Contraction panel BK'); ax.set_ylabel('Mean call time (ms)')
                ax.grid(alpha=.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f'Full contraction on v6e — {dtype} output', fontsize=16)
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .08 if len(groups) == 1 else .045, 1, .94 if len(groups) == 1 else .97))
        fig.savefig(out / f'contraction_{dtype}.png', bbox_inches='tight')
        fig.savefig(out / f'contraction_{dtype}.pdf', bbox_inches='tight')
        plt.close(fig)
    lines += ['', 'A full-K/Native speedup above 1 means full K is faster. The Native denominator is its fastest observed eligible setting.',
              'Best-shorter versus best-full may change output tiles. For the isolated BK effect, use the fixed-output-tile curves and paired intervals in the full report.',
              'Plots omit infeasible or numerically ineligible points; failures remain in the full report.', '',
              'Case status counts: ' + str(data['case_status_counts']) + '.',
              'No LLM prediction accuracy or serving performance is measured.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(dict(source=str(args.report),
        source_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        case_status_counts=data['case_status_counts'], rows=rows), indent=2) + '\n')
    print(json.dumps(dict(completed=True, groups=len(data['results']), rows=len(rows),
                          case_status_counts=data['case_status_counts'])))


if __name__ == '__main__':
    main()
