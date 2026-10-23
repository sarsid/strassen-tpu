"""Offline, source-verified figures for the three completed LLM studies."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
BATCHES = [1, 4, 8, 16, 32, 64]
ARMS = ['native', 'native_tuned', 'cubic_tuned', 'strassen_tuned']
STYLE = {
    'native': dict(label='Default Native', color='#8793A2', marker='x', ls='--', lw=1.7),
    'native_tuned': dict(label='Tuned Native', color='#2F5F91', marker='D', ls=':', lw=1.7),
    'cubic_tuned': dict(label='Tuned Cubic', color='#CE8528', marker='s', ls='-.', lw=1.7),
    'strassen_tuned': dict(label='Tuned Strassen S1', color='#007F6B', marker='o', ls='-', lw=2.8),
}
META = [('qwen3_32b', 'Qwen3-32B', 64), ('gemma4_31b', 'Gemma4-31B', 60),
        ('qwen3_14b', 'Qwen3-14B', 40)]
STACK_NOTE = '* Qwen3-32B B64 uses a newer compiler stack; it is not joined to B32. Within-batch algorithm comparisons are matched.'


def load_verified():
    manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
    records = {s['id']: s for s in manifest['sources']}
    models = []
    for ident, title, layers in META:
        source = records[ident]
        data = (ROOT / source['snapshot_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source['sha256']
        d = json.loads(data)
        rows = []
        for r in d.get('batches', d.get('rows', [])):
            b = r['batch']
            resident = r.get('resident_ms', r.get('resident', {}).get('aggregate_ms'))
            offloaded = r.get('offloaded_prefill', {})
            means = offloaded.get('mean_ms', r.get('prefill_mean_ms'))
            samples = offloaded.get('samples_ms', r.get('prefill_samples_ms'))
            quality = r['natural_quality']; quality = quality.get('results', quality)
            for arm in ARMS:
                assert resident[arm] > 0 and len(samples[arm]) == 3
                assert math.isclose(statistics.mean(samples[arm]), means[arm], rel_tol=1e-12)
                assert quality[arm]['passes'] and quality[arm]['finite']
                assert quality[arm]['positions'] == 4 * b * 1023
            ppl = {a: quality[a]['perplexity'] for a in ARMS}
            rows.append({
                'batch': b, 'sequence': 1024, 'input_tokens': b * 1024,
                'resident_ms': resident,
                'resident_speedup_vs_tuned_native': {a: resident['native_tuned'] / resident[a] for a in ARMS},
                'resident_time_saved_percent_vs_tuned_native': {a: 100 * (1 - resident[a] / resident['native_tuned']) for a in ARMS},
                'offloaded_mean_seconds': {a: means[a] / 1000 for a in ARMS},
                'offloaded_samples_seconds': {a: [x / 1000 for x in samples[a]] for a in ARMS},
                'offloaded_time_saved_percent_vs_tuned_native': {a: 100 * (1 - means[a] / means['native_tuned']) for a in ARMS},
                'natural_perplexity': ppl,
                'natural_perplexity_change_percent_vs_native': {a: 100 * (ppl[a] / ppl['native'] - 1) for a in ARMS},
                'natural_positions': quality['native']['positions'],
                'stack': 'JAX 0.7.2 / libtpu 0.0.21.1' if ident == 'qwen3_32b' and b != 64 else 'JAX 0.11.2 / libtpu 0.0.48',
            })
        rows.sort(key=lambda r: r['batch'])
        assert [r['batch'] for r in rows] == BATCHES
        models.append({'id': ident, 'model': title, 'decoder_layers': layers,
                       'source': source, 'rows': rows})
    return models, manifest


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 14,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'text.color': '#172737', 'axes.labelcolor': '#243648',
        'xtick.color': '#465568', 'ytick.color': '#465568',
        'axes.edgecolor': '#D4DCE3', 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'svg.hashsalt': 'llm-comparison-20261018-v001',
    })


def legend(fig, y):
    handles = [Line2D([], [], markersize=6, **STYLE[a]) for a in ARMS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y), ncol=4,
               frameon=False, handlelength=2.5, columnspacing=2.3)


def header(fig, title, subtitle, legend_y):
    fig.text(.055, .96, title, fontsize=22, weight='bold', va='top')
    fig.text(.055, .905, subtitle, fontsize=11.5, color='#536273', va='top')
    legend(fig, legend_y)


def axis(ax, model, ylabel=None, xlabel=True):
    ax.grid(axis='y', color='#E8EDF1', lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.3, 5.45)
    ax.set_xticks(np.arange(6))
    labels = [str(b) + ('*' if b == 64 and model['id'] == 'qwen3_32b' else '') for b in BATCHES]
    ax.set_xticklabels(labels if xlabel else [])
    ax.tick_params(length=0, pad=7)
    if xlabel: ax.set_xlabel('Batch size (sequences)', labelpad=9)
    if ylabel: ax.set_ylabel(ylabel, labelpad=9)
    if model['id'] == 'qwen3_32b':
        ax.axvspan(4.64, 5.4, color='#EEF1F5', zorder=0)


def series(ax, model, values, arm, alpha=1):
    kw = {**STYLE[arm], 'markersize': 5.8, 'alpha': alpha, 'zorder': 5 if arm == 'strassen_tuned' else 3}
    if arm == 'native': kw['markersize'] = 8
    if arm == 'cubic_tuned': kw['markerfacecolor'] = 'white'
    if model['id'] == 'qwen3_32b':
        ax.plot(np.arange(5), values[:5], **kw)
        kw.pop('label'); kw['ls'] = 'none'
        ax.plot([5], [values[5]], **kw)
    else:
        ax.plot(np.arange(6), values, **kw)


def save(fig, name, output):
    for fmt in ['png', 'pdf', 'svg']:
        metadata = {'Creator': 'plot_results.py; immutable benchmark JSON'}
        if fmt == 'pdf': metadata.update(CreationDate=None, ModDate=None)
        if fmt == 'svg': metadata['Date'] = None
        fig.savefig(output / f'{name}.{fmt}', dpi=200, metadata=metadata)
    plt.close(fig)


def transformer(models, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.1))
    fig.subplots_adjust(left=.06, right=.98, top=.74, bottom=.23, wspace=.23)
    header(fig, 'Strassen improves transformer computation across all three models',
           'Speedup over Tuned Native  |  Higher is better  |  Sequence 1,024  |  BF16 output  |  One TPU v6e', .85)
    for ax, model in zip(axes, models):
        axis(ax, model, 'Speedup vs Tuned Native (×)' if ax == axes[0] else None)
        ax.set_ylim(.73, 1.53); ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}×'))
        ax.axhline(1, color='#657585', lw=1, zorder=1)
        ax.set_title(f"{model['model']}  ·  {model['decoder_layers']} layers", loc='left', pad=14, weight='bold')
        for arm in ARMS:
            vals = [r['resident_speedup_vs_tuned_native'][arm] for r in model['rows']]
            series(ax, model, vals, arm)
            if arm == 'strassen_tuned':
                for i, y in enumerate(vals):
                    ax.annotate(f'{y:.2f}×', (i, y), xytext=(0, 9), textcoords='offset points',
                                ha='center', fontsize=9, color=STYLE[arm]['color'], weight='bold')
        reductions = [r['resident_time_saved_percent_vs_tuned_native']['strassen_tuned'] for r in model['rows']]
        ax.text(.02, .03, f'{min(reductions):.1f}–{max(reductions):.1f}% less compute time',
                transform=ax.transAxes, fontsize=10, weight='bold', color=STYLE['strassen_tuned']['color'])
    fig.text(.06, .105, 'Metric: sum of separately warmed decoder-layer means. Includes layer computation; excludes weight preparation/transfers, embedding/head and compilation.', fontsize=9.2, color='#536273')
    fig.text(.06, .073, 'This is not uninterrupted whole-model latency. Each layer has 2 warmups and 5 measured samples per arm.', fontsize=9.2, color='#536273')
    fig.text(.06, .039, STACK_NOTE, fontsize=9, color='#536273')
    save(fig, '01_transformer_speedup', output)


def full_runtime(models, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw={'height_ratios': [1.05, 1]})
    fig.subplots_adjust(left=.065, right=.98, top=.755, bottom=.18, wspace=.24, hspace=.36)
    header(fig, 'Full offloaded prefill shows much smaller gains',
           'Complete cached-disk-offloaded passes, including preparation and recurring transfers  |  Mean of 3 measured passes per arm', .86)
    for j, model in enumerate(models):
        top, low = axes[:, j]
        axis(top, model, 'Full prefill time (seconds)' if j == 0 else None, xlabel=False)
        top.set_ylim(0, 280); top.yaxis.set_major_locator(MultipleLocator(60))
        top.set_title(model['model'], loc='left', pad=14, weight='bold')
        for arm in ARMS:
            vals = [r['offloaded_mean_seconds'][arm] for r in model['rows']]
            series(top, model, vals, arm)
        last = model['rows'][-1]['offloaded_mean_seconds']
        top.text(.02, .035, f"B64: Tuned Native {last['native_tuned']:.2f} s\n        Strassen S1   {last['strassen_tuned']:.2f} s",
                 transform=top.transAxes, fontsize=10, color='#34485C', linespacing=1.6)
        axis(low, model, 'S1 time reduction vs Tuned Native (%)' if j == 0 else None)
        low.set_ylim(-.45, 1.85); low.yaxis.set_major_locator(MultipleLocator(.5))
        low.axhline(0, color='#657585', lw=1)
        low.set_title('Strassen savings, expanded scale', loc='left', fontsize=10.5, pad=8)
        vals = [r['offloaded_time_saved_percent_vs_tuned_native']['strassen_tuned'] for r in model['rows']]
        series(low, model, vals, 'strassen_tuned')
        for i, y in enumerate(vals):
            low.annotate(f'{y:+.2f}%', (i, y), xytext=(0, 9), textcoords='offset points',
                         ha='center', fontsize=9, color=STYLE['strassen_tuned']['color'])
    fig.text(.065, .095, 'Scope: cached reads, common packing/preparation, H2D, embeddings, all decoder layers and last-position logits; no decode or initial downloads.', fontsize=9.1, color='#536273')
    fig.text(.065, .066, 'Common preparation includes Strassen packing for every arm. Small differences from 3 passes are descriptive, not established deployment gains.', fontsize=9.1, color='#536273')
    fig.text(.065, .037, STACK_NOTE, fontsize=9, color='#536273')
    save(fig, '02_full_offloaded_runtime', output)


def perplexity(models, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw={'height_ratios': [1.05, 1]})
    fig.subplots_adjust(left=.065, right=.98, top=.755, bottom=.18, wspace=.24, hspace=.36)
    header(fig, 'Perplexity remains close across algorithms',
           'WikiText-2 test  |  Four contiguous segments per batch  |  Upper curves nearly overlap; lower panels reveal the differences', .86)
    for j, model in enumerate(models):
        top, low = axes[:, j]
        axis(top, model, 'Perplexity' if j == 0 else None, xlabel=False)
        top.set_ylim(0, 11.2); top.yaxis.set_major_locator(MultipleLocator(2))
        top.set_title(model['model'], loc='left', pad=14, weight='bold')
        for arm in ARMS:
            series(top, model, [r['natural_perplexity'][arm] for r in model['rows']], arm)
        maxdelta = max(abs(r['natural_perplexity_change_percent_vs_native']['strassen_tuned']) for r in model['rows'])
        top.text(.02, .035, f'Max |S1 perplexity change|: {maxdelta:.3f}%', transform=top.transAxes,
                 fontsize=10, color=STYLE['strassen_tuned']['color'], weight='bold')
        axis(low, model, 'Perplexity change vs Native (%)' if j == 0 else None)
        low.set_ylim(-.23, .23); low.yaxis.set_major_locator(MultipleLocator(.1))
        low.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.1f}' if abs(v) > 1e-9 else '0.0'))
        low.set_title('Within-batch difference, expanded scale', loc='left', fontsize=10.5, pad=8)
        for arm in ARMS:
            series(low, model, [r['natural_perplexity_change_percent_vs_native'][arm] for r in model['rows']], arm)
    fig.text(.065, .095, 'Relative change = 100 × (algorithm perplexity / matched Native perplexity − 1). Positive means higher perplexity; negative means lower.', fontsize=9.1, color='#536273')
    fig.text(.065, .066, 'Corpus spans vary by batch and tokenizers vary by model: compare algorithms within each workload. Close perplexity does not mean identical predictions.', fontsize=9.1, color='#536273')
    fig.text(.065, .037, STACK_NOTE, fontsize=9, color='#536273')
    save(fig, '03_perplexity', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    models, inputs = load_verified(); style()
    (args.output.parent / 'plot_data.json').write_text(json.dumps({
        'schema': 'three_llm_figures_v001', 'sequence': 1024, 'architecture': 'single TPU v6e',
        'output_dtype': 'BF16', 'arms': ARMS, 'sources': inputs, 'models': models,
        'scope': {'resident': 'Sum of separately warmed decoder-layer means; not whole-model wall time.',
                  'full_runtime': 'Serialized cached-disk-offloaded prefill, not resident serving or decode.'}}, indent=2) + '\n')
    transformer(models, args.output); full_runtime(models, args.output); perplexity(models, args.output)
    print(json.dumps({'models': len(models), 'workloads': sum(len(m['rows']) for m in models),
                      'algorithm_results': sum(len(m['rows']) for m in models) * len(ARMS),
                      'figures': 3, 'formats': ['png', 'pdf', 'svg'],
                      'matplotlib': matplotlib.__version__, 'numpy': np.__version__}))


if __name__ == '__main__':
    main()
