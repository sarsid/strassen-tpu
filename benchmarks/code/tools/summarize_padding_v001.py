"""Compact boundary-probe comparison and exportable latency figure."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.input.read_text())
    assert data['completed'] and len(data['results'])==13
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    rows=data['results'];arms=['S1_padded','S1_native_fringe','S2_padded','S2_native_fringe']
    summary={}
    for arm in arms:
        summary[arm]={kind:dict(Counter(r['methods'][arm]['classification'] for r in rows if r['shape']['kind']==kind)) for kind in ['boundary','regular']}
    lines=['# Boundary handling: results and costs','',
        '13 shapes completed on one v6e: 10 boundary cases and 3 regular controls. Existing S1/S2 are the primary controls; their kernels and interior tiles are unchanged. No tuning or winner selection occurred in this probe.',
        'Every shape has three fresh BF16 Gaussian inputs and 30 paired timing rounds per input. FP32 output. Timings include all slicing, padding, multiplication, partial sums, concatenation and cropping; compilation and transfers excluded.','',
        '| Policy | Boundary wins / losses / inconclusive | Regular wins / losses / inconclusive |',
        '|---|---|---|']
    for arm in arms:
        def counts(kind):return ' / '.join(str(summary[arm][kind].get(k,0)) for k in ['win','loss','inconclusive'])
        lines.append(f'| {arm} | {counts("boundary")} | {counts("regular")} |')
    lines+=['','Wins/losses use pointwise paired 95% intervals without multiplicity correction. Small statistically detectable changes need not be useful.','',
        '| M × K × N | S1 ms | S1 edges speedup | S1 Native fringe speedup | S2 ms | S2 edges speedup | S2 Native fringe speedup |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['methods'];shape=' × '.join(str(row['shape'][d]) for d in ['m','k','n'])
        fields=[]
        for depth in [1,2]:
            fields.append(f'{m[f"S{depth}"]["mean_ms"]:.4f}')
            fields.extend(f'{m[f"S{depth}"+suffix]["comparison"]["speedup"]:.3f}×' for suffix in ['_padded','_native_fringe'])
        lines.append('| '+shape+' | '+' | '.join(fields)+' |')
    lines+=['','Speedup above 1 means faster than the corresponding current-depth control.','',
        '| Method | Worst relative L2 | Worst sampled absolute error | All 39 input cases eligible? |',
        '|---|---:|---:|---|']
    for arm in ['Native','S1','S1_padded','S1_native_fringe','S2','S2_padded','S2_native_fringe']:
        methods=[r['methods'][arm] for r in rows]
        assert sum(len(m['cases']) for m in methods)==39
        lines.append(f'| {arm} | {100*max(m["errors"]["relative_l2"] for m in methods):.4f}% | {max(m["errors"]["max_abs_error"] for m in methods):.6g} | {all(m["eligible"] for m in methods)} |')
    lines+=['','Errors use the same exact BF16 inputs against 128×128 sampled FP64 references over all K, including both sides of output splits; finiteness checks cover full outputs. They are not full-matrix maximum-error bounds or LLM accuracy measurements.',
        'Native fringes perform some multiplication conventionally, so lower error can reflect less Strassen arithmetic. Regular shapes take the exact incumbent fast path. The smaller-edge implementation uses separate calls and intermediate arrays; no fused boundary loads or fused K correction were tested.',
        'One allocation and a small development set do not establish a general dispatch rule. Full-K tuning remains separate.','',
        '![Paired speedups](boundary_speedups.png)']
    fig,ax=plt.subplots(figsize=(11,8))
    colors=['#D89023','#2171B5','#B94848','#368352']
    labels=['S1 smaller edges','S1 + Native fringe','S2 smaller edges','S2 + Native fringe']
    y=np.arange(len(rows))
    for i,arm in enumerate(arms):
        v=[r['methods'][arm]['comparison'] for r in rows]
        x=np.array([r['speedup'] for r in v]);lo=np.array([r['ci95'][0] for r in v]);hi=np.array([r['ci95'][1] for r in v])
        ax.errorbar(x,y+(i-1.5)*.16,xerr=np.maximum(0,np.array([x-lo,hi-x])),fmt='o',ms=4,capsize=2,color=colors[i],label=labels[i],lw=1)
    ax.axvline(1,color='#555555',lw=1);ax.axhspan(9.5,12.5,color='#eef3f7',zorder=-1)
    ax.set_yticks(y,[' × '.join(str(r['shape'][d]) for d in ['m','k','n']) for r in rows],fontsize=9)
    ax.invert_yaxis();ax.set_xlabel('Speedup vs unchanged same-depth control (greater than 1 is faster)')
    ax.set_title('Boundary handling on v6e: complete-call latency',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
    fig.text(.02,.01,'Bars: pointwise paired 95% intervals. Shaded rows: regular shapes. BF16 inputs, FP32 output.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'boundary_speedups.png',dpi=180);fig.savefig(out/'boundary_speedups.pdf');plt.close(fig)
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(completed=True,summary=summary,
        input_path=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
