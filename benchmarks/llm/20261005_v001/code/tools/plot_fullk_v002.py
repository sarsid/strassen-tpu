"""Plot sealed fresh-confirmation intervals separately for each output dtype."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.report.read_text());assert data['completed']
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    lines=['# Twelve-shape full-contraction study','',
           'Separate FP32-output and BF16-output comparisons. All inputs/preadds are BF16; accumulation and reconstruction are FP32.',
           'Values below are fresh confirmation after screening: three inputs × 30 paired rounds. Native and the shorter/full-K choices are frozen before confirmation.',
           'Full-K speedup above 1 means faster. Intervals are pointwise paired hierarchical 95%, conditional on screening; no multiplicity adjustment or unseen-shape selector validation.','']
    counts={};error_summary={}
    for dtype in ('float32','bfloat16'):
        rows=[r for r in data['results'] if r['shape']['output_dtype']==dtype]
        fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
        labels=[' × '.join(str(r['shape'][d]) for d in ('m','k','n')) for r in rows]
        for depth,ax in enumerate(axes,1):
            bounds=[]
            for offset,baseline,color,label in [(-.14,f'short_s{depth}','#2677b7','vs tuned shorter panels'),(.14,'native','#bd6a28','vs tuned Native')]:
                for y,r in enumerate(rows):
                    c=r['methods'].get(f'full_s{depth}',{}).get('comparisons',{}).get(baseline)
                    if not c or not c['eligible']:continue
                    v=c['speedup'];lo,hi=c['ci95'];bounds.extend([lo,hi])
                    ax.errorbar(v,y+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,
                                capsize=3,label=label if y==0 else None)
            ax.axvline(1,color='#666666',linestyle='--',linewidth=1)
            ax.set_xscale('log',base=2)
            ax.set_xlim(min(.5,min(bounds)*.9) if bounds else .5,max(2,max(bounds)*1.1) if bounds else 2)
            ax.set_xticks([.25,.5,.75,1,1.25,1.5,2,3,4])
            ax.set_xlim(min(.5,min(bounds)*.9) if bounds else .5,max(2,max(bounds)*1.1) if bounds else 2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}×'))
            ax.set_yticks(range(len(rows)),labels);ax.set_xlabel('Confirmed full-K speedup')
            ax.set_title(f'Strassen {depth}');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=9)
        axes[0].invert_yaxis();fig.suptitle(f'Full contraction on v6e — {dtype} output',fontsize=17)
        fig.tight_layout(rect=(0,0,1,.95))
        for ext in ('png','pdf'):fig.savefig(out/f'confirmed_{dtype}.{ext}',bbox_inches='tight')
        plt.close(fig)
        lines+=['## '+dtype+' output','',
            '| M × K × N | Native ms | S1 short → full ms | S2 short → full ms | Full S1 / S2 relative L2 % |',
            '|---|---:|---|---|---|']
        for r in rows:
            m=r['methods']
            def val(role):
                x=m.get(role,{}).get('mean_ms');return '—' if x is None else f'{x:.4f}'
            def err(role):
                x=m.get(role,{}).get('errors',{}).get('relative_l2');return '—' if x is None else f'{100*x:.4f}'
            lines.append('| '+' × '.join(str(r['shape'][d]) for d in ('m','k','n'))+' | '+val('native')+' | '
                +val('short_s1')+' → '+val('full_s1')+' | '+val('short_s2')+' → '+val('full_s2')+' | '+err('full_s1')+' / '+err('full_s2')+' |')
        counts[dtype]={str(depth):dict(Counter(r['methods'][f'full_s{depth}']['candidate']['buffers'] for r in rows if f'full_s{depth}' in r['methods'])) for depth in (1,2)}
        lines+=['','Full-K selected input-buffer counts: '+str(counts[dtype])+'.','',
                'Confirmed comparison counts: '+str(data['summary'][dtype])+'.','']
        metrics=('relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error')
        error_summary[dtype]={}
        lines+=['### Numerical cost','',
            'Each value is the worst recorded metric across these 12 shapes and three confirmation inputs per shape. Absolute errors refer to the study\'s Gaussian A/√K, B inputs; percentile columns are maxima of the per-input sampled percentiles, not a pooled percentile.',
            '', '| Method | Relative L2 % | Max absolute | RMSE | Mean absolute | Median absolute | P99 absolute | Normwise |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for role in ('native','short_s1','full_s1','short_s2','full_s2'):
            worst={}
            for metric in metrics:
                values=[r['methods'].get(role,{}).get('errors',{}).get(metric) for r in rows]
                values=[v for v in values if v is not None]
                worst[metric]=max(values) if values else None
            error_summary[dtype][role]=worst
            formatted=['—' if worst[k] is None else f'{worst[k]*(100 if k=="relative_l2" else 1):.6g}' for k in metrics]
            lines.append('| '+role+' | '+' | '.join(formatted)+' |')
        lines.append('')
    lines+=['## Scope','', 'Case status counts: '+str(data['case_status_counts'])+'.',
            'The complete report preserves existing controls, Native default, matched cubic, all error metrics, all infeasible candidates and raw source hashes.',
            'The two output dtypes are separate experiments and are not pooled. This study measures synthetic MM, not LLM prediction quality.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(completed=True,source=str(args.report),
        source_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),summary=data['summary'],
        buffers=counts,error_summary=error_summary,case_status_counts=data['case_status_counts']),indent=2)+'\n')
    print(json.dumps(dict(completed=True,precision_groups=len(data['results']))))


if __name__=='__main__':main()
