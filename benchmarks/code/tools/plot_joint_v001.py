"""Separate-precision scientific plots of frozen joint-tuner finalists."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();data=json.loads(a.results.read_text())
    if not data['completed']:
        raise ValueError('Only completed joint study results')
    out=a.output_dir or Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(parents=True,exist_ok=False)
    roles=[f's{depth}_{mode}_{panel}' for depth in (1,2) for mode in ('products','outputs') for panel in ('short','full')]
    labels=[f'S{depth}\n{mode}\n{panel} K' for depth in (1,2) for mode in ('products','outputs') for panel in ('short','full')]
    plotted={}
    for dtype in ('float32','bfloat16'):
        rows=[r for r in data['results'] if r['shape']['output_dtype']==dtype]
        values=np.full((len(rows),len(roles)),np.nan);notes={}
        for i,r in enumerate(rows):
            for j,role in enumerate(roles):
                method=r['methods'].get(role,{})
                c=method.get('comparisons',{}).get('native',{})
                if c.get('eligible'):
                    values[i,j]=c['speedup']
                    notes[i,j]=c
        fig,ax=plt.subplots(figsize=(13,7.5),constrained_layout=True)
        cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('#eeeeee')
        image=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(vmin=.5,vcenter=1.,vmax=1.2),aspect='auto')
        ax.set_xticks(range(len(roles)),labels,fontsize=10)
        ax.set_yticks(range(len(rows)),[' × '.join(str(r['shape'][d]) for d in ('m','k','n')) for r in rows],fontsize=10)
        ax.set_ylabel('Matrix shape M × K × N')
        ax.set_title(('FP32' if dtype=='float32' else 'BF16')+' output: confirmed speedup over tuned Native',pad=20,fontsize=15)
        for (i,j),value in np.ndenumerate(values):
            if not np.isfinite(value):
                text='ineligible'
            else:
                c=notes[i,j];suffix='*' if c['ci95'][0]>1 else ''
                text=f'{value:.3f}×{suffix}'
            ax.text(j,i,text,ha='center',va='center',fontsize=10,color='black')
        ax.axvline(3.5,color='white',linewidth=3)
        fig.colorbar(image,ax=ax,label='Native latency / candidate latency',shrink=.85,extend='both')
        fig.supxlabel('Products: seven retained outer products. Outputs: direct FP32 output or four FP32 quadrants.\n'
            'Each column uses its frozen screening winner. * Pointwise paired 95% interval above 1; no multiplicity correction.\n'
            'BF16 inputs/pre-adds, FP32 accumulation; output conversion included. Six development shapes, three fresh seeds × 30 rounds.',fontsize=9)
        for suffix in ('png','pdf'):
            fig.savefig(out/f'joint_{dtype}.{suffix}',dpi=180)
        plt.close(fig)
        plotted[dtype]=dict(roles=roles,shapes=[r['shape'] for r in rows],speedups=values.tolist())
    (out/'plot_data.json').write_text(json.dumps(dict(input=str(a.results),input_sha256=hashlib.sha256(a.results.read_bytes()).hexdigest(),data=plotted),indent=2)+'\n')
    print(json.dumps(dict(completed=True,output=str(out))))


if __name__=='__main__':
    main()
