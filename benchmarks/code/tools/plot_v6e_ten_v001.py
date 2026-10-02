"""Scientific comparison of preselected historical and fresh-confirmed speedups."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
r=json.loads(a.summary.read_text())['rows'];fig,axes=plt.subplots(1,2,figsize=(13.8,7.7),sharey=True)
bounds=[1.0]
for row in r:
 for d in ('1','2'):
  old=row['v5e']['one_level_comparisons']['native'] if d=='1' else row['v5e']['comparisons']['native']
  bounds.extend(old['ci95']);bounds.extend(row['methods'][d]['vs_tuned']['ci95'])
limits=(min(bounds)-.025,max(bounds)+.025)
for d,ax in zip(('1','2'),axes):
 for i,row in enumerate(r):
  h=row['v5e'];old=h['one_level_comparisons']['native'] if d=='1' else h['comparisons']['native'];new=row['methods'][d]['vs_tuned']
  for offset,c,color,label in [(-.14,old,'#8996a4','Historical v5e'),(.14,new,'#1464b5','Current v6e')]:
   x=c['speedup'];lo,hi=c['ci95'];ax.errorbar(x,i+offset,xerr=np.array([[x-lo],[hi-x]]),fmt='o',color=color,markersize=5,capsize=3,elinewidth=1.3,label=label if i==0 else None)
 ax.axvline(1,color='#be6a29',ls='--',lw=1);ax.set_title('Strassen '+d,loc='left',fontsize=14,weight='bold')
 ax.grid(axis='x',color='#e3e8ee',lw=.8);ax.set_axisbelow(True);ax.set_xlabel('Speedup vs tuned Native (×)',fontsize=10)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_xlim(*limits)
 ax.legend(loc='lower right',frameon=False,fontsize=9)
axes[0].set_yticks(range(len(r)),[' × '.join(map(str,x['shape_mkn']))for x in r],fontsize=10)
axes[0].invert_yaxis();axes[0].set_ylabel('M × K × N',fontsize=11)
fig.suptitle('Historical v5e wins retested on v6e',x=.08,y=.98,ha='left',fontsize=19,weight='bold')
fig.text(.08,.934,'Higher is faster · 95% pointwise paired intervals · choices frozen before fresh confirmation',fontsize=11,color='#435363')
fig.text(.08,.025,'BF16 operands/preadds; FP32 output. Synthetic MM shapes. Different hardware/compiler cohorts; not a hardware-only A/B test.',fontsize=9,color='#435363')
fig.tight_layout(rect=(0,.06,1,.91));fig.savefig(a.output_dir/'comparison.png',dpi=180,facecolor='white');fig.savefig(a.output_dir/'comparison.svg',facecolor='white')
print(a.output_dir/'comparison.png')
