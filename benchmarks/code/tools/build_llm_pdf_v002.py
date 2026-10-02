"""Create an immutable, self-contained PDF of the completed 18-shape study."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from xml.sax.saxutils import escape

STUDY = 'runs/20260921-llm-large-shapes-cohort-v001'
REPORT = 'runs/20260921T025202Z-llm-mini-final-report-v001-639727/artifacts/report'
MODELS = [('Qwen/Qwen3-8B', 'Qwen3-8B', 'qwen'),
          ('mistralai/Mistral-7B-v0.3', 'Mistral-7B-v0.3', 'mistral'),
          ('google/gemma-3-12b-pt', 'Gemma3-12B text backbone', 'gemma')]
FAMILIES = ('native', 'cubic', 'strassen')
COLORS = {'native': '#3478AC', 'cubic': '#DB8B32', 'strassen': '#278C78'}

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def ordered(rows, model):
    return sorted((r for r in rows if r['model_id'] == model),
                  key=lambda r: (r['projection'] != 'gate_up_concat', r['m']))

def site(row):
    return 'Gate/up' if row['projection'] == 'gate_up_concat' else 'Down'

def make_charts(root, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = read(root / REPORT / 'comparisons.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#344454'})
    for model, _, key in MODELS:
        rows = ordered(data['complete_call_rows'], model)
        fig, ax = plt.subplots(figsize=(7.33, 2.64))
        fig.subplots_adjust(left=.085, right=.985, top=.85, bottom=.22)
        x = np.arange(6)
        maximum = max(r[f + '_call_ms'] for r in rows for f in FAMILIES)
        for i, family in enumerate(FAMILIES):
            values = [r[family + '_call_ms'] for r in rows]
            positions = x + (i - 1) * .24
            ax.bar(positions, values, width=.22, color=COLORS[family], label={'native':'Native XLA','cubic':'Tuned cubic','strassen':'Tuned Strassen'}[family], zorder=3)
            for p, v in zip(positions, values):
                ax.text(p, v + maximum * .025, (f'{v:.1f}' if v >= 10 else f'{v:.2f}'), fontsize=7.4, ha='center', va='bottom')
        for j, row in enumerate(rows):
            ax.hlines(min(row[f + '_call_ms'] for f in FAMILIES), j-.37, j+.37, color='#172D43', lw=1.1, zorder=4)
        ax.set_xticks(x, [f'{site(r)}\n{r["m"]:,} rows' for r in rows], fontsize=8)
        ax.set_ylabel('Mean latency (ms)', fontsize=9)
        ax.set_ylim(0, maximum * 1.24)
        ax.axvline(2.5, color='#CAD3DD', ls=':', lw=.8)
        ax.grid(axis='y', color='#E4E9EF', zorder=0); ax.set_axisbelow(True)
        ax.legend(loc='upper left', bbox_to_anchor=(-.01, 1.23), ncol=3, frameon=False, fontsize=9)
        with (output / (key + '.png')).open('xb') as stream:
            fig.savefig(stream, format='png', dpi=240, facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(7.33, 7.55), sharex=True)
    fig.subplots_adjust(left=.235, right=.975, top=.91, bottom=.085, hspace=.52)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=COLORS['native'],marker='o',label='Native / Strassen'),
                        Line2D([0],[0],color=COLORS['cubic'],marker='s',label='Cubic / Strassen')],
               loc='upper left', bbox_to_anchor=(.03,1.0), ncol=2, frameon=False, fontsize=9)
    for ax, (model, title, _) in zip(axes, MODELS):
        rows = ordered(data['complete_call_rows'], model)
        for j, r in enumerate(rows):
            for baseline, offset, marker in [('native',-.12,'o'),('cubic',.12,'s')]:
                lo, hi, point = r[baseline+'_ci_low'], r[baseline+'_ci_high'], r['strassen_vs_'+baseline]
                ax.hlines(j+offset,lo,hi,color=COLORS[baseline],lw=1.3)
                ax.plot([lo,hi],[j+offset]*2,ls='none',marker='|',color=COLORS[baseline],ms=5)
                ax.plot(point,j+offset,marker=marker,color=COLORS[baseline],ms=4)
        ax.set_yticks(range(6),[f'{site(r)}  {r["m"]:,}' for r in rows],fontsize=8)
        ax.set_ylim(5.5,-.5); ax.set_xlim(.74,1.19)
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=8)
        ax.axvline(1,color='#344454',ls='--',lw=.9)
        ax.grid(axis='x',color='#E4E9EF'); ax.tick_params(labelsize=8)
    axes[-1].set_xlabel('Alternative mean / Strassen mean', fontsize=9)
    with (output/'confidence.png').open('xb') as stream:
        fig.savefig(stream,format='png',dpi=240,facecolor='white')
    plt.close(fig)

def build(root, output, plot_python):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from pypdf import PdfReader
    data = read(root / REPORT / 'comparisons.json')
    audit = read(root / REPORT / 'audit.json')
    seal = read(root / REPORT / 'artifact-manifest.json')['sha256']
    for name, expected in seal.items():
        assert sha(root / REPORT / name) == expected, name
    assert audit['passed'] and not audit['issues'] and not data['failures']
    cohort = root / STUDY
    frozen = read(cohort/'frozen.json'); campaign = read(cohort/'source'/frozen['campaign_relative'])
    manifest = read(cohort/'source/configs/llm_large_shapes_v001/shapes.json')
    completion = read(cohort/'controller-completion.json')
    assert completion['status']=='completed' and completion['released'] is True
    output.mkdir(parents=True, exist_ok=False)
    assets = output/'assets'; assets.mkdir()
    subprocess.run([plot_python,str(Path(__file__).resolve()),'--charts-only','--root',str(root),'--output-dir',str(assets)], check=True)
    rows=data['complete_call_rows']; assert len(rows)==18
    cases=[]
    confirm = Path(read(cohort/'GRID-confirm-finished.json')['run'])/'artifacts/results.jsonl'
    for line in confirm.read_text().splitlines():
        r=json.loads(line)
        if r.get('event')=='case_result' and r.get('scope')=='call': cases.append(r)
    assert len(cases)==54 and all(r['status']=='ok' for r in cases)
    gate=campaign['correctness']['gate']
    metrics={f: [r['correctness']['relative_l2'] for r in cases if r['family']==f] for f in FAMILIES}
    pdf=output/'larger_llm_mm_study_v002.pdf'
    handle=pdf.open('xb')
    c=canvas.Canvas(handle,pagesize=(612,792),pageCompression=1)
    c.setTitle('Fast matrix multiplication on TPU: larger LLM-shaped study')
    c.setAuthor('Strassen MM Focus'); c.setSubject('18-shape v5e study: protocol, results, interpretation and provenance')
    navy=colors.HexColor('#172D43'); teal=colors.HexColor('#278C78'); muted=colors.HexColor('#52677A')
    pale=colors.HexColor('#F1F5F8'); line=colors.HexColor('#D9E2E9')
    styles={}
    for name,size,leading in [('body',10.2,14.4),('small',8.6,11.8),('cell',8.3,10.8),('head',8.3,10.8),('tiny',7.4,10)]:
        styles[name]=ParagraphStyle(name,fontName='Helvetica-Bold' if name=='head' else 'Helvetica',fontSize=size,
                                    leading=leading,textColor=colors.white if name=='head' else navy,alignment=TA_LEFT)
    page=0; pages=[]
    def para(text,y,style='body',x=42,width=528,gap=10):
        p=Paragraph(text,styles[style]);_,h=p.wrap(width,720)
        if y-h < 48: raise ValueError(f'Page {page} overflow at {text[:50]}: {y-h}')
        p.drawOn(c,x,y-h);return y-h-gap
    def heading(text,y):
        c.setFillColor(navy);c.setFont('Helvetica-Bold',12);c.drawString(42,y-12,text);return y-25
    def new(title,kicker):
        nonlocal page
        if page:c.showPage()
        page+=1;pages.append(title)
        c.setFillColor(navy);c.rect(0,776,612,16,fill=1,stroke=0)
        c.setFillColor(muted);c.setFont('Helvetica',8)
        c.drawString(42,752,'STRASSEN MM FOCUS  /  V5E STUDY')
        c.drawRightString(570,752,'21 SEP 2026  /  REPORT v002')
        c.setFillColor(teal);c.setFont('Helvetica-Bold',9);c.drawString(42,731,kicker.upper())
        c.setFillColor(navy);c.setFont('Helvetica-Bold',23);c.drawString(42,701,title)
        c.setStrokeColor(line);c.line(42,40,570,40)
        c.setFont('Helvetica',8);c.setFillColor(muted)
        c.drawString(42,26,'Synthetic BF16 MM | One identified v5e | Fresh-input confirmation')
        c.drawRightString(570,26,f'{page} / 11')
        c.bookmarkPage(f'page{page}');c.addOutlineEntry(title,f'page{page}',level=0)
        return 677
    def table(headers, values, widths, y, small=False):
        sty=styles['tiny' if small else 'cell']
        body=[[Paragraph(str(v),styles['head']) for v in headers]]
        body += [[Paragraph(str(v),sty) for v in row] for row in values]
        t=Table(body,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),
                              ('LINEBELOW',(0,-1),(-1,-1),.6,line)]))
        _,h=t.wrap(528,720)
        if y-h<48:raise ValueError(f'Table overflow page {page}: {y-h}')
        t.drawOn(c,42,y-h);return y-h-14
    def callout(label,text,y,height=76):
        c.setFillColor(pale);c.roundRect(42,y-height,528,height,6,fill=1,stroke=0)
        c.setFillColor(teal);c.rect(42,y-height,3,height,fill=1,stroke=0)
        para('<b>'+label+'</b><br/>'+text,y-12,x=55,width=501,gap=0)
        return y-height-16
    def ci(row,baseline):
        return f'{row["strassen_vs_"+baseline]:.3f}x<br/>[{row[baseline+"_ci_low"]:.3f}, {row[baseline+"_ci_high"]:.3f}]'

    y=new('Larger LLM-shaped matrix multiplication','Results at a glance')
    y=para('A complete record of the 18-shape comparison of default Native XLA, independently tuned cubic, and independently tuned one-level Strassen.',y)
    y=callout('Main finding','Strassen beat tuned cubic on all 18 shapes. It beat Native XLA on all tested Qwen and Mistral shapes, and on one of six Gemma shapes.',y,82)
    y=table(['Model-derived dimensions','Wins vs Native XLA','Wins vs tuned cubic'],
            [[title.replace(' text backbone',' (text)'),f'{data["classifications_by_model"][title]["native"].get("win",0)} / 6','6 / 6'] for _,title,_ in MODELS]+[['Total','13 / 18','18 / 18']], [252,138,138], y)
    y=para('A win means the pointwise paired 95% confidence interval for complete-call speedup lies entirely above 1. The results are specific to these shapes, synthetic inputs, candidates and this allocation.',y,'small')
    y=heading('Execution and review',y)
    y=para('<b>588 of 588 outcomes passed</b>: 12 smoke, 468 screening and 108 confirmation outcomes. There were no compilation, execution or numerical-gate failures. Passing the gate does not imply zero arithmetic error.',y)
    y=para('<b>Evidence audit:</b> 29,056 integrity and consistency checks passed. All code and measurement archives were committed locally. The v5e allocation was released after verified retrieval.',y)
    y=callout('Scope','These are matrix-multiplication benchmarks using dimensions from larger LLMs. No model weights, real activations, full-model inference or model quality were measured.',y,77)
    para('Reading guide: projection meanings (p. 2), experiment flow and protocol (p. 3), model results (pp. 4-6), confidence intervals (p. 7), tiles and prepared timing (p. 8), accuracy (p. 9), interpretation (p. 10), and provenance (p. 11).',y,'small')

    y=new('What do gate/up and down mean?','Workloads and dimensions')
    y=para('Inside an LLM feed-forward layer, each token starts as a vector of width <b>H</b>. Two separate projections, <b>gate</b> and <b>up</b>, each expand it to width <b>I</b>. After a nonlinear activation and elementwise combination, <b>down</b> projects the I-wide vector back to H.',y)
    y=table(['Operation','Matrix multiplication','Meaning'],[
        ['gate_up_concat','(M x H) @ (H x 2I)','Join the gate and up weight matrices side by side; compute both projections in one multiplication.'],
        ['down','(M x I) @ (I x H)','Project the combined intermediate vector back to the original hidden width.']], [105,154,269],y)
    y=heading('Qwen3-8B example',y)
    y=callout('Width of one token vector','4,096 input values -> 24,576 gate/up values -> 12,288 combined values -> 4,096 down-output values. The middle activation and elementwise combination are outside this benchmark.',y,88)
    y=para('<b>M</b> counts flattened token rows. It may represent batch x sequence positions; M alone does not specify a model context length. We use M = 2,048, 8,192 and 16,384 for every row below.',y)
    y=table(['Model','H','I','Gate/up (K, N)','Down (K, N)'],[
        ['Qwen3-8B','4,096','12,288','4,096; 24,576','12,288; 4,096'],
        ['Mistral-7B-v0.3','4,096','14,336','4,096; 28,672','14,336; 4,096'],
        ['Gemma3-12B text','3,840','15,360','3,840; 30,720','15,360; 3,840']], [136,52,62,139,139],y)
    y=para('Each product is A[M,K] @ B[K,N] = C[M,N]. Three models x two projection sites x three M values gives <b>18 shapes</b>. Gemma dimensions come from its nested text configuration.',y)
    para('The gate/up and down benchmarks are independent synthetic products: the down input is not generated by the measured gate/up result. Concatenation is a chosen MM workload; it does not assert that every official implementation fuses the projections.',y,'small')

    y=new('How the experiment was run','Frozen protocol')
    stages=[('1','Verify and freeze','Official dimensions, 18 shapes, six tiles per custom family, fixed numerical gates.'),
            ('2','Smoke: 2 groups','Small Qwen and largest Gemma gate/up shapes; all 12 outcomes passed.'),
            ('3','Screen: 108 groups','13 candidates per shape: Native 1, cubic 6, Strassen 6. All 468 outcomes passed.'),
            ('4','Freeze winners and confirm','Choose each family\'s fastest eligible call mean; fresh seed, 18 groups, 108 outcomes.'),
            ('5','Audit and release','Replay recorded results and choices, generate graphs, archive evidence, release TPU.')]
    for number,title,detail in stages:
        c.setFillColor(pale);c.roundRect(42,y-48,528,48,5,fill=1,stroke=0)
        c.setFillColor(teal);c.circle(60,y-24,10,fill=1,stroke=0)
        c.setFillColor(colors.white);c.setFont('Helvetica-Bold',9);c.drawCentredString(60,y-27,number)
        para('<b>'+title+'</b><br/>'+detail,y-8,'small',x=80,width=478,gap=0)
        if number!='5':
            c.setStrokeColor(teal);c.line(306,y-49,306,y-58);c.line(303,y-55,306,y-58);c.line(309,y-55,306,y-58)
        y-=61
    y=table(['Phase','Warmups / repeats','Input seed'],[['Smoke','2 / 5','2026092101'],['Screen','2 / 7','2026092102'],['Confirm','5 / 30','2026092103']], [176,176,176],y)
    y=para('<b>Tiles (BM, BN, BK):</b> (512,1024,512), (1024,1024,512), (1024,1024,1024), (2048,1024,512), (1024,2048,512), (2048,2048,512). Cubic and Strassen each received the same six attempts per shape. Native used default XLA without compiler-option tuning.',y,'small')
    y=para('<b>Precision and inputs:</b> BF16 inputs and Strassen pre-adds; FP32 accumulation/output; DEFAULT dot precision. NumPy PCG64 generates A ~ N(0,1)/sqrt(K), then B ~ N(0,1), in FP32 before one BF16 quantization.',y,'small')
    para('<b>Timing:</b> synchronized, seeded interleaving of algorithms. Complete-call timing includes required device padding, layout preparation, multiplication and crop. Compilation, host transfer, input generation and reference calculation are excluded. Prepared-kernel timing reuses prepared device inputs and is secondary.',y,'small')

    interpretations={
      'qwen':('All six shapes favor Strassen','Complete-call speedup over Native ranges from 1.035x to 1.154x; over tuned cubic, from 1.072x to 1.092x. Both projection sites benefit in this cohort. Larger M is not monotonically better: the down projection has its largest relative Native advantage at M=2,048.'),
      'mistral':('All six shapes favor Strassen','Complete-call speedup over Native ranges from 1.044x to 1.162x; over tuned cubic, from 1.077x to 1.103x. The pattern is consistent with Qwen on these related geometries, but this is not evidence of the same speedup in an entire Mistral model.'),
      'gemma':('Native wins five of the six comparisons','Strassen beats tuned cubic on all six shapes, but beats Native only for gate/up at M=16,384: 1.014x, with a pointwise 95% interval [1.013, 1.015]. Logical H=3,840 is preserved and custom padding is timed. Padding is a plausible factor to investigate, not an isolated causal finding.')}
    for model,title,key in MODELS:
        rs=ordered(rows,model)
        y=new(title,'Confirmed results')
        y=para('<b>'+interpretations[key][0]+'.</b> Lower bars are faster. Each group contains one selected candidate per family, measured on the same fresh inputs.',y)
        c.drawImage(str(assets/(key+'.png')),42,y-190,width=528,height=190,mask='auto')
        y-=196
        y=para('Dark horizontal marker: lowest measured mean in that group. It is not itself a significance test. Each bar summarizes 30 confirmation rounds.',y,'small',gap=10)
        vals=[[site(r)+'<br/>M='+f'{r["m"]:,}',f'{r["native_call_ms"]:.3f}',f'{r["cubic_call_ms"]:.3f}',f'{r["strassen_call_ms"]:.3f}',ci(r,'native'),ci(r,'cubic')] for r in rs]
        y=table(['Shape','Native ms','Cubic ms','Strassen ms','Native / Strassen<br/>[95% CI]','Cubic / Strassen<br/>[95% CI]'],vals,[92,53,53,56,137,137],y,small=True)
        y=para('<b>Interpretation.</b> '+interpretations[key][1],y)
        para('Ratios above 1 favor Strassen. Ratios describe MM latency only; they are not end-to-end model acceleration factors.',y,'small')

    y=new('How clear are the speed differences?','Confidence intervals')
    y=para('Each point is alternative mean / Strassen mean. A complete interval to the right of 1 favors Strassen; an interval to the left favors the alternative.',y,'small')
    c.drawImage(str(assets/'confidence.png'),42,y-544,width=528,height=544,mask='auto')
    y-=554
    para('Intervals are pointwise 95% paired bootstraps of the same 30 confirmation rounds, with 2,000 resamples. No multiplicity correction is applied. These intervals do not measure variability across allocations, shapes, seeds or real activation distributions.',y,'small')

    y=new('Tiles and prepared-input timing','Secondary measurements')
    y=para('Tiles are listed as BM x BN x BK. Choices were frozen from screening, not reselected using confirmation. Native has compiler-managed tiling. Prepared-input means exclude reusable input preparation; the complete-call comparisons on pages 4-6 remain the headline.',y,'small')
    table_rows=[]
    for model,title,key in MODELS:
        for r in ordered(rows,model):
            tile=lambda family:' x '.join(str(v) for v in r[family+'_tile'])
            table_rows.append([key.title(),('G/U' if r['projection']=='gate_up_concat' else 'Down')+' '+f'{r["m"]:,}',tile('cubic'),tile('strassen'),f'{r["native_prepared_kernel_ms"]:.3f}',f'{r["cubic_prepared_kernel_ms"]:.3f}',f'{r["strassen_prepared_kernel_ms"]:.3f}'])
    y=table(['Model','Site / M','Cubic tile','Strassen tile','Native ms','Cubic ms','Strassen ms'],table_rows,[56,72,113,113,58,58,58],y,small=True)
    para('<b>Why separate tuning matters:</b> the selected cubic and Strassen tiles differ on 3 of 18 shapes. Most choices are 2048 x 2048 x 512. These are winners within six registered candidates, not proven global optima. G/U means concatenated gate/up.',y,'small')

    y=new('Accuracy and failure checks','Numerical results')
    y=callout('No failed outcomes','All 588 scoped outcomes passed: 12 smoke + 468 screening + 108 confirmation. There were no compile errors, OOMs, execution failures or failed numerical gates in this cohort.',y,76)
    y=heading('What the error means',y)
    y=para('Relative L2 = ||C_test - C_ref||_F / ||C_ref||_F, evaluated on the sampled output submatrix. It measures the magnitude of the difference relative to the reference. It is more informative here than counting how many floating-point entries differ.',y)
    y=table(['Family','Minimum relative L2','Maximum relative L2','Gate'],[[{'native':'Native XLA','cubic':'Tuned cubic','strassen':'Tuned Strassen'}[f],f'{min(metrics[f]):.6g}',f'{max(metrics[f]):.6g}','<= 0.02'] for f in FAMILIES],[152,130,130,116],y)
    y=para('Ranges above cover the 18 complete-call confirmation results for each family. Passing this gate allows a numerical performance comparison; it does not mean the arithmetic error is zero or establish model-quality equivalence.',y,'small')
    y=heading('Fixed test and reference',y)
    y=para('The reference uses the exact BF16-quantized operands, cast to NumPy FP32. For these large matrices, it computes 128 sampled rows x 128 sampled columns, including the first and last indices, and uses every K term. It is not a full-output error norm or an exact-arithmetic reference.',y)
    y=para('The whole device output is checked for finite values. Eligibility also requires relative L2 <= 0.02 and max absolute error <= 0.001 + 0.05 x max|reference|. An eligible timed candidate must satisfy the gates and all registered repeats in both scopes.',y)
    y=heading('Memory and validation scope',y)
    y=para('Kernel VMEM limit: 48 MiB. Estimated live-device budget: 8 GiB. The worst preflight estimate across all possible selected custom tile pairs was 4.806 GiB; the largest logical FP32 output was 1.875 GiB. These are estimates, not measured runtime/compiler peaks.',y,'small')
    para('This study uses one Gaussian input distribution. Cancellation-sensitive inputs, actual model weights and activations, downstream quality and higher-precision references remain separate questions.',y,'small')

    y=new('What the study establishes','Interpretation and next questions')
    y=heading('Supported by this experiment',y)
    y=para('<b>1. Tuned Strassen consistently improves on our tuned cubic implementation.</b> All 18 complete-call comparisons favor Strassen under the frozen candidate budget and numerical gates.',y)
    y=para('<b>2. Native XLA remains a necessary baseline.</b> Strassen beats it on 13 shapes, while Native wins five Gemma shapes. Beating a custom cubic kernel alone does not establish a practical win over the compiler baseline.',y)
    y=para('<b>3. Model name and size are not enough to predict performance.</b> Exact projection geometry and M matter. Gemma has a different hidden width and padding behavior, and its results differ from Qwen and Mistral. Causal attribution would require a controlled ablation.',y)
    y=heading('Limits of the claim',y)
    for txt in [
      'One v5e allocation was used throughout this study. Timings were not pooled with the previous campaign, and this does not establish repeatability across machines or v6e.',
      'Confirmation uses fresh synthetic inputs on the same tuned shapes. It is not held-out-shape validation; only one screening and one confirmation seed were used.',
      'The comparison uses optimized custom kernels and one native default. It is not a new vanilla-implementation ablation or a search over native compiler options.',
      'The feed-forward activation, elementwise gating, residuals, attention, full-model throughput and quality are excluded. The earlier Gemma1B adapter qualification issue is not resolved by these shape-only results.']:
        y=para(txt,y,'small')
    y=heading('Useful next experiments',y)
    y=para('Repeat the frozen protocol on v6e as a separate cohort; isolate padding/alignment effects on Gemma; and test real weights and activation distributions before making application-level claims. Additional seeds and allocations would quantify repeatability.',y)
    para('<b>Deferred note:</b> two-level Strassen remains an explicitly separate experiment. The present results use one Strassen level per tile.',y,'small')

    y=new('Sources and reproducibility','Evidence and recorded decisions')
    identity=data['environment_identity']
    y=table(['Item','Recorded value'],[
      ['Cohort','20260921-llm-large-shapes-cohort-v001'],
      ['Allocation',escape(frozen['allocation_id'])],
      ['Hardware','Single TPU v5 lite (v5e); serial device timings'],
      ['Pinned software','JAX 0.7.2; jaxlib 0.7.2; libtpu 0.0.21.1'],
      ['Host / boot',escape(identity['hostname'])+'<br/>'+escape(identity['boot_id'])],
      ['Completion','2026-09-21 03:05:53 UTC; allocation release verified'],
      ['Source Git commit',frozen['baseline_commit']],
      ['Audit','29,056 checks passed; no evidence inconsistencies']], [133,395],y,small=True)
    y=heading('Verified model configurations',y)
    for model in manifest['models']:
        name=escape(model['model_id']);url=escape(model['config_url'], {'"':'&quot;'})
        y=para(f'<b>{name}</b> - <link href="{url}" color="#3478AC">immutable official configuration</link><br/>Revision: {model["revision"]}<br/>Config SHA256: {model["config_sha256"]}',y,'tiny',gap=8)
    y=heading('Repository evidence map',y)
    y=para('All paths below are relative to <b>Strassen_MM_Focus/</b>. The PDF is a summary; immutable journals preserve the individual samples, failures (none here), source/config hashes, selections and phase receipts.',y,'small')
    y=para('<b>Protocol:</b> plans/llm_large_shapes_v001/ and configs/llm_large_shapes_v001/<br/><b>Device evidence:</b> runs/20260921-llm-large-shapes-cohort-v001/<br/><b>Audited report and source figures:</b> runs/20260921T025202Z-llm-mini-final-report-v001-639727/artifacts/<br/><b>Readable review:</b> reports/llm_large_shapes_v001/REVIEW.md',y,'tiny')
    para('The audit replays recorded gates, timing statistics, candidate selection and bootstrap intervals, and verifies hashes and identity. It does not independently recompute matrix products. Local controller-fixture v002 failed on macOS path canonicalization; preserved v003 passed without changes to the device benchmark or numerical thresholds.',y,'small')
    assert page==11
    c.save();handle.close()
    reader=PdfReader(str(pdf));assert len(reader.pages)==11
    extracted=[p.extract_text() for p in reader.pages]
    assert all(len(t)>200 for t in extracted)
    save(output/'provenance.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'pdf':pdf.name,'pdf_sha256':sha(pdf),
        'pages':pages,'source_report_sha256':sha(root/REPORT/'comparisons.json'),'audit_sha256':sha(root/REPORT/'audit.json'),
        'cohort_source_archive_sha256':frozen['archive_sha256'],'builder_sha256':sha(__file__),
        'accuracy_relative_l2_by_family':{f:{'min':min(v),'max':max(v),'median':statistics.median(v)} for f,v in metrics.items()},
        'scope':'Compilation of saved, audited study; no new device measurements'})
    with (output/'extracted_text.txt').open('x') as stream:
        stream.write('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(extracted)))
    print(json.dumps({'pdf':str(pdf),'pages':11,'sha256':sha(pdf)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--plot-python')
    p.add_argument('--charts-only',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    output=a.output_dir or Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    if a.charts_only:make_charts(root,output)
    else:build(root,output,a.plot_python or str(root/'.venv-plots-v001/bin/python'))

if __name__=='__main__':main()
