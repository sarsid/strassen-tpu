"""Audit both tuning phases, keep all failures, and graph independent confirmation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def key(row):return row['group_id'],row['seed'],row['arm_id']
def number(x):return 'unavailable' if x is None else f'{x:.3f}'


def audit_phase(cohort,stage,cfg):
    receipt=read(cohort/('LEVEL-tune-'+stage+'-finished.json'));run=Path(receipt['run'])
    assert run.resolve().is_relative_to(cohort/'phases')
    assert sha(run/'completion.json')==receipt['completion_sha256']
    completed=read(run/'completion.json')
    assert completed['status']=='completed' and completed['remote_may_still_be_running'] is False
    checks=0
    for name,expected in read(run/'artifact-manifest.json')['sha256'].items():
        assert sha(run/name)==expected,name;checks+=1
    out=run/'artifacts';rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    groups=read(out/'planned_cases.json');summary=read(out/'summary.json');env=read(out/'environment.json')
    assert summary['completed'] and len(summary['completed_groups'])==len(groups)
    assert env['qualified_single_v5e']
    cases=[r for r in rows if r['event']=='case_result'];expected=32 if stage=='screen' else 12
    assert len(cases)==expected and dict(Counter(c['status'] for c in cases))==summary['case_status_counts']
    by={};samples={};fingerprints={}
    for r in rows:
        if r['event']=='case_start':
            k=(r['shape_id'],r['seed'])
            if k in fingerprints:assert fingerprints[k]==r['input_fingerprint']
            fingerprints[k]=r['input_fingerprint'];checks+=1
    for c in cases:
        k=key(c);assert k not in by;by[k]=c
        ss=sorted([s for s in rows if s['event']=='sample' and key(s)==k],key=lambda s:s['round']);samples[k]=ss
        assert len(ss)==c['timing']['sample_count']
        if ss:
            assert len(ss)==cfg['timing'][stage]['repeats']
            assert [s['round'] for s in ss]==list(range(len(ss)))
            values=np.array([s['elapsed_ms'] for s in ss]);assert np.all(np.isfinite(values)) and np.all(values>0)
            np.testing.assert_allclose(values.mean(),c['timing']['mean_ms'],rtol=1e-13)
        metric=c.get('correctness')
        if metric:
            gate=cfg['correctness']['gate']
            passed=metric['finite'] and metric['relative_l2']<=gate['relative_l2_max'] and metric['max_abs_error']<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*metric['max_abs_reference']
            assert bool(passed)==metric['pass']
            assert c['eligible_for_speedup_claim']==(c['status']=='ok')
        checks+=1
    for group in groups:
        for inp in group['inputs']:
            k=(group['group_id'],inp['seed'])
            a,b=(samples[k+(arm,)] for arm in cfg['arms'])
            if a and b:
                assert all({sa['position'],sb['position']}=={0,1} for sa,sb in zip(a,b))
                assert Counter(s['position'] for s in a)=={0:len(a)//2,1:len(a)//2};checks+=1
    return dict(run=run,groups=groups,summary=summary,env=env,cases=cases,by=by,samples=samples,fingerprints=fingerprints,checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    cohort=p.parse_args().cohort.resolve();out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    frozen=read(cohort/'frozen.json');assert sha(cohort/'source.tar')==frozen['archive_sha256'];checks=1
    for name,expected in frozen['source_sha256'].items():assert sha(cohort/'source'/name)==expected;checks+=1
    cfg=read(cohort/'source/configs/two_level_tune_v001/campaign.json')
    shapes=read(cohort/'source/configs/two_level_tune_v001/shapes.json')['shapes']
    screen=audit_phase(cohort,'screen',cfg);confirm=audit_phase(cohort,'confirm',cfg)
    checks+=screen['checks']+confirm['checks']
    assert screen['env']['identity']==confirm['env']['identity']
    assert screen['env']['identity']['allocation_id']==frozen['allocation_id'];checks+=1
    selection=read(screen['run']/'artifacts/selections.json')
    assert selection==read(confirm['run']/'artifacts/selection_used.json')
    assert selection['campaign_sha256']==sha(cohort/'source/configs/two_level_tune_v001/campaign.json')
    assert set(screen['fingerprints']).isdisjoint(confirm['fingerprints']);checks+=1
    for shape in shapes:
        for arm in cfg['arms']:
            eligible=[c for c in screen['cases'] if c['shape_id']==shape['id'] and c['arm_id']==arm and c['status']=='ok' and c['eligible_for_speedup_claim'] and c['timing']['sample_count']==10]
            best=min(eligible,key=lambda c:(c['timing']['mean_ms'],tuple(c['kernel_metadata']['tile_bm_bn_bk'])))
            chosen=selection['selected'][shape['id']][arm]
            assert chosen['tile']==best['kernel_metadata']['tile_bm_bn_bk'] and chosen['screen_group_id']==best['group_id']
            for c in confirm['cases']:
                if c['shape_id']==shape['id'] and c['arm_id']==arm and c.get('kernel_metadata'):
                    assert c['kernel_metadata']['tile_bm_bn_bk']==chosen['tile']
            checks+=1
    paired=[];aggregate=[];rng=np.random.default_rng(2026092151)
    for g in confirm['groups']:
        aa=[];bb=[];eligible=True
        for inp in g['inputs']:
            k=(g['group_id'],inp['seed']);one,two=(confirm['by'][k+(arm,)] for arm in cfg['arms'])
            a,b=(np.array([s['elapsed_ms'] for s in confirm['samples'][k+(arm,)]]) for arm in cfg['arms'])
            valid=one['eligible_for_speedup_claim'] and two['eligible_for_speedup_claim'];eligible=eligible and valid
            item=dict(shape_id=g['shape']['id'],seed=inp['seed'],eligible=valid,one_level_ms=one['timing'].get('mean_ms'),two_level_ms=two['timing'].get('mean_ms'))
            if len(a)==len(b)==30:
                ix=rng.integers(0,30,(4000,30));ratios=a[ix].mean(1)/b[ix].mean(1)
                item.update(speedup_two_over_one=float(a.mean()/b.mean()),ci95=np.quantile(ratios,[.025,.975]).tolist());aa.append(a);bb.append(b)
            paired.append(item)
        item=dict(shape_id=g['shape']['id'],eligible=eligible,selected=selection['selected'][g['shape']['id']])
        if len(aa)==len(bb)==3:
            a,b=np.array(aa),np.array(bb);s=rng.integers(0,3,(4000,3,1));r=rng.integers(0,30,(4000,3,30))
            ratios=a[s,r].mean((1,2))/b[s,r].mean((1,2))
            item.update(one_level_ms=float(a.mean()),two_level_ms=float(b.mean()),speedup_two_over_one=float(a.mean()/b.mean()),
                        latency_reduction_percent=float(100*(1-b.mean()/a.mean())),ci95_hierarchical=np.quantile(ratios,[.025,.975]).tolist())
        aggregate.append(item)
    result=dict(allocation=frozen['allocation_id'],cohort=str(cohort),screen_run=str(screen['run']),confirm_run=str(confirm['run']),
                selection=selection,screen_cases=screen['cases'],confirmation_cases=confirm['cases'],paired=paired,aggregate=aggregate,
                audit=dict(passed=True,checks=checks,screen_statuses=screen['summary']['case_status_counts'],confirmation_statuses=confirm['summary']['case_status_counts']))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Small two-level Strassen tuning experiment','',
        'Two shapes, eight tile choices per depth, then independent confirmation on three fresh Gaussian seeds. Kernels are unchanged. Both depths are selected separately from the same search grid.', '',
        '## Confirmation: selected configurations','', '| Shape | One-level tile | Two-level tile | One-level ms | Two-level ms | One / two [95% interval] | All seeds eligible |', '|---|---|---|---:|---:|---:|---|']
    for c in aggregate:
        ratio='unavailable'
        if 'speedup_two_over_one' in c:ratio=f'{c["speedup_two_over_one"]:.3f} [{c["ci95_hierarchical"][0]:.3f}, {c["ci95_hierarchical"][1]:.3f}]'
        lines.append(f'| {c["shape_id"]} | {c["selected"]["one_level"]["tile"]} | {c["selected"]["two_level"]["tile"]} | {number(c.get("one_level_ms"))} | {number(c.get("two_level_ms"))} | {ratio} | {c["eligible"]} |')
    lines+=['','Tiles are (BM,BN,BK); latency is the mean of 90 synchronized calls per arm, balanced over three seeds. Higher one/two ratios favor two levels. Intervals resample seeds and paired timing rounds; with only three seeds and one allocation they do not establish broad reproducibility.','',
        '## Every confirmation seed','', '| Shape | Seed | One-level ms | Two-level ms | One / two | Eligible |','|---|---:|---:|---:|---:|---|']
    for c in paired:lines.append(f'| {c["shape_id"]} | {c["seed"]} | {number(c["one_level_ms"])} | {number(c["two_level_ms"])} | {number(c.get("speedup_two_over_one"))} | {c["eligible"]} |')
    lines+=['','## Numerical error during confirmation','', '| Shape | Depth | Seed | Relative L2 | Status |','|---|---|---:|---:|---|']
    for c in confirm['cases']:lines.append(f'| {c["shape_id"]} | {c["arm_id"]} | {c["seed"]} | {(c.get("correctness") or {}).get("relative_l2")} | {c["status"]} |')
    lines+=['','## All screening candidates','', 'Screening means use ten timed rounds after three warmups. These data select candidates; they are not the confirmation evidence. All failures remain visible.','', '| Shape | Tile BM,BN,BK | Depth | Mean ms | Relative L2 | Status |','|---|---|---|---:|---:|---|']
    group_tiles={g['group_id']:g['tile'] for g in screen['groups']}
    for c in screen['cases']:lines.append(f'| {c["shape_id"]} | {group_tiles[c["group_id"]]} | {c["arm_id"]} | {number(c["timing"].get("mean_ms"))} | {(c.get("correctness") or {}).get("relative_l2")} | {c["status"]} |')
    lines+=['','## Scope and evidence','',
        '- Synthetic shapes: 12288 x 12288 x 12288 and Qwen gate/up 16384 x 4096 x 24576 (M x K x N). No full-model inference claim.',
        '- BF16 inputs and input combinations, DEFAULT dot precision, FP32 accumulation/output. Full-call timing includes padding/crop and excludes compilation/host transfers.',
        '- Confirmation uses five warmups and 30 alternating rounds for each fresh seed. Selection is unchanged after screening.',
        '- Numerical gates: whole-output finiteness, relative L2 <= 0.02, and max absolute error <= 0.001 + 0.05 * maximum absolute reference. Reference uses exact quantized operands, every K term, and 128 x 128 sampled output positions.',
        '- Passing these gates does not imply equal accuracy or unchanged downstream LLM quality. Error and performance must be assessed together.',
        '- This is an eight-choice local tile search with a fixed scratch-buffer schedule, not globally optimal tuning. No AlphaTensor comparison was run.',
        f'- Allocation: `{frozen["allocation_id"]}`. Same identity verified across screening and confirmation.',
        f'- Frozen revision: `{frozen["baseline_commit"]}`. {checks} source/artifact/statistics/selection checks passed.',
        f'- Screening outcomes: `{screen["summary"]["case_status_counts"]}`. Confirmation outcomes: `{confirm["summary"]["case_status_counts"]}`.']
    for c in screen['cases']+confirm['cases']:
        if c['status']!='ok':lines.append(f'- Failure `{c["group_id"]}` / `{c["arm_id"]}`: {c.get("error_message")}')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors=['#3478AC','#D18B32'];labels=['12288 square','Qwen gate/up']
    fig,ax=plt.subplots(figsize=(8.5,5.4));fig.subplots_adjust(top=.80,bottom=.24,left=.12,right=.97)
    maximum=max(c.get(k,0) for c in aggregate for k in ('one_level_ms','two_level_ms')) or 1
    for i,arm in enumerate(cfg['arms']):
        values=[c.get(arm+'_ms',np.nan) for c in aggregate]
        bars=ax.bar(np.arange(2)+(i-.5)*.3,values,.27,color=colors[i],label=arm.replace('_',' ').title(),zorder=3)
        for bar,c,value in zip(bars,aggregate,values):
            if not c['eligible']:bar.set_hatch('///')
            if np.isfinite(value):ax.text(bar.get_x()+bar.get_width()/2,value+maximum*.025,f'{value:.2f}',ha='center')
    for i,c in enumerate(aggregate):
        if c['eligible'] and 'one_level_ms' in c:ax.hlines(min(c['one_level_ms'],c['two_level_ms']),i-.34,i+.34,linestyle='--',color='#222222',linewidth=1,zorder=4)
    ax.set_xticks(range(2),labels);ax.set_xlim(-.6,1.6);ax.set_ylim(0,maximum*1.2);ax.set_ylabel('Mean latency (ms); lower is faster')
    ax.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.18));ax.grid(axis='y',alpha=.2,zorder=0);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One versus two Strassen levels after tile tuning',fontsize=16)
    fig.text(.12,.075,'Eight tile choices per depth; winners selected before confirmation.\nThree fresh seeds, 30 rounds each, one TPU v5e. BF16 inputs, FP32 output.\nDashed lines mark the faster eligible result. Synthetic matrix products.',fontsize=10)
    fig.savefig(out/'confirmation.png',dpi=180,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,9));fig.subplots_adjust(top=.87,bottom=.16,hspace=.65,left=.08,right=.98)
    ticklabels=[' x '.join(map(str,t)) for t in cfg['tiles']]
    for ax,shape,label in zip(axes,shapes,labels):
        relevant=[c for c in screen['cases'] if c['shape_id']==shape['id']]
        lookup={(tuple(group_tiles[c['group_id']]),c['arm_id']):c for c in relevant}
        high=max(c['timing'].get('mean_ms',0) for c in relevant if c['timing'].get('mean_ms') is not None)
        for i,arm in enumerate(cfg['arms']):
            entries=[lookup[tuple(t),arm] for t in cfg['tiles']];values=[c['timing'].get('mean_ms',np.nan) for c in entries]
            bars=ax.bar(np.arange(8)+(i-.5)*.34,values,.3,color=colors[i],zorder=3)
            for j,(bar,c) in enumerate(zip(bars,entries)):
                if c['status']!='ok':
                    bar.set_hatch('///');ax.text(j+(i-.5)*.34,high*.025,c['status'],rotation=90,ha='center',fontsize=8)
        ax.set_title(label,loc='left');ax.set_xticks(range(8),ticklabels,rotation=30,ha='right',fontsize=9)
        ax.set_xlim(-.6,7.6);ax.set_ylim(0,high*1.15);ax.set_ylabel('Screen mean (ms)');ax.grid(axis='y',alpha=.2,zorder=0);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('All eight tile choices — screening only',fontsize=16)
    fig.legend(handles=[Patch(color=colors[0],label='One level'),Patch(color=colors[1],label='Two levels')],loc='upper left',bbox_to_anchor=(.08,.965),ncol=2,frameon=False)
    fig.text(.08,.035,'Tiles are BM x BN x BK. Ten rounds per executable arm. Failed candidates have no latency bar.\nUse the separate fresh-seed confirmation for performance conclusions.',fontsize=10)
    fig.savefig(out/'screening.png',dpi=180,facecolor='white');plt.close(fig)
    print(json.dumps(dict(output=str(out),audit=result['audit'],aggregate=aggregate),indent=2))


if __name__=='__main__':main()
