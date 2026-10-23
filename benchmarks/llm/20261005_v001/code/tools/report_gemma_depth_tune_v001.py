"""Audit archived Gemma tuning, replay raw samples, and report paired confirmation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from strassen_mm.benchmark_gemma_depth_tune_v001 import make_groups, select_candidates


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['group_id'],r['seed'],r['arm_id']


def audit(cohort,stage,cfg,shapes):
    receipt=read(cohort/f'GEMMA-tune-{stage}-finished.json');run=Path(receipt['run']);checks=0
    assert run.resolve().is_relative_to(cohort/'phases')
    assert receipt['status']=='completed' and sha(run/'completion.json')==receipt['completion_sha256']
    assert read(run/'completion.json')['remote_may_still_be_running'] is False
    for name,value in read(run/'artifact-manifest.json')['sha256'].items():assert sha(run/name)==value;checks+=1
    out=run/'artifacts';env=read(out/'environment.json');groups=read(out/'planned_cases.json');summary=read(out/'summary.json')
    assert env['qualified_single_v5e'] and summary['completed']
    selection=read(out/'selection_used.json')['selected'] if stage=='confirm' else None
    assert groups==make_groups(cfg,shapes,stage,selection);checks+=1
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    cases=[r for r in rows if r['event']=='case_result'];samples={};fingerprints={}
    expected={(g['group_id'],i['seed'],a['arm_id']) for g in groups for i in g['inputs'] for a in g['arms']}
    assert len(cases)==len(expected) and {key(r) for r in cases}==expected
    assert dict(Counter(c['status'] for c in cases))==summary['case_status_counts'];checks+=1
    for r in rows:
        if r['event']=='case_start':
            if r['seed'] in fingerprints:assert fingerprints[r['seed']]==r['input_fingerprint']
            fingerprints[r['seed']]=r['input_fingerprint'];checks+=1
    for c in cases:
        raw=sorted([r for r in rows if r['event']=='sample' and key(r)==key(c)],key=lambda r:r['round'])
        assert len(raw)==c['timing']['sample_count']
        if raw:
            assert [r['round'] for r in raw]==list(range(cfg['timing'][stage]['repeats']))
            arr=np.array([r['elapsed_ms'] for r in raw]);assert np.isfinite(arr).all() and (arr>0).all()
            np.testing.assert_allclose(arr.mean(),c['timing']['mean_ms'],rtol=1e-13)
        samples[key(c)]=raw
        met=c.get('correctness')
        if met:
            gate=cfg['correctness']['gate']
            ok=met['finite'] and met['relative_l2']<=gate['relative_l2_max'] and met['max_abs_error']<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*met['max_abs_reference']
            assert bool(ok)==met['pass']
            assert met['all_k_used']==3840 and met['reference_input_dtype']=='quantized_bf16'
            assert met['reference_accumulation_dtype']=='float32'
        if c.get('kernel_metadata'):
            meta=c['kernel_metadata'];assert meta['shape_mkn']==[16384,3840,30720]
            assert (meta['input_dtype'],meta['accumulation_dtype'],meta['output_dtype'],meta['dot_precision'])==('bfloat16','float32','float32','DEFAULT')
            arm=next(a for g in groups if g['group_id']==c['group_id'] for a in g['arms'] if a['arm_id']==c['arm_id'])
            assert meta['tile_bm_bn_bk']==arm['tile'] and meta['compiler_options']==arm['compiler_options']
        assert c['eligible_for_speedup_claim']==(c['status']=='ok');checks+=1
    # Each recorded round visits each executable once, with rotating order.
    for g in groups:
        for inp in g['inputs']:
            active=[samples[(g['group_id'],inp['seed'],a['arm_id'])] for a in g['arms']]
            active=[x for x in active if x]
            for rid in range(cfg['timing'][stage]['repeats']):
                assert sorted(x[rid]['position'] for x in active)==list(range(len(active)));checks+=1
    return dict(run=str(run),env=env,groups=groups,cases=cases,samples=samples,fingerprints=fingerprints,checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--stage',choices=['screen','both'],default='both')
    args=p.parse_args();cohort=args.cohort.resolve();out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    frozen=read(cohort/'frozen.json');assert sha(cohort/'source.tar')==frozen['archive_sha256'];checks=1
    for name,value in frozen['source_sha256'].items():assert sha(cohort/'source'/name)==value;checks+=1
    cfg_path=cohort/'source/configs/gemma_depth_tune_v001/campaign.json';cfg=read(cfg_path)
    shapes=read(cfg_path.parent/'shapes.json')['shapes'];screen=audit(cohort,'screen',cfg,shapes);checks+=screen['checks']
    selection=read(Path(screen['run'])/'artifacts/selections.json')
    assert selection['campaign_sha256']==sha(cfg_path) and selection['allocation_id']==frozen['allocation_id']
    assert selection['selected']==select_candidates(screen['cases'],cfg,shapes);checks+=1
    assert screen['env']['identity']['allocation_id']==frozen['allocation_id']
    result=dict(stage=args.stage,selection=selection,screen_statuses=dict(Counter(r['status'] for r in screen['cases'])),
                screen_cases=screen['cases'],screen_run=screen['run'],allocation=frozen['allocation_id'])
    if args.stage=='screen':
        result.update(audit_passed=True,checks=checks)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('stage','audit_passed','checks','screen_statuses','selection')}));return
    confirm=audit(cohort,'confirm',cfg,shapes);checks+=confirm['checks']
    assert screen['env']['identity']==confirm['env']['identity'];checks+=1
    assert selection==read(Path(confirm['run'])/'artifacts/selection_used.json')
    assert set(screen['fingerprints']).isdisjoint(confirm['fingerprints']);checks+=1
    g=confirm['groups'][0];summary=[];arrays={};by={key(c):c for c in confirm['cases']}
    for arm in g['arms']:
        kk=[(g['group_id'],inp['seed'],arm['arm_id']) for inp in g['inputs']]
        rows=[by[k] for k in kk]
        values=[[s['elapsed_ms'] for s in confirm['samples'][k]] for k in kk]
        good=all(r['eligible_for_speedup_claim'] and r['timing']['sample_count']==30 for r in rows)
        item=dict(candidate=arm,eligible=good,per_seed=[dict(seed=r['seed'],mean_ms=r['timing'].get('mean_ms'),relative_l2=(r.get('correctness') or {}).get('relative_l2'),status=r['status']) for r in rows])
        if all(len(v)==30 for v in values):
            arrays[arm['arm_id']]=np.array(values)
            errors=[r['correctness']['relative_l2'] for r in rows]
            item.update(mean_ms=float(np.mean(values)),sample_count=90,relative_l2_mean=float(np.mean(errors)),relative_l2_range=[min(errors),max(errors)],kernel_metadata=rows[0]['kernel_metadata'])
        summary.append(item)
    pairs=[];rng=np.random.default_rng(2026092171)
    for i,a in enumerate(summary):
        for b in summary[i+1:]:
            aid,bid=a['candidate']['arm_id'],b['candidate']['arm_id']
            if aid not in arrays or bid not in arrays:continue
            aa,bb=arrays[aid],arrays[bid]
            ratios=[]
            for _ in range(4000):
                seeds=rng.integers(0,3,3);rounds=rng.integers(0,30,(3,30))
                av=aa[seeds[:,None],rounds];bv=bb[seeds[:,None],rounds]
                ratios.append(av.mean()/bv.mean())
            pairs.append(dict(reference=aid,candidate=bid,eligible=a['eligible'] and b['eligible'],
                speedup=float(aa.mean()/bb.mean()),latency_reduction_pct=float(100*(1-bb.mean()/aa.mean())),
                ci95_hierarchical=np.quantile(ratios,[.025,.975]).tolist(),
                per_seed_speedup=(aa.mean(axis=1)/bb.mean(axis=1)).tolist()))
    result.update(confirmation_run=confirm['run'],confirmation_statuses=dict(Counter(r['status'] for r in confirm['cases'])),
                  confirmation_cases=confirm['cases'],summary=summary,pairs=pairs,audit_passed=True,checks=checks,
                  interpretation='One v5e allocation; independently selected fixed candidates; 3 fresh Gaussian seeds. Higher Strassen error remains explicit. No global tuning or model-quality claim.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Large Gemma: independently tuned Native and Strassen','',
           'Shape (16384 x 3840) @ (3840 x 30720), Gemma 3 12B gate/up dimensions. Synthetic MM; no model inference.',
           '', '| Candidate | Mean ms | Relative L2 | Samples | Eligible |', '|---|---:|---:|---:|---|']
    for s in summary:lines.append(f'| {s["candidate"]["candidate_id"]} | {s.get("mean_ms",float("nan")):.4f} | {s.get("relative_l2_mean",float("nan")):.3g} | {s.get("sample_count",0)} | {s["eligible"]} |')
    lines += ['', '## Paired comparisons', '', '| Reference | Candidate | Speedup | Lower latency | 95% paired hierarchical interval | Eligible |', '|---|---|---:|---:|---|---|']
    for r in pairs:lines.append(f'| {r["reference"]} | {r["candidate"]} | {r["speedup"]:.4f}x | {r["latency_reduction_pct"]:.2f}% | {r["ci95_hierarchical"][0]:.4f}–{r["ci95_hierarchical"][1]:.4f} | {r["eligible"]} |')
    lines += ['', 'Native candidates: default and per-compile scoped VMEM 32/48/64 MiB. Native tiles remain compiler-managed. Eight tiles per Strassen depth. Select by eligible screening mean; choices remain fixed in confirmation. If a non-default Native wins screening, default is retained as a fourth confirmation control. Small compiler-option differences may be noise.', '',
              'Both phases use the same verified allocation and pinned software. Every confirmed candidate has three fresh inputs with 30 rotating rounds each. Paired intervals resample input seeds and rounds together, and describe uncertainty within this allocation only.', '',
              'BF16 inputs and Strassen pre-additions; FP32 accumulation/output; DEFAULT dot precision. Device padding and cropping are included. Host transfer, compilation and correctness checks are excluded. All K terms enter the sampled host FP32 reference; whole-output finiteness is checked. Relative L2 <= 0.02 and maximum-error gates are fixed. Passing does not imply Native-equivalent accuracy or unchanged model quality.', '',
              f'Screening outcomes: {result["screen_statuses"]}. Confirmation outcomes: {result["confirmation_statuses"]}. Audit passed {checks} source, artifact, plan, timing, identity and selection checks.', '', '## Selected configurations', '']
    for s in summary:
        meta=s.get('kernel_metadata',{})
        lines.append(f'- {s["candidate"]["candidate_id"]}: tile={s["candidate"]["tile"]}; compiler options={s["candidate"]["compiler_options"]}; padded (M,K,N)={meta.get("padded_shape_mkn")}.')
    lines += ['', '## Evidence', '', f'- Screen: `{screen["run"]}`', f'- Confirmation: `{confirm["run"]}`', f'- Frozen source: `{cohort / "frozen.json"}`']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted=[s for s in summary if 'mean_ms' in s]
    labels=[]
    for s in plotted:
        c=s['candidate'];family=c['family']
        labels.append(('Native selected\n'+('default' if not c['compiler_options'] else str(next(iter(c['compiler_options'].values()))//1024)+' MiB')) if family=='native' and s is summary[0] else ('Native default\ncontrol' if family=='native' else 'Strassen\n'+('1 level' if family=='one_level' else '2 levels')))
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained');vals=[s['mean_ms'] for s in plotted]
    bars=ax.bar(labels,vals,color=['#64748b','#2563eb','#0d9488','#94a3b8'][:len(vals)],width=.6)
    ax.bar_label(bars,labels=[f'{s["mean_ms"]:.2f} ms'+('' if s['eligible'] else '\nFAILED GATE') for s in plotted],padding=6)
    good=[s['mean_ms'] for s in plotted if s['eligible']]
    if good:ax.axhline(min(good),color='#0d9488',linestyle='--',linewidth=1)
    ax.set_ylim(0,max(vals)*1.23);ax.set_ylabel('Mean device-call latency (ms); lower is better')
    ax.set_title('Large Gemma: independent tuning, fresh-input confirmation\n(16384 × 3840) @ (3840 × 30720)',pad=15)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.yaxis.grid(alpha=.2)
    fig.supxlabel('One v5e allocation · 3 inputs × 30 rounds per candidate · padding included\nBF16 inputs / FP32 output; Strassen has higher numerical error.',fontsize=10)
    fig.savefig(out/'confirmation.png',dpi=170);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('audit_passed','checks','screen_statuses','confirmation_statuses','summary','pairs')}))


if __name__=='__main__':main()
