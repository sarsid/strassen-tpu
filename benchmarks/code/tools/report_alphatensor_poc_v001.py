"""Audit saved proof-of-concept evidence and produce a compact comparison."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    args=p.parse_args();cohort=args.cohort.resolve()
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    receipt=read(cohort/'ALPHA-poc-finished.json');run=Path(receipt['run'])
    assert run.resolve().is_relative_to(cohort/'phases')
    assert sha(run/'completion.json')==receipt['completion_sha256']
    completion=read(run/'completion.json');assert completion['remote_may_still_be_running'] is False
    assert completion['status']=='completed'
    manifest=read(run/'artifact-manifest.json')['sha256'];checks=0
    for name,expected in manifest.items():assert sha(run/name)==expected,name;checks+=1
    frozen=read(cohort/'frozen.json');assert sha(cohort/'source.tar')==frozen['archive_sha256']
    for name,expected in frozen['source_sha256'].items():assert sha(cohort/'source'/name)==expected,name;checks+=1
    rows=[json.loads(line) for line in (run/'artifacts/results.jsonl').read_text().splitlines()]
    cases=[r for r in rows if r['event']=='case_result'];assert len(cases)==24
    config=read(cohort/'source/configs/alphatensor_poc_v001/campaign.json')
    shapes=read(cohort/'source/configs/alphatensor_poc_v001/shapes.json')['shapes']
    summary=read(run/'artifacts/summary.json');assert summary['completed'] and len(summary['completed_groups'])==4
    assert dict(Counter(r['status'] for r in cases))==summary['case_status_counts']
    env=read(run/'artifacts/environment.json')
    assert env['qualified_single_v5e'] and env['identity']['allocation_id']==frozen['allocation_id']
    by={};sample_map={}
    for r in cases:
        key=(r['shape_id'],r['arm_id']);assert key not in by;by[key]=r
        samples=[s for s in rows if s['event']=='sample' and s['shape_id']==key[0] and s['arm_id']==key[1]]
        sample_map[key]=samples
        assert len(samples)==r['timing']['sample_count']
        if samples:
            assert len(samples)==30 and sorted(s['round'] for s in samples)==list(range(30))
            np.testing.assert_allclose(np.mean([s['elapsed_ms'] for s in samples]),r['timing']['mean_ms'],rtol=1e-13)
        metric=r.get('correctness')
        if metric:
            gate=config['correctness']['gate']
            passed=metric['finite'] and metric['relative_l2']<=gate['relative_l2_max'] and metric['max_abs_error']<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*metric['max_abs_reference']
            assert bool(passed)==metric['pass']
            assert r['eligible_for_speedup_claim']==(r['status']=='ok')
        checks+=1
    comparisons=[]
    for shape in shapes:
        name=shape['id']
        for reference,candidate in [('native_xla','our_strassen'),('native_xla','alpha_fp32_recombine'),
                                     ('our_strassen','alpha_fp32_recombine'),('native_bf16','alpha_upstream_bf16')]:
            a=sorted(sample_map[name,reference],key=lambda s:s['round'])
            b=sorted(sample_map[name,candidate],key=lambda s:s['round'])
            if len(a)!=30 or len(b)!=30:continue
            a=np.array([s['elapsed_ms'] for s in a]);b=np.array([s['elapsed_ms'] for s in b])
            indices=np.random.default_rng(2026092120).integers(0,30,(2000,30))
            ratios=a[indices].mean(axis=1)/b[indices].mean(axis=1)
            comparisons.append(dict(shape_id=name,reference=reference,candidate=candidate,
                ratio_of_means=float(a.mean()/b.mean()),ci95=np.quantile(ratios,[.025,.975]).tolist(),
                eligible=by[name,reference]['eligible_for_speedup_claim'] and by[name,candidate]['eligible_for_speedup_claim']))
    result={'cohort':str(cohort),'run':str(run),'allocation':frozen['allocation_id'],'cases':cases,'comparisons':comparisons,
            'audit':{'passed':True,'checks':checks,'failures_retained':summary['case_status_counts']},
            'scope':'Four shapes, one allocation, fixed existing tiles, two explicit AlphaTensor precision modes; no broad algorithm claim.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    names={'native_xla':'Native FP32 out','cubic_fixed':'Our cubic','our_strassen':'Our Strassen',
           'alpha_upstream_bf16':'AlphaTensor released BF16','alpha_fp32_recombine':'AlphaTensor FP32 adaptation','native_bf16':'Native BF16 out'}
    lines=['# AlphaTensor: four-shape proof of concept','',
           'Official TPU-v2 factorization at upstream commit `1949163da3bef7e3eb268a3ac015fd1c2dbfc767`, tested on one v5e.',
           'The released algorithm has 49 block products. Our Strassen uses one level per tile. This comparison includes implementation and decomposition differences.',
           '', '## Complete-call results','',
           'Milliseconds are arithmetic means of 30 synchronized rounds after five warmups; lower is faster. One Gaussian BF16 input per shape. No new tuning sweep. All preparation and output assembly within the JIT are timed; compilation and transfers are excluded.',
           '', '| Shape (M x K x N) | Implementation | Mean ms | Relative L2 | Numerical gate |',
           '|---|---|---:|---:|---|']
    for s in shapes:
        for arm in config['arms']:
            r=by[s['id'],arm];t=r['timing'].get('mean_ms');e=(r.get('correctness') or {}).get('relative_l2')
            lines.append(f'| {s["id"]} ({s["m"]} x {s["k"]} x {s["n"]}) | {names[arm]} | '+(f'{t:.3f}' if t is not None else 'unavailable')+' | '+(f'{e:.6g}' if e is not None else 'unavailable')+f' | {r["status"]} |')
    lines += ['', '## Exploratory paired comparisons','', 'Ratios above 1 favor the candidate. Failed numerical gates make a timing ratio diagnostic only. Intervals are pointwise paired 95% bootstraps, not cross-machine or cross-shape guarantees.','',
              '| Shape | Reference / candidate | Ratio [95% CI] | Both eligible |','|---|---|---:|---|']
    for c in comparisons:
        lines.append(f'| {c["shape_id"]} | {names[c["reference"]]} / {names[c["candidate"]]} | {c["ratio_of_means"]:.3f} [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}] | {c["eligible"]} |')
    lines += ['', '## Precision and interpretation','',
        '- Released mode uses the exact upstream algebra with BF16 subproduct results and recombination. Native BF16 is provided as context.',
        '- Adapted mode changes only subproduct dot output to FP32; recombination then stays FP32. BF16 inputs and pre-additions, coefficients and ordering are unchanged. This is our adaptation, not an upstream performance result.',
        '- Both modes include splitting and assembly of full arrays. Upstream timing supplied pre-split blocks. We use independent calls, because repeated A @ C is not valid for general rectangular A.',
        '- Our Pallas Strassen and cubic retain prior selected tiles on the LLM shapes. The square uses a declared fixed tile. AlphaTensor subproducts are compiled by XLA; none of these choices proves a global optimum.',
        '- Numerical checks use exact quantized operands, host FP32 reference on 128 x 128 positions and every K term, plus whole-output finiteness. Relative L2 <= 0.02 and the existing absolute gate are unchanged.',
        '- All failures remain visible. A failed-gate speed cannot establish an acceptable speedup. Passing the gate also does not prove equal precision or model quality.',
        '- This is not an AlphaTensor search rerun, a tuned Pallas AlphaTensor kernel, a reproduction on TPU v2, or an end-to-end LLM experiment.',
        '', '## Evidence','', f'- Allocation: `{frozen["allocation_id"]}`.',f'- Frozen source: `{frozen["baseline_commit"]}`.',
        f'- Benchmark archive: `{run.relative_to(cohort.parent.parent)}`.',f'- Evidence audit: {checks} hash/statistics/status checks passed.',
        f'- Outcome counts: `{json.dumps(summary["case_status_counts"],sort_keys=True)}`.',
        '- [Official source](https://github.com/google-deepmind/alphatensor/tree/1949163da3bef7e3eb268a3ac015fd1c2dbfc767/benchmarking).']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors=['#3478AC','#DB8B32','#278C78','#8B67AD']
    fig,axes=plt.subplots(2,1,figsize=(11,8.5));fig.subplots_adjust(top=.89,bottom=.10,hspace=.55,left=.08,right=.98)
    for ax,arms,title in [(axes[0],['native_xla','cubic_fixed','our_strassen','alpha_fp32_recombine'],'FP32-output implementations'),
                          (axes[1],['native_bf16','alpha_upstream_bf16'],'BF16-output implementations (released AlphaTensor algebra)')]:
        x=np.arange(4);width=.19 if len(arms)==4 else .29
        for i,arm in enumerate(arms):
            values=[by[s['id'],arm]['timing'].get('mean_ms',np.nan) for s in shapes]
            bars=ax.bar(x+(i-(len(arms)-1)/2)*width,values,width*.9,label=names[arm],color=colors[i],zorder=3)
            for bar,s,val in zip(bars,shapes,values):
                if not by[s['id'],arm]['eligible_for_speedup_claim']:bar.set_hatch('///');bar.set_edgecolor('#333333')
                if np.isfinite(val):ax.text(bar.get_x()+bar.get_width()/2,val+max(values)*.022,f'{val:.1f}',ha='center',va='bottom',fontsize=8)
        ax.set_title(title,loc='left',fontsize=12,pad=33)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=2 if len(arms)==2 else 4,fontsize=8,frameon=False)
        ax.set_xticks(x,['8192 square','Qwen gate/up','Mistral down','Gemma gate/up'])
        ax.set_ylabel('Mean latency (ms)');ax.set_ylim(0,ax.get_ylim()[1]*1.16)
        ax.grid(axis='y',alpha=.2,zorder=0);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('AlphaTensor proof of concept on one TPU v5e',fontsize=17,x=.08,ha='left')
    fig.text(.08,.035,'Lower bars are faster. Hatched bars fail the frozen numerical gate; their timings are diagnostic only.\n30 rounds per arm; fixed prior tiles; synthetic inputs. This is not a fully tuned AlphaTensor comparison.',fontsize=9)
    fig.savefig(out/'latency.png',dpi=180,facecolor='white');plt.close(fig)
    (out/'artifact-manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps({'output':str(out),'audit_checks':checks,'outcomes':summary['case_status_counts']}))


if __name__=='__main__':main()
