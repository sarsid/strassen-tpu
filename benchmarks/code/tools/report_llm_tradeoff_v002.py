"""Report matched-composition and best-Native comparisons without selection bias."""
import argparse
import hashlib
import json
from pathlib import Path


def report(cohort,out):
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((cohort/'plan.json').read_text());data={};hashes={}
    def read(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def stage(key,action):
        receipt=cohort/(key+'-'+action+'-finished.json')
        if not receipt.exists():return dict(status='not_executed'),None
        item=read(receipt);root=Path(item['run'])/'artifacts'
        if (root/'model_summary.json').exists():return read(root/'model_summary.json'),root
        return dict(status='failed',details=read(root/'summary.json') if (root/'summary.json').exists() else item),root
    lines=['# MM-to-model comparison','',
        'Hardware: '+plan['hardware']+'. Exact real BF16 weights, batch 1, sequence 2048.',
        'The primary baseline is tuned composed Native: only the two MLP products change across composed policies. The fastest Native control is also shown separately.',
        'Resident timing includes embeddings, every layer and vocabulary head, dispatch and synchronization. Weight loading/transfer, tokenization and compilation are excluded.',
        '27 timed forwards per arm/scope: nine randomized paired rounds across three held-out prompts. Timing intervals are within one allocation, not hardware-session replication.',
        'Prediction quality uses 16 disjoint held-out windows and 32752 targets. Quality errors use default composed Native as reference. No decode, KV-cache or production-serving claim.','']
    for key in plan['models']:
        stages={};roots={}
        for action in ('tune','resident','quality'):stages[action],roots[action]=stage(key,action)
        data[key]=stages
        lines+=['## '+stages['tune'].get('model_id',key),'']
        for action,row in stages.items():lines.append(action+': '+row['status'])
        qualification=stages['tune'].get('qualification',{})
        lines+=['','Official Native qualification: '+str(qualification.get('passed','unavailable'))+'.','']
        quality=stages['quality'].get('quality',{})
        timings=stages['resident'].get('resident_timings',{})
        for scope,p in timings.items():
            means=p['timings'];best=min((a for a in ('native','native_default','native_full_jit') if a in means),key=lambda a:means[a]['mean_ms'])
            lines += ['### '+scope,'',
                'Fastest measured Native control: `'+best+'`. Positive latency changes mean slower.','',
                '| Arm | Mean ms | Change vs matched Native | Paired 95% interval | Change vs best Native | Quality gate |',
                '|---|---:|---:|---:|---:|---|']
            for arm,t in means.items():
                c=p['comparisons'][arm]['native'];b=p['comparisons'][arm][best];lo,hi=c['paired_round_ci95']
                lines.append(f"| {arm} | {t['mean_ms']:.4f} | {100*(c['latency_ratio']-1):+.2f}% | {100*(lo-1):+.2f}% to {100*(hi-1):+.2f}% | {100*(b['latency_ratio']-1):+.2f}% | {quality.get(arm,{}).get('passed','unavailable')} |")
            lines.append('')
        if quality:
            lines += ['### Prediction and numerical cost','',
                '| Arm | Top-1 accuracy | Delta pp | 95% interval pp | Perplexity | Logit relative L2 | Top-1 agreement | Mean KL |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm,q in quality.items():
                if not q.get('all_finite'):
                    lines.append(f'| {arm} | NONFINITE | — | — | — | — | — | — |');continue
                acc=q['paired_accuracy_delta'];lo,hi=acc['ci95']
                lines.append(f"| {arm} | {100*q['candidate_top1_accuracy']:.4f}% | {100*acc['mean']:+.4f} | {100*lo:+.4f} to {100*hi:+.4f} | {q['candidate_perplexity']:.6f} | {100*q['relative_l2']:.4f}% | {100*q['top1_agreement']:.4f}% | {q['mean_kl']:.7f} |")
            lines+=['','Token archives also retain NLL, forward/reverse KL, total variation, Brier error, top-5 accuracy, confidence/calibration and correct-to-wrong flips. Local and propagated hidden-state errors are stored separately.','']
        root=roots['tune'];mm={}
        if root:
            lines+=['### Isolated MM confirmation','',
                '| Projection | Method | Candidate ms | Tuned Native ms | Change | 95% interval |',
                '|---|---|---:|---:|---:|---:|']
            for proj in ('gateup','down'):
                path=root/('m2048_'+proj+'_confirmation_statistics.json')
                if not path.exists():continue
                d=read(path);mm[proj]=d
                headline=next(iter(d['by_shape'].values()))['headline']
                for arm in ('cubic','one_level','two_level'):
                    c=headline.get(arm,{}).get('comparison_vs_tuned_native')
                    if not c:continue
                    lo,hi=c['ci95']
                    lines.append(f"| {proj} | {arm} | {c['candidate_mean_ms']:.6f} | {c['reference_mean_ms']:.6f} | {100*(c['candidate_over_reference_latency_ratio']-1):+.2f}% | {100*(lo-1):+.2f}% to {100*(hi-1):+.2f}% |")
            lines+=['','MM intervals use paired rounds across three fresh operand windows, conditional on the frozen screening choices. No confirmation reselection.','']
        stages['isolated_mm']=mm
        for action in ('tune','resident','quality'):
            if roots[action]:lines.append(action+' evidence: '+str(roots[action]))
        lines.append('')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(models=data,input_sha256=hashes),indent=2)+'\n')
    (out/'artifact-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(reported_models=list(data),output=str(out))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    report(a.cohort,a.output_dir)


if __name__=='__main__':main()
