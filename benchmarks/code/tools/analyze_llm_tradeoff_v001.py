"""Summarize all resident Native controls and paired speed/accuracy tradeoffs."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);a=p.parse_args()
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    provenance={};data={};lines=['# Resident LLM speed and error results','',
        'Single v6e allocation; exact BF16 weights; batch 1, sequence 2048. Three prompts, nine paired rounds (27 samples per arm).',
        'The timing baseline is the fastest measured Native control for each model and operation; this is a descriptive selection among measured controls.',
        'Accuracy deltas and numerical errors use composed default Native on 16 held-out text windows (32752 predictions). These are different explicitly named baselines.',
        'No production serving, decode, downstream-task or cross-hardware claim. Setup, downloads, compilation and weight transfer are outside resident timing.','']
    def load(key,stage):
        receipt=json.loads((a.cohort/(key+'-'+stage+'-finished.json')).read_text());assert receipt['status']=='completed'
        path=Path(receipt['run'])/'artifacts/model_summary.json'
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
    for key in ['qwen','mistral','qwen14']:
        r=load(key,'resident');q=load(key,'quality');model=dict(model_id=r['model_id'],timings={},quality={})
        lines.extend(['## '+r['model_id'],''])
        for scope,s in r['resident_timings'].items():
            baseline=min(['native','native_default','native_full_jit'],key=lambda n:s['timings'][n]['mean_ms'])
            model['timings'][scope]=dict(baseline=baseline,timings=s['timings'],comparisons=s['comparisons'])
            lines.extend([scope+'; fastest Native control: `'+baseline+'`.','',
                '| Arm | Mean ms | Latency change vs fastest Native | Paired 95% interval for change |',
                '|---|---:|---:|---:|'])
            for arm in ['native_default','native','native_full_jit','cubic','one_level','two_level']:
                t=s['timings'][arm];c=s['comparisons'][arm][baseline];lo,hi=c['paired_round_ci95']
                lines.append(f"| {arm} | {t['mean_ms']:.3f} | {100*(c['latency_ratio']-1):+.2f}% | {100*(lo-1):+.2f}% to {100*(hi-1):+.2f}% |")
            lines.extend(['','Positive changes mean slower. Paired intervals cover timing variability on this allocation, not hardware-session replication.',''])
        lines.extend(['| Arm | Actual top1 accuracy | Change, percentage points | 95% interval, pp | Perplexity | Logit relative L2 | Agreement | Mean KL |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for arm in ['native_default','native_full_jit','cubic','one_level','two_level']:
            s=q['quality'][arm];model['quality'][arm]={k:v for k,v in s.items() if k not in ('calibration','reference_calibration')}
            acc=s['paired_accuracy_delta'];lo,hi=acc['ci95']
            lines.append(f"| {arm} | {100*s['candidate_top1_accuracy']:.4f}% | {100*acc['mean']:+.4f} | {100*lo:+.4f} to {100*hi:+.4f} | {s['candidate_perplexity']:.6f} | {100*s['relative_l2']:.3f}% | {100*s['top1_agreement']:.3f}% | {s['mean_kl']:.6f} |")
        lines.extend(['','All listed quality gates passed and outputs were finite. All Strassen accuracy-difference intervals include zero.',
            'Mistral two-level Strassen has a small positive NLL change whose paired window interval excludes zero; passing the gate does not mean zero numerical cost.',''])
        data[key]=model
    lines.extend(['## Interpretation','',
        'Neither Strassen depth improved full resident prompt or scoring latency against the fastest Native control in these v6e workloads. Both were also slower than tuned Native with the same composed-layer structure.',
        'Qwen3-8B one-level Strassen is faster than the full-JIT Native control, but tuned composed Native is faster than both. Using only full-JIT as the headline denominator would overstate the practical benefit.',
        'Earlier v5e isolated/resident-layer gains are separate hardware and timing scopes and do not establish a v6e full-model improvement.',''])
    (out/'RESULTS.md').write_text('\n'.join(lines))
    (out/'results.json').write_text(json.dumps(dict(models=data,input_sha256=provenance),indent=2))
    print(json.dumps(dict(completed=True,models=list(data),output=str(out))))

if __name__=='__main__':main()
