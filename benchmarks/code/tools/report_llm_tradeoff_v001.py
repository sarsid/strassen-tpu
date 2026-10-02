"""Combine archived quality and resident timings without hiding missing runs."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.cohort/'plan.json').read_text());all_results={}
    lines=['# Large LLM speed and accuracy tradeoffs','',
        'Exact BF16 weights. Kernel tuning and text evaluation use disjoint windows. Hardware: '+plan['hardware']+'.',
        'Resident prompt forward includes embedding lookup, all layers and last-token vocabulary logits. Scoring includes all-token losses. These are research forwards, not production serving.','']
    for key in plan['models']:
        row={};lines += ['## '+key,'']
        for action in ('tune','quality','resident'):
            receipt=a.cohort/(key+'-'+action+'-finished.json')
            if not receipt.exists():row[action]=dict(status='not_executed');continue
            d=json.loads(receipt.read_text());run=Path(d['run']);path=run/'artifacts/model_summary.json'
            if path.exists():row[action]=json.loads(path.read_text())
            else:
                s=run/'artifacts/summary.json'
                row[action]=dict(status='failed',details=json.loads(s.read_text()) if s.exists() else d)
            row[action]['evidence']=str(path if path.exists() else run/'completion.json')
        all_results[key]=row;quality=row['quality'].get('quality',{})
        perf=row['resident'].get('resident_timings',{}).get('prompt_forward_last_token',{})
        if not quality:lines += ['Quality unavailable: '+row['quality']['status']+'.','']
        else:
            lines += ['| Algorithm | Resident forward ms | Change vs full-JIT Native | Actual top1 accuracy | Accuracy delta, pp | Perplexity | Logit rel L2 | Agreement | Quality gate |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
            for arm,q in quality.items():
                if not q.get('all_finite'):
                    lines += [f'| {arm} | — | — | — | — | — | — | — | NONFINITE |'];continue
                timing=perf.get('timings',{}).get(arm,{});comparison=perf.get('comparisons',{}).get(arm,{}).get('native_full_jit',{})
                ms=f"{timing['mean_ms']:.3f}" if timing else 'unavailable'
                reduction=f"{comparison['latency_reduction']:.2%}" if comparison else 'unavailable'
                lines += [f"| {arm} | {ms} | {reduction} | {q['candidate_top1_accuracy']:.3%} | {100*(q['candidate_top1_accuracy']-q['reference_top1_accuracy']):+.4f} | {q['candidate_perplexity']:.5f} | {q['relative_l2']:.3%} | {q['top1_agreement']:.3%} | {q['passed']} |"]
            lines += ['','Positive latency reduction means faster; negative means slower. Accuracy is against the actual next tokens. Agreement is with composed default Native.','']
        for action in row:lines += [action+': '+row[action]['status']+'. Evidence: '+row[action].get('evidence','no completed stage')]
        lines += ['']
    (a.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (a.output_dir/'results.json').write_text(json.dumps(all_results,indent=2)+'\n')
    (a.output_dir/'artifact-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output_dir.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps({'reported_models':list(all_results),'output':str(a.output_dir)}))
if __name__=='__main__':main()
