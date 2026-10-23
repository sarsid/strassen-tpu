"""Frozen-choice v6e recursion timings, paired intervals and numerical errors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);hashes={}
    def read(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    roots={k:Path(read(args.cohort/('opt-'+k+'-finished.json'))['run'])/'artifacts' for k in ('screen','confirm')}
    selected=read(roots['screen']/'selections.json')['selected']
    files={k:roots[k]/'results.jsonl' for k in roots}
    events={k:[json.loads(line) for line in file.read_text().splitlines()] for k,file in files.items()}
    hashes.update({str(file):hashlib.sha256(file.read_bytes()).hexdigest() for file in files.values()})
    cases=[r for r in events['confirm'] if r.get('event')=='case_result'];samples=[r for r in events['confirm'] if r.get('event')=='sample']
    lines=['# v6e: Native versus within-tile Strassen recursion', '',
        'Five original-kernel tiles per depth; eight reconstruction/tile choices per optimized depth; three cubic tiles; five Native settings. Four shapes selected for confirmed v5e wins. Frozen finalists: three fresh inputs, 30 paired rounds each.',
        'BF16 inputs/preadds, FP32 accumulation/output. Complete device call includes padding/crop; host transfer and compilation excluded. One hardware allocation.',
        'Numerical errors compare the same quantized inputs against host FP64 over every K and 128 sampled rows × 128 columns. Output finiteness is checked in full.',
        'A numerical failure is retained as a diagnostic and is not a qualified speedup. These are synthetic MM results, not an LLM benchmark.', '']
    output={}
    for shape,selection in selected.items():
        selection=dict(selection)
        selection['native_default']=dict(candidate=dict(arm_id='native_default',tile=None))
        lines+=['## '+shape,'','| Method | Mean ms | Latency change vs tuned Native | Paired 95% interval | Worst relative L2 | Worst sampled max error | Numerical gate | Selected tile BM,BN,BK |',
                '|---|---:|---:|---:|---:|---:|---|---|'];output[shape]={}
        baseline=selection['native'].get('candidate')
        if baseline is None:raise ValueError('No Native baseline for '+shape)
        def raw(arm):
            return {(r['seed'],r['round']):r['elapsed_ms'] for r in samples if r['shape_id']==shape and r['arm_id']==arm}
        native=raw(baseline['arm_id'])
        for family,item in selection.items():
            candidate=item.get('candidate')
            if candidate is None:
                lines.append('| '+family+' | — | — | — | — | — | no measured candidate | — |');output[shape][family]=item;continue
            arm=candidate['arm_id'];rows=[r for r in cases if r['shape_id']==shape and r['arm_id']==arm];values=raw(arm)
            keys=sorted(values.keys() & native.keys())
            if len(keys)!=90 or len(rows)!=3:
                lines.append('| '+family+' | — | — | — | — | — | incomplete confirmation | '+str(candidate['tile'])+' |')
                output[shape][family]=dict(candidate=candidate,status='incomplete',cases=rows);continue
            c=np.array([values[k] for k in keys]).reshape(3,30);n=np.array([native[k] for k in keys]).reshape(3,30)
            rng=np.random.default_rng(923);seed_ix=rng.integers(0,3,(4000,3,1));round_ix=rng.integers(0,30,(4000,3,30))
            ratios=c[seed_ix,round_ix].mean((1,2))/n[seed_ix,round_ix].mean((1,2));low,high=np.quantile(ratios,[.025,.975])
            metrics=[r['correctness'] for r in rows];passed=all(r['eligible_for_speedup_claim'] for r in rows)
            rel=max(q['relative_l2'] for q in metrics);maximum=max(q['max_abs_error'] for q in metrics);ratio=float(c.mean()/n.mean())
            output[shape][family]=dict(candidate=candidate,mean_ms=float(c.mean()),native_ms=float(n.mean()),latency_ratio=ratio,
                ci95=[float(low),float(high)],numerically_eligible=passed,worst_relative_l2=rel,worst_sampled_max_abs=maximum,cases=rows)
            lines.append(f'| {family} | {c.mean():.6f} | {100*(ratio-1):+.2f}% | {100*(low-1):+.2f}% to {100*(high-1):+.2f}% | {100*rel:.3f}% | {maximum:.6g} | {passed} | {candidate["tile"]} |')
        lines.append('')
    errors=[r for v in events.values() for r in v if r.get('event')=='error']
    compiles=[r for v in events.values() for r in v if r.get('event')=='compilation']
    lines+=['## Search costs and limits','',f'{len(errors)} recorded compile/execution failures; {len(compiles)} successful compilation records across screening and confirmation.',
            'Intervals resample fresh input seeds and paired timing rounds within this allocation; they do not establish reproducibility across TPU sessions.',
            'The bounded tile search does not establish an optimal implementation for any recursion depth. All raw timings, errors, compiler memory analyses and selected policies are retained.', '']
    for error in errors:lines.append('- '+error['arm_id']+': '+error.get('status','error')+' — '+error.get('message','').split('\n')[0][:350])
    (args.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (args.output_dir/'results.json').write_text(json.dumps(dict(results=output,input_sha256=hashes,errors=errors,compilations=compiles),indent=2)+'\n')
    print(json.dumps(dict(reported_shapes=list(output),output=str(args.output_dir))))


if __name__=='__main__':main()
