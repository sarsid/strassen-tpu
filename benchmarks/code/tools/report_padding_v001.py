"""Report every frozen boundary policy, paired against its existing-depth control."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from report_v6e_suite_v001 import paired


def build(cohort,out):
    out.mkdir(parents=True,exist_ok=False);hashes={};events=[];missing=[]
    cfg=json.loads((cohort/'source/configs/padding_v001/campaign.json').read_text())
    shapes=json.loads((cohort/'source/configs/padding_v001/shapes.json').read_text())['shapes']
    for i in range(1,4):
        receipt=cohort/f'padding-{i:02d}-measure-finished.json'
        if not receipt.exists():missing.append(str(receipt));continue
        r=json.loads(receipt.read_text());path=Path(r['run'])/'artifacts/results.jsonl'
        if not path.exists():missing.append(str(path));continue
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    lines=['# Boundary-handling probe on v6e','',
      'Ten boundary shapes and three regular shapes. Current S1/S2 are the primary controls; Native default is context only. All policies are fixed in advance, with matched interior kernels and tiles per depth.',
      'BF16 inputs and Strassen preadds, FP32 output. Complete-call timings include device slicing, padding, all multiplications, partial-result additions, concatenation and crop; transfer/compile excluded.',
      'Three fresh Gaussian inputs × 30 rotated paired rounds. Pointwise hierarchical paired 95% intervals, without multiplicity adjustment. No winner selection or new-shape generalization claim.',
      'Errors use 128×128 host FP64 reference values across all K, including outer and internal boundaries. Finiteness covers the full output. Sampled maximum error is not a full-matrix bound.','',
      '| M × K × N | Method | Mean ms | Speedup vs same-depth control [95%] | Relative L2 % | Sampled max abs error | Gate |',
      '|---|---|---:|---|---:|---:|---|']
    results=[]
    for s in shapes:
        sid=s['id'];methods={}
        cases=[e for e in events if e.get('event')=='case_result' and e['shape_id']==sid]
        samples=[e for e in events if e.get('event')=='sample' and e['shape_id']==sid]
        raw={arm:{(e['seed'],e['round']):e['elapsed_ms'] for e in samples if e['arm_id']==arm} for arm in cfg['arms']}
        for arm in cfg['arms']:
            rows=[e for e in cases if e['arm_id']==arm]
            metrics=[e['correctness'] for e in rows if e.get('correctness')]
            complete=len(rows)==3 and len(raw[arm])==90
            eligible=complete and all(e['eligible_for_speedup_claim'] for e in rows)
            errors={k:max(q[k] for q in metrics) for k in ['relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error'] if metrics and all(k in q for q in metrics)}
            reference='S1' if arm.startswith('S1') else 'S2' if arm.startswith('S2') else 'Native'
            compare=paired(raw[reference],raw[arm]) if complete else None
            method=dict(mean_ms=statistics.mean(raw[arm].values()) if raw[arm] else None,
                complete=complete,eligible=eligible,errors=errors,comparison=compare,
                comparison_control=reference,cases=rows)
            methods[arm]=method
            speed='—' if compare is None else f'{compare["speedup"]:.3f} [{compare["ci95"][0]:.3f}, {compare["ci95"][1]:.3f}]'
            ms='—' if method['mean_ms'] is None else f'{method["mean_ms"]:.6f}'
            rel='—' if not errors else f'{100*errors["relative_l2"]:.4f}'
            maximum='—' if not errors else f'{errors["max_abs_error"]:.6g}'
            lines.append('| '+' × '.join(str(s[d]) for d in ('m','k','n'))+f' | {arm} | {ms} | {speed} | {rel} | {maximum} | {eligible} |')
        for arm,m in methods.items():
            c=m['comparison'];m['comparison_eligible']=m['eligible'] and methods[m['comparison_control']]['eligible']
            m['classification']='unavailable' if not c or not m['comparison_eligible'] else 'win' if c['ci95'][0]>1 else 'loss' if c['ci95'][1]<1 else 'inconclusive'
        results.append(dict(shape=s,methods=methods))
    failures=[e for e in events if e.get('event')=='error']
    complete=not missing and all(m['complete'] for r in results for m in r['methods'].values())
    lines+=['','## Limits','',f'Complete measurement coverage: {complete}. Missing phases/files: {missing}. Recorded execution/compile failures: {len(failures)}.',
       'Native-fringe policy changes the fraction of arithmetic performed with Strassen. Any error reduction is reported alongside that fraction, not attributed solely to padding.',
       'Regular shapes route through the unchanged incumbent. Small timing differences on these rows provide an overhead/noise check.',
       'This probe does not include full-K retuning, fused boundary loads, fused K corrections, or LLMs.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'results.json').write_text(json.dumps(dict(completed=complete,results=results,failures=failures,missing=missing,input_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(completed=complete,shapes=len(results),failures=len(failures),output=str(out))))
    return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
