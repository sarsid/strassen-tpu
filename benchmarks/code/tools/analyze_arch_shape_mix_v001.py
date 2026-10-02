"""Exploratory shape strata and baseline sensitivity of a fixed interim report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def dims(s):
    return [s[k] for k in ('m','k','n')]


STRATA = {
    'all': lambda s: True,
    'any_dimension_below_2048': lambda s: min(dims(s)) < 2048,
    'all_dimensions_at_least_2048': lambda s: min(dims(s)) >= 2048,
    'all_dimensions_multiple_of_2048': lambda s: all(x % 2048 == 0 for x in dims(s)),
    'large_but_not_all_2048_aligned': lambda s: min(dims(s)) >= 2048 and any(x % 2048 for x in dims(s)),
    'all_dimensions_at_least_4096': lambda s: min(dims(s)) >= 4096,
    'all_dimensions_at_least_8192': lambda s: min(dims(s)) >= 8192,
}


def classify(method, baseline):
    c = method.get('comparisons', {}).get(baseline)
    if not method.get('eligible') or not c or not c.get('eligible'):
        return 'unavailable'
    return 'win' if c['ci95'][0] > 1 else 'loss' if c['ci95'][1] < 1 else 'unresolved'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--shapes', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    data=json.loads(a.report.read_text())['results']
    manifest=json.loads(a.shapes.read_text())['shapes']
    completed={tuple(dims(r['shape'])) for r in data}
    summary={}
    for name,predicate in STRATA.items():
        coverage=dict(planned=sum(predicate(s) for s in manifest),
                      completed=sum(predicate(r['shape']) for r in data if r['shape']['output_dtype']=='float32'))
        coverage['remaining']=coverage['planned']-coverage['completed']
        by_output={}
        for dtype in ('float32','bfloat16'):
            rows=[r for r in data if r['shape']['output_dtype']==dtype and predicate(r['shape'])]
            methods={}
            for role in ('native','cubic','s1','s2'):
                methods[role]={}
                for baseline in ('native_default','native','cubic'):
                    arms=[r['methods'].get(role,{}) for r in rows]
                    ratios=[m['comparisons'][baseline]['speedup'] for m in arms if classify(m,baseline)!='unavailable']
                    methods[role][baseline]=dict(counts=dict(Counter(classify(m,baseline) for m in arms)),
                        median_speedup=statistics.median(ratios) if ratios else None)
            by_output[dtype]=dict(shapes=len(rows),methods=methods)
        summary[name]=dict(coverage=coverage,by_output=by_output)
    examples=[]
    for r in data:
        if STRATA['all_dimensions_multiple_of_2048'](r['shape']):
            examples.append(dict(shape_mkn=dims(r['shape']),output_dtype=r['shape']['output_dtype'],
                methods={role:dict(mean_ms=r['methods'][role]['mean_ms'],
                    comparisons=r['methods'][role]['comparisons']) for role in ('native_default','native','cubic','s1','s2')}))
    out=dict(verified_shapes=len(completed),strata=summary,aligned_examples=examples,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.report,a.shapes)},
        interpretation='Exploratory descriptive strata of the saved development results, not new measurements or unseen-shape validation. Categories overlap. Pointwise intervals and frozen choices are reused unchanged from the verified report; no retry samples are pooled and no overall winner is reselected. Large uses a minimum dimension threshold, not only operation count.')
    (a.output_dir/'shape_mix.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
