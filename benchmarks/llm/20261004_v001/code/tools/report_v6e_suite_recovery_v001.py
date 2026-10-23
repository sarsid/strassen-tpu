"""Verified row-wise join of two v6e cohorts; never pool paired measurements."""
import argparse,hashlib,json,shutil
from pathlib import Path
from run_large_real_v002 import verify_phase
from analyze_v6e_opt_profiles_v002 import analyze
from report_v6e_suite_v001 import build

def report(original,cohort,out):
    out.mkdir(parents=True,exist_ok=False);combined=out/'combined-evidence';combined.mkdir()
    provenance={};profile_results=[];profile_errors=[]
    def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
    for name in ('campaign.json','shapes.json'):
        old=original/'source/configs/v6e_suite_v001'/name;new=cohort/'source/configs/v6e_suite_v001'/name
        if old.read_bytes()!=new.read_bytes():raise ValueError('Scientific configuration differs across cohorts')
        target=combined/'source/configs/v6e_suite_v001'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,target)
    for i in range(1,13):
        owner=original if i<=9 else cohort
        allocation=json.loads((owner/'cohort.json').read_text())['allocation_id']
        prefix=f'suite-{i:02d}'
        for suffix in ('screen','confirm'):
            stage=prefix+'-'+suffix;run,summary=verify_phase(owner,stage)
            receipt=json.loads((owner/(stage+'-finished.json')).read_text());save(combined/(stage+'-finished.json'),receipt)
            selection=run/'artifacts'/('selections.json'if suffix=='screen'else 'selection_used.json')
            if json.loads(selection.read_text())['allocation_id']!=allocation:raise ValueError('Selection allocation mismatch')
            groups=json.loads((run/'artifacts/planned_cases.json').read_text())
            for g in groups:
                sid=g['shape']['id'];v=dict(cohort=str(owner),allocation_id=allocation,batch=prefix)
                if sid in provenance and provenance[sid]!=v:raise ValueError('A shape spans allocations or batches')
                provenance[sid]=v
            if suffix=='confirm':
                for trace in sorted((run/'artifacts/profiles').glob('**/*.trace.json.gz')):
                    try:profile_results.append(analyze(trace))
                    except Exception as exc:profile_errors.append(dict(trace=str(trace),error_type=type(exc).__name__,message=str(exc)))
    if len(provenance)!=168:raise ValueError('Expected exactly 168 distinct completed shapes')
    save(combined/'operations/device-analysis/artifacts/profiles.json',dict(results=profile_results,errors=profile_errors))
    code=build(combined,out/'report')
    if code:raise ValueError('Combined shape report incomplete')
    p=out/'report/results.json';r=json.loads(p.read_text())
    for row in r['results']:row['provenance']=provenance[row['shape_id']]
    r['allocation_policy']='126 original + 42 recovery shapes. Every shape uses screen and confirmation from the same allocation; no paired samples pooled across allocations.'
    save(p,r);save(out/'provenance.json',provenance)
    p=out/'report/RESULTS.md';p.write_text('RECOVERY JOIN: first 126 shapes from the original v6e allocation; remaining 42 from a replacement allocation. Each shape has its own paired Native controls on its own allocation. No cross-allocation sample pooling. Old partial batch 10 is excluded. Per-shape allocation IDs are in results.json and provenance.json.\n\n'+p.read_text())
    print(json.dumps(dict(completed=True,shapes=168,original_shapes=126,recovery_shapes=42,device_profiles=len(profile_results),profile_errors=len(profile_errors))))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('original','cohort','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.original,a.cohort,a.output_dir)
