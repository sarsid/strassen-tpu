"""Read a bounded progress snapshot, including immutable per-batch exports."""
import argparse
import json
import hashlib
from pathlib import Path
import re


def make_snapshot(run):
    result=dict(kind='gemma4_batch_snapshot',tails={})
    for name in ('status.json','progress.json','archive-ready.json'):
        if (run/name).exists(): result[name]=json.loads((run/name).read_text())
    active=result.get('status.json',{}).get('active')
    files=[run/'worker.log']
    if active:
        files.extend([run/'artifacts'/(active+'.log'),run/'artifacts'/active/'events.jsonl'])
        if active=='parent-setup': files.extend([run/'artifacts/parent-install.log',run/'artifacts/torch-install.log'])
    for path in files:
        if path.is_file():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-14000)); result['tails'][str(path.relative_to(run))]=f.read().decode('utf8','replace')
    result['stage_exports']=[]
    for row in result.get('status.json',{}).get('stages',[]):
        name=row['name']
        if row['returncode']!=0 or not re.fullmatch(r'(b(1|4|8|16|32|64)-(tune|qualify|confirm|quality|prefill)|oracle)-a[01]',name):continue
        files={}
        for filename in ('profile.json','summary.json'):
            path=run/'artifacts'/name/filename
            if path.exists():
                if path.stat().st_size>2*1024**2:raise ValueError('Stage JSON artifact too large')
                content=path.read_bytes();json.loads(content)
                files[filename]=dict(content_utf8=content.decode('utf8'),sha256=hashlib.sha256(content).hexdigest())
        if files:result['stage_exports'].append(dict(name=name,files=files))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',a.run_id): raise ValueError('Invalid run ID')
    run=Path('/content/Strassen_MM_Focus/runs')/a.run_id
    print(json.dumps(make_snapshot(run)),flush=True)


if __name__=='__main__': main()
