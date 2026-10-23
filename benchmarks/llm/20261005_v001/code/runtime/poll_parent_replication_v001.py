"""Bounded read-only snapshot of one parent-replication worker."""
import argparse
import json
from pathlib import Path
import re

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid run ID')
    run=Path('/content/Strassen_MM_Focus/runs')/a.run_id
    result=dict(kind='parent_replication_snapshot')
    for name in ['status.json','archive-ready.json']:
        if (run/name).exists():result[name]=json.loads((run/name).read_text())
    active=result.get('status.json',{}).get('active','')
    files=[run/'worker.log']
    if active:
        files.extend([run/'artifacts'/(active+'.log'),run/'artifacts'/active/'events.jsonl'])
        if active.endswith('-setup'):files.append(run/'artifacts'/(active[:-6]+'-install.log'))
    result['tails']={}
    for path in files:
        if path.exists():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-12000))
                result['tails'][str(path.relative_to(run))]=f.read().decode('utf8','replace')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
