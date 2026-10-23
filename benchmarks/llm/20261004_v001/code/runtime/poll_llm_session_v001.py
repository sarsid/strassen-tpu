"""Read-only snapshot plus all durable group/case archives in one model session."""
import argparse,json,re
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid run')
    run=Path('/content/Strassen_MM_Focus/runs')/a.run_id;result=dict(kind='parent_replication_snapshot',checkpoints=[],cases=[])
    for name in ('status.json','archive-ready.json'):
        if (run/name).exists():result[name]=json.loads((run/name).read_text())
    for case in sorted((run/'artifacts/cases').glob('*')):
        if not case.is_dir():continue
        ready=case/'main-prefill/checkpoint-ready.json'
        if ready.exists():result['checkpoints'].append(dict(json.loads(ready.read_text()),job_id=case.name))
        ready=case/'case-ready.json'
        if ready.exists():result['cases'].append(json.loads(ready.read_text()))
    status=result.get('status.json',{});active=status.get('active','');subpath=status.get('active_case_path','')
    folder=run/'artifacts'/subpath;files=[run/'worker.log']
    if active:files += [folder/(active+'.log'),folder/active/'events.jsonl']
    result['tails']={}
    for path in files:
        if path.exists():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-14000));result['tails'][str(path.relative_to(run))]=f.read().decode('utf8','replace')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
