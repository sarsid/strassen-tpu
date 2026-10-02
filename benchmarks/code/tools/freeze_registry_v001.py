"""Append fingerprints for code declared frozen after actual execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def register(run,files):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT/'runs') or not (run/'completion.json').is_file():
        raise ValueError('Register a completed/failed archived execution under this study')
    manifest=json.loads((run/'source-manifest.json').read_text())
    records=[]
    for name in files:
        path=(run/'source'/name).resolve()
        if not path.is_relative_to(run/'source') or not path.is_file():raise ValueError('Missing frozen source file')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=manifest['source_sha256'][name]:raise ValueError('Frozen snapshot hash mismatch')
        current=ROOT/name
        if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest()!=digest:
            raise ValueError('Working file diverges; preserve improvement in a new version before registration: '+name)
        records.append({'utc':datetime.now(timezone.utc).isoformat(),'path':name,'sha256':digest,
                        'execution_run_id':run.name,'source_commit':manifest['source_commit']})
    with (ROOT/'protocols/code_freezes_v001.jsonl').open('a') as f:
        for row in records:f.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps({'registered':len(records),'run_id':run.name}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--file',action='append',required=True)
    a=p.parse_args();register(a.run,a.file)
