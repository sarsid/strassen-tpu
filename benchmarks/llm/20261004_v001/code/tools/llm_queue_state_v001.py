"""Pure queue/ledger invariants. Planned, failed and measured stay distinct."""
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path


def utc():return datetime.now(timezone.utc).isoformat()
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default

def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def identity(key):return hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
def plan(protocol):
    jobs=[]
    # Workload-major gives every model an early small qualification workload.
    for w in protocol['prefill']:
        for model in protocol['models']:
            key=dict(model=model['id'],revision=model['revision'],**w,architecture='v6e')
            groups={dtype:[identity(dict(key,algorithm=alg,output_dtype=dtype)) for alg in protocol['algorithms']] for dtype in protocol['output_dtypes']}
            jobs.append(dict(job_id=identity(key),**key,groups=groups,status='queued',attempts=[]))
    return jobs

def ledger(path):
    rows={}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            row=json.loads(line)
            if row['entry_id'] in rows:raise ValueError('Duplicate ledger identity')
            rows[row['entry_id']]=row
    return rows

def remaining(job,rows):
    missing=[]
    for dtype,ids in job['groups'].items():
        count=sum(i in rows for i in ids)
        if count not in (0,len(ids)):raise ValueError('Incomplete five-arm group in durable ledger')
        if not count:missing.append(dtype)
    return missing

def merge(path,new_rows,job,evidence):
    rows=ledger(path);allowed={i for group in job['groups'].values() for i in group}
    incoming={r['entry_id']:r for r in new_rows}
    if len(incoming)!=len(new_rows) or not set(incoming)<=allowed:raise ValueError('Unplanned or duplicate measurement')
    for group in job['groups'].values():
        count=sum(i in incoming for i in group)
        if count not in (0,len(group)):raise ValueError('Partial arm group cannot be promoted')
    for key,row in incoming.items():
        identity_keys={k:row[k] for k in ('model','revision','batch','sequence','algorithm','output_dtype','architecture')}
        if identity(identity_keys)!=key or row.get('status')!='completed' or not row.get('measurement'):raise ValueError('Not a completed matched measurement')
        saved=dict(row,evidence_run=evidence)
        if key in rows and rows[key]!=saved:raise ValueError('Refusing to replace completed evidence')
        rows[key]=saved
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows.values():f.write(json.dumps(row,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    return len(rows)

def failure_action(receipt):
    status=receipt.get('status')
    if receipt.get('release_exit_code',0)!=0:return 'release_blocked'
    if status=='allocation_failed':return 'allocation_wait'
    if receipt.get('status')=='completed':return 'completed'
    # Independent workloads still run after bounded retries; never relax gates.
    if receipt.get('failure_class')=='model_access':return 'model_blocked'
    return 'retry_workload'
