"""Acquire one replacement and restore verified data before the unchanged worker."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from resume_accuracy_remote_v001 import inspect_seed

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def records(text):
    rows=[]
    for line in text.splitlines():
        try:rows.append(json.loads(line))
        except ValueError:pass
    return rows


def owned(assignments,session):
    matches=[x for x in assignments if x.get('name')==session]
    if len(matches)==1 and len(assignments)==1:return matches[0]['endpoint']
    if assignments:raise RuntimeError('An unowned or ambiguous allocation requires reconciliation')
    return None


def observer_command(root,run_id,endpoint):
    return [sys.executable,str(root/'runtime/observe_accuracy_study_v002.py'),'--run-id',run_id,'--session',run_id,'--endpoint',endpoint]


def wait_ready(observer,root,endpoint,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        code=observer.poll()
        if code is not None:raise RuntimeError('Observer exited before readiness: '+str(code))
        path=root/'observer-ready-v002.json'
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('pid')==observer.pid and value.get('endpoint')==endpoint:return
        time.sleep(.2)
    raise RuntimeError('Observer did not acknowledge readiness')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.run_id):raise ValueError('Invalid run ID')
    root=ROOT/'runs'/a.run_id;manifest=json.loads((root/'source-manifest.json').read_text())
    if hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Changed frozen source')
    seed=json.loads((root/'seed-receipt.json').read_text())
    inspect_seed(root/'resume-seed.tar.gz',seed['sha256'],manifest['sha256'])
    lock=(root/'starter.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=root/'starter-receipts-v003';receipts.mkdir(exist_ok=True)
    control=[sys.executable,str(ROOT/'runtime/colab_control_v003.py')]
    def status(stage,**fields):
        value=dict(observed_utc=datetime.now(timezone.utc).isoformat(),remote_state=dict(status='preparing',stage=stage,**fields),test_evaluations_completed=seed['completed_task_evaluations'],test_evaluations_total=9)
        save(root/'live-status.json',value)
    def command(label,args,timeout=180):
        path=receipts/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+label+'.jsonl')
        with path.open('x') as f:
            try:result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        return code,records(path.read_text())
    endpoint=None;attempt=1
    while endpoint is None:
        status('Reconciling allocation',allocation_attempt=attempt)
        code,rows=command('assignments',control+['list'])
        listing=next((x for x in rows if x.get('kind')=='allocations'),None)
        if code or listing is None:time.sleep(60);continue
        endpoint=owned(listing['assignments'],a.run_id)
        if endpoint is not None:break
        status('Waiting for Colab v6e',allocation_attempt=attempt)
        code,rows=command('allocation', [sys.executable,str(ROOT/'runtime/allocate_llm_diagnostic_v001.py'),
            '--session',a.run_id,'--journal-root',str(ROOT),'--events',str(root/'allocation-events.jsonl')],timeout=660)
        allocation=next((x for x in rows if x.get('kind')=='allocation'),None)
        if allocation:endpoint=allocation['endpoint'];break
        status('Allocation unavailable; automatic retry pending',allocation_attempt=attempt,
            last_error=next((x for x in reversed(rows) if x.get('kind')=='allocation_error'),{'error_type':'No completed allocation response'}))
        attempt+=1;time.sleep(min(300,60*attempt))
    identity=['--session',a.run_id,'--expect-endpoint',endpoint]
    # create with an existing named, reconciled session only refreshes its
    # keepalive; it cannot allocate a second endpoint under expect-endpoint.
    code,rows=command('keepalive',control+['create',*identity])
    if code:raise RuntimeError('Owned allocation could not be refreshed')
    heartbeat=next(x['heartbeat_pid'] for x in rows if x.get('kind')=='allocation')
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(heartbeat)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    save(root/'awake-policy.json',dict(heartbeat_pid=heartbeat,caffeinate_pid=awake.pid))
    save(root/'owned-allocation.json',dict(session=a.run_id,endpoint=endpoint))
    status('Preparing allocated TPU',endpoint=endpoint)
    for retry in range(5):
        code,_=command('bootstrap',control+['exec-file',*identity,'--file',str(ROOT/'runtime/bootstrap_parent_replication_v001.py'),'--timeout','60'])
        if not code:break
        time.sleep(15)
    else:raise RuntimeError('TPU bootstrap failed; retain allocated machine')
    remote='/content/Strassen_MM_Focus/'+a.run_id+'-source.tar.gz'
    code,rows=command('upload',control+['upload',*identity,'--local',str(root/'source.tar.gz'),'--remote',remote])
    if code:
        # A lost upload response is reconciled by downloading the existing
        # unique path and verifying bytes before any execution is attempted.
        downloaded=root/'upload-reconciliation.tar.gz'
        if downloaded.exists():code=0
        else:code,_=command('verify-upload',control+['download',*identity,'--remote',remote,'--local',str(downloaded)])
        if code or hashlib.sha256(downloaded.read_bytes()).hexdigest()!=manifest['sha256']:raise RuntimeError('Source upload needs reconciliation; retain TPU')
    remote_seed='/content/Strassen_MM_Focus/'+a.run_id+'-resume-seed.tar.gz'
    code,_=command('upload-seed',control+['upload',*identity,'--local',str(root/'resume-seed.tar.gz'),'--remote',remote_seed])
    if code:
        downloaded=root/'seed-upload-reconciliation.tar.gz'
        if not downloaded.exists():code,_=command('verify-seed-upload',control+['download',*identity,'--remote',remote_seed,'--local',str(downloaded)])
        if not downloaded.exists() or hashlib.sha256(downloaded.read_bytes()).hexdigest()!=seed['sha256']:raise RuntimeError('Checkpoint upload needs reconciliation; retain TPU')
    status('Resuming unchanged study from saved checkpoints',endpoint=endpoint)
    launch_remote='/content/Strassen_MM_Focus/runs/'+a.run_id+'/launch.json'
    code,rows=command('prelaunch',control+['read',*identity,'--remote',launch_remote])
    if code==0:
        value=next(x for x in rows if x.get('kind')=='remote_file')
        if (json.loads(value['text'])['source_sha256']!=manifest['sha256'] or json.loads(value['text']).get('seed_sha256')!=seed['sha256']):raise RuntimeError('Existing remote launch has different source')
    elif not any(x.get('http_status')==404 for x in rows):raise RuntimeError('Cannot establish absence before launch; retain TPU')
    else:
        code,rows=command('launch',control+['exec-file',*identity,'--file',str(ROOT/'runtime/resume_accuracy_remote_v001.py'),'--timeout','90',
            '--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+manifest['sha256'],'--script-arg=--run-id','--script-arg='+a.run_id,'--script-arg=--seed','--script-arg='+remote_seed,'--script-arg=--seed-sha256','--script-arg='+seed['sha256']])
        if code:
            time.sleep(5)
            code,rows=command('reconcile-launch',control+['read',*identity,'--remote',launch_remote])
            value=next((x for x in rows if x.get('kind')=='remote_file'),None)
            if code or not value or (json.loads(value['text'])['source_sha256']!=manifest['sha256'] or json.loads(value['text']).get('seed_sha256')!=seed['sha256']):raise RuntimeError('Launch needs reconciliation; never duplicate remote worker')
    with (root/'observer-v002.log').open('a') as log:
        args=observer_command(ROOT,a.run_id,endpoint)
        observer=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'observer-start-v002.json',dict(pid=observer.pid,command=args))
    wait_ready(observer,root,endpoint)
    # A live process is insufficient: require a successful remote status read.
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        if observer.poll() is not None:raise RuntimeError('Observer exited before first remote status')
        value=json.loads((root/'live-status.json').read_text())
        if isinstance(value.get('remote_state'),dict) and value['remote_state'].get('scope')=='three-model paired downstream subsets':break
        time.sleep(2)
    else:raise RuntimeError('No verified remote status after launch')
    save(root/'starter-completion-v003.json',dict(status='queue_launched_and_observed',endpoint=endpoint,observer_pid=observer.pid,finished_utc=datetime.now(timezone.utc).isoformat()))
    # Continue supervising the observer, rather than abandoning a child after
    # Popen. Restarts only the read-only observer, never the remote experiment.
    while True:
        completion=root/'observer-completion-v002.json'
        if completion.exists():
            save(root/'supervisor-completion-v003.json',json.loads(completion.read_text()));return
        if observer.poll() is not None:
            with (root/'observer-v002.log').open('a') as log:
                observer=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'observer-restart-v002.json',dict(pid=observer.pid,utc=datetime.now(timezone.utc).isoformat()))
            wait_ready(observer,root,endpoint)
        time.sleep(15)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps(dict(kind='starter_failure',error_type=type(error).__name__,error=str(error) if type(error) in (RuntimeError,ValueError) else None)),flush=True)
        raise SystemExit(1)
