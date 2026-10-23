"""Persistent diagnostic allocator using the user-requested retained-TPU policy."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import itertools
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def records(path):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except ValueError: pass
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--start-authorized',action='store_true'); p.add_argument('--controller-python',required=True); p.add_argument('--await-existing',action='store_true'); p.add_argument('--keep-waiting',action='store_true'); a=p.parse_args()
    if not a.start_authorized: p.error('Prepared only: explicit user resumption is required before allocation')
    run=a.run.resolve()
    if not run.is_relative_to(ROOT/'runs'): raise ValueError('Run must belong to this repository')
    lock=(run/'starter.lock').open('ab'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'control').exists(): raise RuntimeError('Controller already started; do not launch a duplicate')
    manifest=json.loads((run/'source-manifest.json').read_text())
    controller=ROOT/'tools/run_gemma4_diagnostic_v002.py'
    override=json.loads((run/'controller-override-v002.json').read_text())
    if hashlib.sha256(controller.read_bytes()).hexdigest()!=override['files']['tools/run_gemma4_diagnostic_v002.py']:
        raise RuntimeError('Controller differs from the validated frozen source')
    events=run/'allocation-events.jsonl'; last=None; endpoint=None
    def progress(title,detail,state='working'):
        value=dict(status='waiting' if state=='working' else state,allocation='waiting',stage='allocation',stage_label=title,test=title,detail=detail,utc=datetime.now(timezone.utc).isoformat())
        temp=run/'progress.json.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(run/'progress.json')
        subprocess.run(['python3',str(ROOT/'tools/record_progress_v001.py'),'--kind','activity','--state',state,'--title',title,'--detail',detail,'--evidence',str(events.relative_to(ROOT))],check=True,stdout=subprocess.DEVNULL)
    def owned():
        values=[r for r in records(events) if r.get('kind')=='allocation' and r.get('created') and r.get('session')==run.name]
        if len(values)>1: raise RuntimeError('Multiple allocation receipts require reconciliation')
        return values[0]['endpoint'] if values else None
    if a.await_existing:
        deadline=time.monotonic()+1300
        while time.monotonic()<deadline:
            endpoint=owned()
            if endpoint: break
            rows=records(events); last=rows[-1] if rows else {}
            if last.get('kind')=='allocation_error': break
            time.sleep(10)
        else: raise TimeoutError('Existing allocation request did not settle; no duplicate request made')
    for attempt in (itertools.count() if a.keep_waiting else range(24)):
        endpoint=owned()
        if endpoint: break
        if last and last.get('http_status') in (401,403):
            progress('Colab authentication needs attention','Allocation returned an authentication/access error. No measurements started.','blocked')
            return 1
        if attempt or a.await_existing:
            progress('Waiting to retry Colab allocation','No confirmed allocation. Retrying the same tracked request after a five-minute pause; no measurements started.')
            # A single relative sleep hides liveness and can extend across host
            # sleep. Anchor the pause to the provider failure's wall-clock time.
            failed_at=last.get('utc') if last else None
            try:
                retry_at=datetime.fromisoformat(failed_at).timestamp()+300
            except (ValueError,TypeError):
                retry_at=time.time()+300
            while time.time()<retry_at:
                remaining=max(0,int(retry_at-time.time()))
                progress('Colab allocation retry countdown',str(remaining)+' seconds until the next tracked request; no measurements started.')
                time.sleep(min(30,max(0,retry_at-time.time())))
        command=[a.controller_python,str(run/'allocation-tools/allocate_llm_diagnostic_v001.py'),'--session',run.name,'--journal-root',str(ROOT),'--events',str(events)]
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (run/f'provisioning-retry-v003-{stamp}-{attempt:02d}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1300)
        rows=records(events); last=rows[-1] if rows else {}
        if last.get('message','').startswith(('Existing assignment','Session already exists')):
            progress('Allocation reconciliation needs attention','Colab lists an allocation that has no confirmed owned receipt. No duplicate allocation or benchmark was started.','blocked')
            return 1
    endpoint=owned()
    if not endpoint:
        progress('Colab capacity wait exhausted','No confirmed allocation after the bounded retry window; all requests retained. No benchmark started.','blocked')
        return 1
    progress('v6e assigned; starting Gemma qualification','Confirmed owned allocation. Running only matched-operation numerical diagnosis; retain this TPU for repairs, including after errors.')
    command=[a.controller_python,str(controller),'--run',str(run),'--controller-python',a.controller_python,'--allocated-endpoint',endpoint]
    (run/'controller-launch.json').write_text(json.dumps(dict(command=command,utc=datetime.now(timezone.utc).isoformat(),source_sha256=manifest['source_sha256'],controller_override=override),indent=2)+'\n')
    with (run/'controller.log').open('x') as log:
        code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    if (run/'retained-allocation.json').exists():
        read_command=[a.controller_python,str(run/'source/runtime/colab_control_v003.py'),'list']
        while True:
            try:
                check=subprocess.run(read_command,capture_output=True,text=True,timeout=90)
            except (subprocess.TimeoutExpired,OSError):
                check=subprocess.CompletedProcess(read_command,124,stdout='')
            snapshots=[]
            for line in check.stdout.splitlines():
                try:
                    value=json.loads(line)
                    if value.get('kind')=='allocations':snapshots.append(value)
                except ValueError:pass
            present=any(item.get('endpoint')==endpoint for row in snapshots for item in row['assignments'])
            record=dict(utc=datetime.now(timezone.utc).isoformat(),endpoint=endpoint,
                state='allocated' if present else 'absent' if snapshots and check.returncode==0 else 'unknown',
                release_attempted=False)
            with (run/'retention-watch.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
            if record['state']=='absent':break
            time.sleep(60)
    return code


if __name__=='__main__': raise SystemExit(main())
