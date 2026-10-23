"""Checkpoint/observe the running study and release only after verified success."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COUNTS={'mmlu':114,'ppl':32,'gsm8k':32}


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--session',required=True);p.add_argument('--endpoint',required=True)
    p.add_argument('--validate-cli',action='store_true')
    return p.parse_args(argv)


def main():
    a=parse_args()
    if a.validate_cli:
        print(json.dumps(dict(run_id=a.run_id,session=a.session,endpoint=a.endpoint)));return
    root=ROOT/'runs'/a.run_id
    if root.parent!=ROOT/'runs' or not root.exists():raise ValueError('Expected existing local run')
    lock=(root/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(root/'observer-ready-v002.json',dict(pid=os.getpid(),endpoint=a.endpoint,run_id=a.run_id,ready_utc=datetime.now(timezone.utc).isoformat()))
    receipts=root/'observer-receipts-v002';receipts.mkdir(exist_ok=True)
    remote='/content/Strassen_MM_Focus/runs/'+a.run_id
    controller=[sys.executable,str(ROOT/'runtime/colab_control_v003.py')]
    identity=['--session',a.session,'--expect-endpoint',a.endpoint]
    def request(operation,name,extra=()):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');path=receipts/(stamp+'-'+Path(name).name+'.jsonl')
        command=controller+[operation,*identity,'--remote',name,*extra]
        with path.open('x') as output:
            try:code=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=100).returncode
            except subprocess.TimeoutExpired:return None
        if code:return None
        try:return json.loads(path.read_text().splitlines()[-1])
        except (ValueError,IndexError):return None
    def read(name):
        value=request('read',remote+'/'+name,['--max-bytes','1048576'])
        try:return json.loads(value['text']) if value and not value.get('truncated') else None
        except (ValueError,KeyError):return None
    def download(name,local):
        if local.exists():return True
        local.parent.mkdir(parents=True,exist_ok=True)
        return request('download',remote+'/'+name,['--local',str(local)]) is not None
    def verified_archive(name,local):
        ready=read(name)
        if not ready:return False
        path=ready['path']
        if not path.startswith(remote+'/'):raise ValueError('Unexpected archive path')
        if not download(path[len(remote)+1:],local):return False
        if local.stat().st_size!=ready['bytes'] or hashlib.sha256(local.read_bytes()).hexdigest()!=ready['sha256']:raise ValueError('Archive hash mismatch')
        atomic(local.with_suffix('.verified.json'),ready);return True
    previous=None;verified=set();failures=0
    while True:
        state=read('status.json');now=datetime.now(timezone.utc).isoformat()
        if state is None:
            failures+=1;atomic(root/'live-status.json',dict(observed_utc=now,observation='connection unavailable',remote_state='unknown'))
            if failures>=5:
                try:
                    result=subprocess.run(controller+['list'],capture_output=True,text=True,timeout=100)
                    assignments=json.loads(result.stdout)['assignments']
                except (ValueError,KeyError,subprocess.TimeoutExpired,OSError):assignments=None
                if assignments is not None and not any(x['endpoint']==a.endpoint for x in assignments):
                    atomic(root/'observer-completion-v002.json',dict(status='runtime_lost',observed_utc=now,completed_archives=sorted(verified),remaining_work=True))
                    return
            time.sleep(30);continue
        failures=0;progress=read('progress.json') or {};model=state.get('model')
        done=state.get('test_evaluations_completed',0)
        if progress.get('model')==model and model not in state.get('completed_models',[]):done+=len(progress.get('completed_tasks',[]))
        snapshot=dict(observed_utc=now,remote_state=state,progress=progress,test_evaluations_completed=done,test_evaluations_total=9)
        atomic(root/'live-status.json',snapshot)
        if model and progress.get('model')==model:
            for task,total in COUNTS.items():
                completed=total if task in progress.get('completed_tasks',[]) else progress.get('completed_examples',0) if progress.get('task')==task else 0
                for first in range(0,completed,8):
                    if min(first+8,total)>completed:continue
                    name=f'artifacts/{model}/batches/{task}-{first//8:03d}.json'
                    download(name,root/'checkpoints'/model/'batches'/f'{task}-{first//8:03d}.json')
            if done or progress.get('task'):
                for name in ('manifest.json','execution-contract.json'):
                    download(f'artifacts/{model}/{name}',root/'checkpoints'/model/name)
        for model_name in state.get('completed_models',[]):
            if model_name not in verified and verified_archive(model_name+'.tar.ready.json',root/(model_name+'.tar.gz')):verified.add(model_name)
        event=progress.get('latest_event',{});key=(state.get('status'),model,progress.get('task'),progress.get('batch'),event.get('kind'),event.get('step'),event.get('layer'),done)
        if key!=previous:
            detail=dict(model=model,stage=state.get('stage'),task=progress.get('task'),batch=progress.get('batch'),latest_event=event)
            subprocess.run([sys.executable,str(ROOT/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state',state.get('status','unknown'),
                '--title','Paired downstream accuracy study','--detail',json.dumps(detail),'--completed',str(done),'--total','9','--evidence',str((root/'live-status.json').relative_to(ROOT))],stdout=subprocess.DEVNULL,check=False)
            previous=key
        if state.get('status') in ('completed','failed') and verified_archive('archive-ready.json',root/'all-artifacts.tar.gz'):
            if state['status']=='completed':
                if done!=9 or len(verified)!=3:raise ValueError('Refuse release before all three verified model archives')
                command=[sys.executable,str(ROOT/'runtime/release_allocation_v003.py'),*identity]
                with (root/'release-receipt-v002.jsonl').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=100).returncode
                if code:raise RuntimeError('Final archive verified but release needs reconciliation')
            atomic(root/'observer-completion-v002.json',dict(status=state['status'],finished_utc=now,archive_verified=True,completed_models=sorted(verified),tpu_released=state['status']=='completed',remaining_work=state['status']!='completed'))
            return
        time.sleep(30)


if __name__=='__main__':main()
