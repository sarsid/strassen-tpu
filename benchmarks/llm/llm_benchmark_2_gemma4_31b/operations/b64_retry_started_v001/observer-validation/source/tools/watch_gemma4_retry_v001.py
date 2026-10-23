"""Observe an already-launched B64 retry; never allocate, launch or release."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from run_gemma4_batch_v004 import control_command, save_stage_exports


def snapshot_from(records):
    result = None
    for record in records:
        if record.get('kind') != 'remote_stream': continue
        for line in record.get('text', '').splitlines():
            try: value = json.loads(line)
            except ValueError: continue
            if isinstance(value,dict) and value.get('kind') == 'gemma4_batch_snapshot': result=value
    return result


def require_owned(records, spec):
    matches=[item for row in records if row.get('kind')=='allocations'
             for item in row['assignments'] if item.get('endpoint')==spec['endpoint']
             and item.get('name')==spec['session'] and item.get('accelerator')=='V6E1']
    if len(matches)!=1:raise RuntimeError('Expected exactly one matching retained v6e assignment')


def unpack_verified(package, expected_sha, destination):
    if hashlib.sha256(package.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('Archive integrity mismatch')
    if not destination.exists():
        with tarfile.open(package) as handle:handle.extractall(destination,filter='data')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--controller-python',required=True);args=parser.parse_args()
    run=args.run.resolve();root=run.parents[1];source=run/'source'
    spec=json.loads((source/'configs/gemma4_b64_metric_retry_v001.json').read_text())
    if run.name!=spec['run_id'] or run.parent.name!='runs':raise ValueError('Unexpected retry run')
    lock=(run/'observer.lock').open('ab');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');control=run/'observer-control'/stamp;control.mkdir(parents=True)
    cli=[args.controller_python,str(source/'runtime/colab_control_v003.py')]
    common=['--session',spec['session'],'--expect-endpoint',spec['endpoint']]
    receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),session=spec['session'],endpoint=spec['endpoint'],ownership_origin=spec['predecessor_run'],allocation_retained=True,release_attempted=False)
    def call(name,argv,timeout=180):return control_command(control,name,cli+argv,timeout)
    def commit(paths,message):
        relative=[str(p.relative_to(root)) for p in paths]
        subprocess.run(['git','add','--',*relative],cwd=root,check=True)
        changed=subprocess.run(['git','diff','--quiet','HEAD','--',*relative],cwd=root)
        if changed.returncode:subprocess.run(['git','commit','--only','-m',message,'--',*relative],cwd=root,check=True)
    def archive(name,ready,destination):
        expected='/content/Strassen_MM_Focus/runs/'+run.name+('/b64-results.tar.gz' if name=='batch' else '.tar.gz')
        if ready['path']!=expected:raise ValueError('Unexpected remote archive path')
        destination.mkdir(parents=True,exist_ok=True);package=destination/'results.tar.gz'
        if not package.exists():
            for attempt in range(3):
                candidate=destination/f'results-{stamp}-{attempt}.tar.gz'
                code,_=call(f'download-{name}-{attempt}',['download',*common,'--remote',ready['path'],'--local',str(candidate)],300)
                if not code and hashlib.sha256(candidate.read_bytes()).hexdigest()==ready['sha256']:
                    candidate.rename(package);break
            else:raise RuntimeError('Archive download could not be verified')
        unpack_verified(package,ready['sha256'],destination/'artifacts')
        receipt_file=destination/'receipt.json';content=json.dumps(ready,indent=2)+'\n'
        if receipt_file.exists():
            if receipt_file.read_text()!=content:raise ValueError('Archive receipt changed')
        else:receipt_file.write_text(content)
        commit([destination],'Preserve Gemma B64 retry '+name+' archive')
    try:
        code,records=call('verify-allocation',['list'])
        if code:raise RuntimeError('Allocation verification failed')
        require_owned(records,spec)
        unknown_since=None;empty=0;active_pointer_set=False;last_stage=None
        for iteration in range(1440):
            code,records=call(f'poll-{iteration:04d}',['exec-file',*common,'--file',str(source/'runtime/poll_gemma4_batch_v001.py'),'--timeout','60','--script-arg=--run-id','--script-arg='+run.name],120)
            snapshot=snapshot_from(records)
            if snapshot is None:
                if unknown_since is None:unknown_since=time.monotonic()
                if time.monotonic()-unknown_since>=5400:raise RuntimeError('Ninety minutes without a valid retry snapshot; TPU retained')
                time.sleep(30);continue
            if not snapshot.get('status.json'):
                empty+=1
                if empty>=3:raise RuntimeError('No retry worker status after three successful polls; TPU retained')
                time.sleep(30);continue
            unknown_since=None;empty=0;snapshot['observed_utc']=datetime.now(timezone.utc).isoformat()
            temp=run/'latest.json.tmp';temp.write_text(json.dumps(snapshot,indent=2)+'\n');temp.replace(run/'latest.json')
            state=snapshot['status.json']
            if not active_pointer_set:
                pointer=root/'status/llm_benchmark_active_v001.json'
                value=dict(run=str(run.relative_to(root)),title='Gemma4-31B · B64 recovery · five other batches passed',model=spec['model'],batches=[64],sequence=1024,links=[])
                tmp=pointer.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(pointer);active_pointer_set=True
            checkpoints=save_stage_exports(run,snapshot)
            if checkpoints:commit(checkpoints,'Checkpoint completed Gemma B64 retry stages')
            stage=state.get('active',state.get('status'))
            if stage!=last_stage:
                with (run/'dashboard-events.jsonl').open('a') as f:f.write(json.dumps(dict(utc=snapshot['observed_utc'],title=stage,detail='Saved-profile B64 retry; original five batches passed; same TPU retained'))+'\n')
                subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state','working','--title','Gemma4 B64 recovery: '+str(stage),'--detail','The saved-profile retry is on the original retained TPU. No retuning or allocation occurred.','--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
                last_stage=stage
            if 'archive-ready.json' in snapshot:
                for batch in state.get('batch_archives',[]):
                    archive('batch',batch,root/'results/v6e/gemma4_parent_batch_20261016_v003'/run.name/'b64')
                archive('full',snapshot['archive-ready.json'],run/'remote')
                receipt.update(status=state.get('status'),remote_summary=state,result_sha256=snapshot['archive-ready.json']['sha256']);break
            time.sleep(30)
        else:raise TimeoutError('Retry observer exceeded twelve hours; TPU retained')
    except BaseException as error:receipt.update(status='observer_failed',error_type=type(error).__name__,error=str(error))
    finally:
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
        (control/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (run/'observer-completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        commit([run],'Archive Gemma B64 retry observation and retained disposition')
        lock.close()
    print(json.dumps(receipt),flush=True)
    return 0 if receipt.get('status')=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
