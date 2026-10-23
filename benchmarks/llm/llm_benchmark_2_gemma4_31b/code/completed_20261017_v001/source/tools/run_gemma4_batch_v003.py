"""Gemma controller: standalone bundle launch and explicit missing-worker failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tarfile
import time

def control_command(out,name,argv,timeout=240):
    """Return timed-out RPCs as failed observations, preserving partial output.

    subprocess.run kills/waits for its local client on timeout. A read-only
    polling timeout is not evidence that the separate remote worker failed.
    The controller's bounded reconnect loop decides when to stop waiting.
    """
    path=out/(name+'.log')
    with path.open('x') as log:
        try:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
            log.write('\n'+json.dumps(dict(kind='control_command_timeout',operation=name,
                timeout_seconds=timeout,remote_worker_state='unknown',
                utc=datetime.now(timezone.utc).isoformat()))+'\n')
            log.flush()
    records=[]
    for line in path.read_text().splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    return code,records


def save_stage_exports(run,snapshot):
    """Persist final small JSON artifacts from successful stages immutably."""
    saved=[]
    for stage in snapshot.get('stage_exports',[]):
        name=stage['name']
        if not re.fullmatch(r'(b(1|4|8|16|32|64)-(tune|qualify|confirm|quality|prefill)|oracle)-a[01]',name):
            raise ValueError('Unexpected checkpoint stage name')
        for filename,payload in stage['files'].items():
            if filename not in ('profile.json','summary.json'):
                raise ValueError('Unexpected stage artifact')
            content=payload['content_utf8'].encode('utf8')
            if len(content)>2*1024**2 or hashlib.sha256(content).hexdigest()!=payload['sha256']:
                raise ValueError('Stage artifact failed integrity check')
            json.loads(content)
            path=run/'checkpoints'/name/filename
            if path.exists():
                if path.read_bytes()!=content:raise ValueError('Final stage checkpoint changed')
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(content)
            saved.append(path)
    return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--controller-python',required=True)
    p.add_argument('--allocated-endpoint',required=True);a=p.parse_args()
    run=a.run.resolve();source=run/'source';out=run/'control';out.mkdir()
    root=Path(__file__).resolve().parents[1]
    controller=[a.controller_python,str(source/'runtime/colab_control_v003.py')]
    session=run.name[:90]
    def command(name,argv,timeout=240):
        return control_command(out,name,argv,timeout)
    receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),session=session,status='verifying_allocation')
    endpoint=a.allocated_endpoint
    events=[json.loads(line) for line in (run/'allocation-events.jsonl').read_text().splitlines()]
    owned=[r for r in events if r.get('kind')=='allocation' and r.get('created')
           and r.get('session')==session and r.get('endpoint')==endpoint]
    if len(owned)!=1:
        raise RuntimeError('Require exactly one owned allocation receipt for this run')
    code,records=command('verify-allocation',controller+['list'])
    active=[item for row in records if row.get('kind')=='allocations'
            for item in row['assignments'] if item.get('endpoint')==endpoint
            and item.get('name')==session and item.get('accelerator')=='V6E1']
    if code or len(active)!=1:
        raise RuntimeError('Owned v6e allocation is not live in the saved session')
    common=['--session',session,'--expect-endpoint',endpoint]
    receipt.update(endpoint=endpoint,status='running')
    (run/'progress.json').write_text(json.dumps(dict(status='running',allocation='v6e allocated',utc=datetime.now(timezone.utc).isoformat(),stage_label='Starting remote worker'))+'\n')
    (run/'allocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    plan=json.loads((source/'configs/gemma4_parent_batch_v002.json').read_text())
    previous=None
    downloaded=set()
    try:
        code,_=command('bootstrap',controller+['exec-file',*common,'--file',str(source/'runtime/bootstrap_parent_replication_v001.py'),'--timeout','60'])
        if code:raise RuntimeError('Remote bootstrap failed')
        remote='/content/Strassen_MM_Focus/'+run.name+'-source.tar'
        code,_=command('upload',controller+['upload',*common,'--local',str(run/'source.tar'),'--remote',remote])
        if code:raise RuntimeError('Source upload failed')
        code,launch_records=command('launch',controller+['exec-file',*common,'--file',str(source/'runtime/launch_gemma4_bundle_v003.py'),'--timeout','90',
            '--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),
            '--script-arg=--run-id','--script-arg='+run.name])
        # Launch response can be lost. Poll before deciding whether work exists.
        deadline=time.monotonic()+172800
        failures=0; unavailable_since=None; empty_snapshots=0
        definite_launch_failure=any(r.get('kind')=='remote_error' or (r.get('kind')=='execution_finished' and r.get('success') is False) for r in launch_records)
        for number in range(5760):
            if time.monotonic()>deadline:raise TimeoutError('Parent-only reproduction exceeded 48 hours')
            code,records=command('poll-'+str(number).zfill(3),controller+['exec-file',*common,'--file',str(source/'runtime/poll_gemma4_batch_v001.py'),'--timeout','60',
                '--script-arg=--run-id','--script-arg='+run.name],120)
            snapshot=None
            for record in records:
                if record.get('kind')=='remote_stream':
                    for line in record.get('text','').splitlines():
                        try:
                            value=json.loads(line)
                            if value.get('kind')=='gemma4_batch_snapshot':snapshot=value
                        except ValueError:pass
            if snapshot is not None and not snapshot.get('status.json'):
                empty_snapshots+=1
                if definite_launch_failure:
                    raise RuntimeError('Remote launcher failed before a worker status existed; see launch.log')
                if empty_snapshots>=3:
                    raise RuntimeError('Three successful polls found no worker status; refusing to label the experiment running')
                snapshot=None
            elif snapshot is not None:
                empty_snapshots=0
            if snapshot is None:
                absent=any(row.get('kind')=='controller_error' and row.get('message')=='The saved endpoint is not currently allocated' for row in records)
                if absent:
                    verify_code,verify_rows=command('verify-runtime-loss-'+str(number),controller+['list'])
                    allocation_rows=[row for row in verify_rows if row.get('kind')=='allocations']
                    if verify_code==0 and allocation_rows and all(not any(item.get('endpoint')==endpoint for item in row['assignments']) for row in allocation_rows):
                        receipt.update(status='runtime_lost',allocation_verified_absent=True,release_not_needed=True,
                            error='Colab confirms the owned endpoint is absent; no result completeness is inferred.')
                        break
                failures+=1
                if unavailable_since is None: unavailable_since=time.monotonic()
                unavailable_seconds=time.monotonic()-unavailable_since
                subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state','working',
                    '--title','Gemma4 monitoring reconnecting; TPU retained',
                    '--detail',f'Polling attempt {number} returned no snapshot (exit {code}); {failures} consecutive failures over {unavailable_seconds:.0f} seconds. The remote worker may still be running. Retrying without relaunching or releasing it.',
                    '--evidence',str((out/('poll-'+str(number).zfill(3)+'.log')).relative_to(root))],check=False,stdout=subprocess.DEVNULL)
                if unavailable_seconds>=5400:raise RuntimeError('Ninety minutes continuously without a remote snapshot')
            else:
                failures=0; unavailable_since=None
                snapshot['observed_utc']=datetime.now(timezone.utc).isoformat()
                tmp=run/'latest.json.tmp';tmp.write_text(json.dumps(snapshot,indent=2)+'\n');tmp.replace(run/'latest.json')
                checkpoint_paths=save_stage_exports(run,snapshot)
                if checkpoint_paths:
                    paths=[str(path.relative_to(root)) for path in checkpoint_paths]
                    subprocess.run(['git','add','--',*paths],cwd=root,check=True)
                    subprocess.run(['git','commit','--only','-m','Checkpoint completed Gemma4 stages','--',*paths],cwd=root,check=True)
                state=snapshot.get('status.json',{})
                last={}
                for tail in snapshot.get('tails',{}).values():
                    for line in tail.splitlines():
                        try:
                            row=json.loads(line)
                            if isinstance(row,dict) and row.get('kind'):last=row
                        except ValueError:pass
                current=(state,last)
                if current!=previous:
                    previous=current
                    with (run/'dashboard-events.jsonl').open('a') as f:
                        f.write(json.dumps(dict(utc=snapshot['observed_utc'],title=state.get('active',state.get('status','Starting')),detail=str({k:last[k] for k in ('kind','arm','site','tile','layer','segment') if k in last})))+'\n')
                    detail='Stage '+state.get('active',state.get('status','starting'))+'; '+str(len(state.get('batches',[])))+'/'+str(len(plan['batches']))+' batches archived. '+str({k:last[k] for k in ('kind','arm','site','tile','layer','segment','scored_positions','candidate_mib') if k in last})
                    subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state','working',
                        '--title','Gemma4-31B four-arm benchmark','--detail',detail,'--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
                for ready_batch in state.get('batch_archives',[]):
                    batch=ready_batch['batch']
                    if batch in downloaded:continue
                    dest=root/plan['result_root']/run.name/('b'+str(batch))
                    dest.mkdir(parents=True,exist_ok=True)
                    package=dest/'results.tar.gz'
                    code,_=command('download-b'+str(batch),controller+['download',*common,'--remote',ready_batch['path'],'--local',str(package)],300)
                    if code or hashlib.sha256(package.read_bytes()).hexdigest()!=ready_batch['sha256']:raise RuntimeError('Unverified batch export')
                    with tarfile.open(package) as t:t.extractall(dest/'artifacts',filter='data')
                    (dest/'receipt.json').write_text(json.dumps(ready_batch,indent=2)+'\n')
                    relative=str(dest.relative_to(root))
                    subprocess.run(['git','add','--',relative],cwd=root,check=True)
                    subprocess.run(['git','commit','--only','-m','Archive Gemma4-31B B'+str(batch)+' batch results','--',relative],cwd=root,check=True)
                    downloaded.add(batch)
                if 'archive-ready.json' in snapshot:
                    ready=snapshot['archive-ready.json']
                    for attempt in range(3):
                        package=run/('remote-results-'+str(attempt)+'.tar.gz')
                        code,_=command('download-'+str(attempt),controller+['download',*common,'--remote',ready['path'],'--local',str(package)],300)
                        if not code and hashlib.sha256(package.read_bytes()).hexdigest()==ready['sha256']:break
                    else:raise RuntimeError('Result download not verified')
                    with tarfile.open(package) as t:t.extractall(run/'remote',filter='data')
                    receipt.update(status=state.get('status'),remote_summary=state,result_sha256=ready['sha256'])
                    break
            time.sleep(30)
        else:raise TimeoutError('Poll budget exhausted')
    except BaseException as error:
        receipt.update(status='controller_failed',error_type=type(error).__name__,error=str(error))
    finally:
        if receipt.get('allocation_verified_absent'):
            code=0
        else:
            code,_=command('release',[a.controller_python,str(source/'runtime/release_allocation_v002.py'),*common])
        receipt.update(release_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
        (run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    relative=str(run.relative_to(root))
    subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state','complete' if receipt.get('release_exit_code')==0 else 'blocked','--title','Gemma4 batch worker finished','--detail','Status '+str(receipt['status'])+'; release exit code '+str(receipt.get('release_exit_code'))+'; results preserved in '+relative,'--evidence',relative+'/completion.json'],check=False)
    subprocess.run(['git','add','--',relative,'status/work_progress_v001.jsonl'],cwd=root,check=True)
    subprocess.run(['git','commit','--only','-m','Archive Gemma4 batch study completion and release','--',relative,'status/work_progress_v001.jsonl'],cwd=root,check=True)
    return 0 if receipt['status'] in ('completed','completed_with_failures') and receipt['release_exit_code']==0 else 1

if __name__=='__main__':raise SystemExit(main())
