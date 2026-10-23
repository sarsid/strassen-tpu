"""Matched comparison controller; frozen source, bounded polling and release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--controller-python',required=True);a=p.parse_args()
    run=a.run.resolve();source=run/'source';out=run/'control';out.mkdir()
    root=Path(__import__('os').environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
    controller=[a.controller_python,str(source/'runtime/colab_control_v003.py')]
    session=run.name[:90]
    def command(name,argv,timeout=240):
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        records=[]
        for line in (out/(name+'.log')).read_text().splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
        return result.returncode,records
    receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),session=session,status='allocating')
    for attempt in range(2):
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state','working',
            '--title','Requesting a v6e runtime','--detail','Allocation attempt '+str(attempt+1)+' of 2 for the first complete Qwen3-8B workload. No measurements have started.',
            '--evidence',str((run/'frozen.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
        try:
            code,records=command('allocate-'+str(attempt),[a.controller_python,str(source/'runtime/allocate_tradeoff_v002.py'),'v6e','create','--session',session])
        except subprocess.TimeoutExpired:
            code,records=1,[dict(kind='allocation_error',error_type='TimeoutExpired')]
        found=[r for r in records if r.get('kind')=='allocation' and r.get('created')]
        if not code and len(found)==1:break
        transient=any(r.get('error_type') in ('ReadTimeout','TimeoutExpired','ConnectionError') or r.get('http_status') in (429,500,502,503,504) for r in records)
        inspect_code,allocations=command('allocation-reconcile-'+str(attempt),controller+['list'],120)
        empty=not inspect_code and any(r.get('kind')=='allocations' and r.get('assignments')==[] for r in allocations)
        if not transient or not empty:break
    if code or len(found)!=1:
        receipt.update(status='allocation_failed',records=records,finished_utc=datetime.now(timezone.utc).isoformat())
        (run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state','blocked',
            '--title','Colab allocation failed','--detail','No confirmed TPU endpoint after bounded allocation recovery. No main measurements ran.',
            '--evidence',str((run/'completion.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
        subprocess.run(['git','add','--',str(run.relative_to(root))],cwd=root,check=True)
        subprocess.run(['git','commit','--only','-m','Archive bounded Colab allocation failure','--',str(run.relative_to(root))],cwd=root,check=True)
        return 1
    endpoint=found[0]['endpoint'];common=['--session',session,'--expect-endpoint',endpoint]
    receipt.update(endpoint=endpoint,status='running')
    (run/'allocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    previous=None
    try:
        for attempt in range(3):
            code,_=command('bootstrap-'+str(attempt),controller+['exec-file',*common,'--file',str(source/'runtime/bootstrap_parent_replication_v001.py'),'--timeout','60'])
            if not code:break
            time.sleep(5)
        if code:raise RuntimeError('Remote bootstrap failed')
        remote='/content/Strassen_MM_Focus/'+run.name+'-source.tar'
        code,_=command('upload',controller+['upload',*common,'--local',str(run/'source.tar'),'--remote',remote])
        if code:raise RuntimeError('Source upload failed')
        code,_=command('launch',controller+['exec-file',*common,'--file',str(source/'runtime/llm_case_remote_v001.py'),'--timeout','90',
            '--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),
            '--script-arg=--run-id','--script-arg='+run.name])
        # Launch response can be lost. Poll before deciding whether work exists.
        deadline=time.monotonic()+21600
        failures=0
        for number in range(720):
            if time.monotonic()>deadline:raise TimeoutError('Reproduction exceeded six hours')
            code,records=command('poll-'+str(number).zfill(3),controller+['exec-file',*common,'--file',str(source/'runtime/poll_parent_replication_v001.py'),'--timeout','60',
                '--script-arg=--run-id','--script-arg='+run.name],120)
            snapshot=None
            for record in records:
                if record.get('kind')=='remote_stream':
                    for line in record.get('text','').splitlines():
                        try:
                            value=json.loads(line)
                            if value.get('kind')=='parent_replication_snapshot':snapshot=value
                        except ValueError:pass
            if snapshot is None:
                failures+=1
                if failures>=5:raise RuntimeError('Five consecutive unavailable remote snapshots')
            else:
                failures=0
                (run/'latest.json').write_text(json.dumps(snapshot,indent=2)+'\n')
                state=snapshot.get('status.json',{})
                recent=[]
                for tail in snapshot.get('tails',{}).values():
                    for line in tail.splitlines():
                        try:
                            row=json.loads(line)
                            if isinstance(row,dict) and row.get('kind'):recent.append(row)
                        except ValueError:pass
                event=recent[-1] if recent else {}
                marker=(state,event)
                if marker!=previous:
                    previous=marker
                    detail='Stage '+state.get('active',state.get('status','starting'))+'. Latest event: '+str(event.get('kind','starting'))+' '+str(event.get('site',''))+' '+str(event.get('family',''))+' '+str(event.get('output_dtype',''))+'. The first matched workload is in progress; incomplete timings do not count as finished comparisons.'
                    subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state','working',
                        '--title','Qwen3-8B: first complete-model comparison','--detail',detail,'--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
                if 'archive-ready.json' in snapshot:
                    ready=snapshot['archive-ready.json']
                    for attempt in range(3):
                        package=run/('remote-results-'+str(attempt)+'.tar.gz')
                        code,_=command('download-'+str(attempt),controller+['download',*common,'--remote',ready['path'],'--local',str(package)],300)
                        if not code and hashlib.sha256(package.read_bytes()).hexdigest()==ready['sha256']:break
                    else:raise RuntimeError('Result download not verified')
                    with tarfile.open(package) as t:t.extractall(run/'remote',filter='data')
                    completed=run/'remote'/run.name/'artifacts/main-prefill/completed_entries.jsonl'
                    if completed.exists():
                        ledger=run.parent/'completed_entries.jsonl'
                        known={json.loads(line)['entry_id']:json.loads(line) for line in ledger.read_text().splitlines()} if ledger.exists() else {}
                        for line in completed.read_text().splitlines():
                            row=json.loads(line)
                            if row['entry_id'] in known:raise RuntimeError('Refusing to overwrite an existing completed comparison')
                            known[row['entry_id']]=dict(row,evidence_run=run.name)
                        temporary=ledger.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(row)+'\n' for row in known.values()));temporary.replace(ledger)
                    receipt.update(status=state.get('status'),remote_summary=state,result_sha256=ready['sha256'])
                    break
            time.sleep(30)
        else:raise TimeoutError('Poll budget exhausted')
    except BaseException as error:
        receipt.update(status='controller_failed',error_type=type(error).__name__,error=str(error))
    finally:
        code,_=command('release',[a.controller_python,str(source/'runtime/release_allocation_v002.py'),*common])
        receipt.update(release_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
        (run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current',
            '--state','complete' if receipt['status']=='completed' and code==0 else 'blocked',
            '--title','First complete-model workload finished' if receipt['status']=='completed' else 'Complete-model workload needs attention',
            '--detail','Outcome: '+receipt['status']+'. Release command exit code: '+str(code)+'. See archived stage evidence; the remaining main grid is not automatically complete.',
            '--next-step','Inspect the matched results and any qualification failure before extending the sweep.',
            '--evidence',str((run/'completion.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
        if code==0:
            subprocess.run(['git','add','--',str(run.relative_to(root)),*([str((run.parent/'completed_entries.jsonl').relative_to(root))] if (run.parent/'completed_entries.jsonl').exists() else [])],cwd=root,check=True)
            subprocess.run(['git','commit','--only','-m','Archive first complete-model v6e comparison','--',str(run.relative_to(root)),*([str((run.parent/'completed_entries.jsonl').relative_to(root))] if (run.parent/'completed_entries.jsonl').exists() else [])],cwd=root,check=True)
    return 0 if receipt['status']=='completed' and receipt['release_exit_code']==0 else 1

if __name__=='__main__':raise SystemExit(main())
