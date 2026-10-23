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
    code,records=command('allocate',[a.controller_python,str(source/'runtime/allocate_tradeoff_v002.py'),'v6e','create','--session',session])
    found=[r for r in records if r.get('kind')=='allocation' and r.get('created')]
    if code or len(found)!=1:
        receipt.update(status='allocation_failed',records=records)
        (run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        raise RuntimeError('No confirmed allocation: inspect sanitized allocation evidence')
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
        code,_=command('launch',controller+['exec-file',*common,'--file',str(source/'runtime/llm_campaign_remote_v002.py'),'--timeout','90',
            '--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),
            '--script-arg=--run-id','--script-arg='+run.name])
        # Launch response can be lost. Poll before deciding whether work exists.
        deadline=time.monotonic()+9000
        failures=0
        for number in range(300):
            if time.monotonic()>deadline:raise TimeoutError('Reproduction exceeded 150 minutes')
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
                if state!=previous:
                    previous=state
                    detail='Stage '+state.get('active',state.get('status','starting'))+'; '+str(len(state.get('stages',[])))+' setup/qualification outcomes. Main prefill grid: 0/700 measured.'
                    subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state','working',
                        '--title','Multi-model v6e campaign: qualification and pilot','--detail',detail,'--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
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
        code,_=command('release',[a.controller_python,str(source/'runtime/release_allocation_v002.py'),*common])
        receipt.update(release_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
        (run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        if code==0:
            subprocess.run(['git','add','--',str(run.relative_to(root))],cwd=root,check=True)
            subprocess.run(['git','commit','--only','-m','Archive LLM campaign qualification and pilot','--',str(run.relative_to(root))],cwd=root,check=True)
    return 0 if receipt['status']=='completed' and receipt['release_exit_code']==0 else 1

if __name__=='__main__':raise SystemExit(main())
