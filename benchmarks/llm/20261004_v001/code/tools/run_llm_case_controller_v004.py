"""Recoverable queued case controller. One endpoint, checkpointed arm groups."""
import argparse,fcntl,hashlib,json,os,subprocess,tarfile,time,uuid
from pathlib import Path
from llm_queue_state_v001 import read,save,utc,merge


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--controller-python',required=True);p.add_argument('--recover',action='store_true');a=p.parse_args()
    run=a.run.resolve();source=Path(__file__).resolve().parents[1]
    root=Path(os.environ['STRASSEN_PROJECT_ROOT']);out=run/'control';out.mkdir(exist_ok=True)
    lock=(run/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=read(run/'job.json');session=run.name[:90];controller=[a.controller_python,str(source/'runtime/colab_control_v003.py')]
    def command(name,argv,timeout=240):
        logpath=out/(name+'-'+uuid.uuid4().hex[:8]+'.log')
        try:
            with logpath.open('x') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            code=result.returncode
        except subprocess.TimeoutExpired:code=124
        records=[]
        for line in logpath.read_text().splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
        return code,records
    def journal(title,detail,state='working'):
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state',state,'--title',title,'--detail',detail,'--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
    receipt=read(run/'allocation.json') if a.recover else None
    if not receipt:
        receipt=dict(started_utc=utc(),session=session,status='allocating')
        code,records=command('allocate',[a.controller_python,str(source/'runtime/allocate_llm_diagnostic_v001.py'),'--session',session,'--journal-root',str(root),'--events',str(run/'allocation-events.jsonl')],660)
        found=[r for r in records if r.get('kind')=='allocation' and r.get('created')]
        if code or len(found)!=1:
            receipt.update(status='allocation_failed',records=records,finished_utc=utc());save(run/'completion.json',receipt);return 1
        receipt.update(endpoint=found[0]['endpoint'],status='running');save(run/'allocation.json',receipt)
    endpoint=receipt['endpoint'];common=['--session',session,'--expect-endpoint',endpoint]
    def promoted(ready,label,final=False):
        old=read(run/(label+'-receipt.json'))
        if old and old.get('sha256')==ready['sha256']:return
        for attempt in range(3):
            package=run/(label+'-'+uuid.uuid4().hex[:8]+'.tar.gz')
            code,_=command('download-'+label,controller+['download',*common,'--remote',ready['path'],'--local',str(package)],600)
            if not code and hashlib.sha256(package.read_bytes()).hexdigest()==ready['sha256']:break
        else:raise RuntimeError('Result download not verified')
        destination=run/('remote' if final else label)
        with tarfile.open(package) as t:t.extractall(destination,filter='data')
        completed=destination/run.name/'artifacts/main-prefill/completed_entries.jsonl' if final else destination/'main-prefill/completed_entries.jsonl'
        if completed.exists():merge(run.parent/'completed_entries.jsonl',[json.loads(line) for line in completed.read_text().splitlines()],job,run.name)
        save(run/(label+'-receipt.json'),dict(sha256=ready['sha256'],utc=utc(),package=package.name))
    def snapshot():
        code,records=command('poll',controller+['exec-file',*common,'--file',str(source/'runtime/poll_llm_queue_v001.py'),'--timeout','60','--script-arg=--run-id','--script-arg='+run.name],120)
        for record in records:
            if record.get('kind')=='remote_stream':
                for line in record.get('text','').splitlines():
                    try:
                        row=json.loads(line)
                        if row.get('kind')=='parent_replication_snapshot':return row
                    except (ValueError,AttributeError):pass
        return None
    try:
        if not a.recover:
            for _ in range(3):
                code,_=command('bootstrap',controller+['exec-file',*common,'--file',str(source/'runtime/bootstrap_parent_replication_v001.py'),'--timeout','60'])
                if not code:break
                time.sleep(5)
            if code:raise RuntimeError('Remote bootstrap failed')
            if job['model'].startswith('gemma'):
                code,auth=command('stage-auth',[a.controller_python,str(source/'tools/stage_mlsys_auth_v002.py'),'--session',session,'--endpoint',endpoint,'--private-root','/content/Strassen_MM_Focus/.runtime_private/'+run.name+'/model','--hardware','v6e'])
                if code or not any(r.get('existing_hf_credential_available') for r in auth):
                    receipt['failure_class']='model_access';raise RuntimeError('Authorized Gemma credential unavailable')
            remote='/content/Strassen_MM_Focus/'+run.name+'-source.tar'
            code,_=command('upload',controller+['upload',*common,'--local',str(run/'source.tar'),'--remote',remote])
            if code:raise RuntimeError('Source upload failed')
            command('launch',controller+['exec-file',*common,'--file',str(source/'runtime/llm_case_remote_v002.py'),'--timeout','90','--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),'--script-arg=--run-id','--script-arg='+run.name])
        deadline=time.time()+43200
        failures=0;empty=0;previous=None
        while time.time()<deadline:
            data=snapshot()
            if data is None:
                failures+=1
                if failures>=20:raise RuntimeError('Twenty unavailable remote snapshots; preserve checkpoints and release')
            else:
                failures=0;save(run/'latest.json',data);state=data.get('status.json',{})
                empty=0 if state else empty+1
                if empty>=5:raise RuntimeError('No remote worker after five observations')
                marker=(state.get('active'),str(data.get('tails',{}))[-1000:])
                if marker!=previous:
                    previous=marker;journal(job['model']+': queued full-model comparison','B'+str(job['batch'])+' × S'+str(job['sequence'])+'; stage '+state.get('active',state.get('status','launching'))+'. Full sweep advances automatically after archiving and release.')
                checkpoint=data.get('checkpoint-ready.json')
                if checkpoint:promoted(checkpoint,'checkpoint-'+str(checkpoint['completed']))
                if 'archive-ready.json' in data:
                    ready=data['archive-ready.json'];promoted(ready,'final',True)
                    receipt.update(status=state.get('status'),remote_summary=state,result_sha256=ready['sha256'])
                    break
            time.sleep(30)
        else:raise TimeoutError('Case exceeded twelve-hour controller budget')
    except BaseException as error:
        receipt.update(status='controller_failed',error_type=type(error).__name__,error=str(error))
    finally:
        code=1
        for _ in range(3):
            code,records=command('release',[a.controller_python,str(source/'runtime/release_allocation_v003.py'),*common])
            if code==0:break
            time.sleep(10)
        receipt.update(release_exit_code=code,finished_utc=utc());save(run/'completion.json',receipt)
        print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='completed' and code==0 else 1

if __name__=='__main__':raise SystemExit(main())
