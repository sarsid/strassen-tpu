"""Retain the existing TPU briefly after the final stage, then restore release."""
import argparse,hashlib,json,os,shlex,signal,subprocess,tarfile,time
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--main-run',type=Path,required=True)
    p.add_argument('--main-controller-pid',type=int,required=True);p.add_argument('--controller-python',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve();mainrun=a.main_run.resolve();source=run/'source'
    out=run/'control';out.mkdir();allocation=json.loads((mainrun/'allocation.json').read_text())
    common=['--session',allocation['session'],'--expect-endpoint',allocation['endpoint']]
    controller=[a.controller_python,str(source/'runtime/colab_control_v003.py')]
    receipt=dict(status='waiting-for-final-main-stage',started_utc=datetime.now(timezone.utc).isoformat(),main_run=str(mainrun),
                 main_controller_pid=a.main_controller_pid,endpoint=allocation['endpoint'],session=allocation['session'])
    (run/'allocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    suspended=False
    def command(name,argv,timeout=180):
        with (out/(name+'.log')).open('x') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        rows=[]
        for line in (out/(name+'.log')).read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
        return r.returncode,rows
    def identity():
        text=subprocess.check_output(['ps','-p',str(a.main_controller_pid),'-o','command='],text=True).strip()
        argv=shlex.split(text)
        return len(argv)>1 and argv[1]=='tools/run_parent_comparison_v003.py' and str(mainrun.relative_to(root)) in argv
    try:
        deadline=time.monotonic()+7200
        while True:
            if (mainrun/'completion.json').exists():raise RuntimeError('Main controller already completed; no runtime is retained')
            if time.monotonic()>deadline:raise TimeoutError('Final main stage was not reached')
            try:state=json.loads((mainrun/'latest.json').read_text()).get('status.json',{})
            except (ValueError,FileNotFoundError):time.sleep(5);continue
            if state.get('active')=='s2-float32-quality':break
            time.sleep(15)
        if not identity():raise RuntimeError('Main controller identity mismatch')
        os.kill(a.main_controller_pid,signal.SIGSTOP);suspended=True
        receipt.update(status='audit-running',retained_utc=datetime.now(timezone.utc).isoformat())
        (run/'retention.json').write_text(json.dumps(receipt,indent=2)+'\n')
        remote='/content/Strassen_MM_Focus/'+run.name+'-source.tar'
        code,_=command('upload',controller+['upload',*common,'--local',str(run/'source.tar'),'--remote',remote])
        if code:raise RuntimeError('Supplemental source upload failed')
        command('launch',controller+['exec-file',*common,'--file',str(source/'runtime/parent_budget_remote_v002.py'),'--timeout','90',
            '--script-arg=--archive','--script-arg='+remote,'--script-arg=--sha256','--script-arg='+hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),
            '--script-arg=--run-id','--script-arg='+run.name,'--script-arg=--main-run-id','--script-arg='+mainrun.name])
        deadline=time.monotonic()+3000;failures=0;previous=None
        for number in range(100):
            if time.monotonic()>deadline:raise TimeoutError('Supplemental audit exceeded 50 minutes')
            code,rows=command('poll-'+str(number).zfill(3),controller+['exec-file',*common,'--file',str(source/'runtime/poll_parent_replication_v001.py'),'--timeout','60',
                                  '--script-arg=--run-id','--script-arg='+run.name],120)
            snapshot=None
            for row in rows:
                if row.get('kind')=='remote_stream':
                    for line in row.get('text','').splitlines():
                        try:value=json.loads(line)
                        except ValueError:continue
                        if value.get('kind')=='parent_replication_snapshot':snapshot=value
            if snapshot is None:
                failures+=1
                if failures>=5:raise RuntimeError('Five missing supplemental snapshots')
            else:
                failures=0;(run/'latest.json').write_text(json.dumps(snapshot,indent=2)+'\n');state=snapshot.get('status.json',{})
                if state!=previous:
                    previous=state
                    subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state','working',
                        '--title','Down-projection v6e memory-cap audit','--detail','Stage '+state.get('active',state.get('status','starting'))+'; the main frozen study remains unchanged.',
                        '--evidence',str((run/'latest.json').relative_to(root))],stdout=subprocess.DEVNULL,check=False)
                if 'archive-ready.json' in snapshot:
                    ready=snapshot['archive-ready.json']
                    for attempt in range(3):
                        package=run/('remote-results-'+str(attempt)+'.tar.gz')
                        code,_=command('download-'+str(attempt),controller+['download',*common,'--remote',ready['path'],'--local',str(package)],300)
                        if not code and hashlib.sha256(package.read_bytes()).hexdigest()==ready['sha256']:break
                    else:raise RuntimeError('Supplemental download not verified')
                    with tarfile.open(package) as archive:archive.extractall(run/'remote',filter='data')
                    receipt.update(status=state['status'],remote_summary=state,result_sha256=ready['sha256']);break
            time.sleep(30)
        else:raise TimeoutError('Supplemental poll budget exhausted')
    except BaseException as error:receipt.update(status='controller_failed',error_type=type(error).__name__,error=str(error))
    finally:
        if suspended:
            try:
                if not identity():raise RuntimeError('Release controller identity changed')
                os.kill(a.main_controller_pid,signal.SIGCONT);receipt['main_controller_resumed']=True
            except BaseException as error:
                receipt['resume_error']=type(error).__name__
                code,_=command('fallback-release',[a.controller_python,str(source/'runtime/release_allocation_v002.py'),*common])
                receipt['fallback_release_exit_code']=code
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat();(run/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
