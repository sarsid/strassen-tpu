"""Serial, checkpointed model sessions; adopt an existing legacy case unchanged."""
import argparse,collections,fcntl,hashlib,io,json,os,shutil,subprocess,tarfile,time,uuid
from pathlib import Path
from llm_queue_state_v002 import read,save,utc,ledger,remaining,failure_action


def running(run,pid):
    if not pid:return False
    p=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True)
    return not p.stdout.strip().startswith('Z') and any(n in p.stdout for n in ('run_llm_case_controller_v005.py','run_llm_session_controller_v001.py')) and str(run) in p.stdout


def reconcile(state,rows,remote=None):
    session=state.get('active_session');current=(remote or {}).get('current_job',{}).get('job_id')
    for j in state['jobs']:
        if not remaining(j,rows):j['status']='completed'
        elif session and j['job_id'] in session['job_ids']:
            j['status']='running' if j['job_id']==(current or session['job_ids'][0]) else 'reserved'
    state['measured_comparisons']=len(rows);state['counts']=dict(collections.Counter(j['status'] for j in state['jobs']))


def finish(state,receipt,rows):
    session=state['active_session'];byid={j['job_id']:j for j in state['jobs']}
    remote=receipt.get('remote_summary',{});case_results={r['job_id']:r for r in remote.get('cases',[])}
    last=remote.get('current_job',{}).get('job_id')
    action=failure_action(receipt)
    if action=='release_blocked':state.update(status='blocked',reason='Owned TPU release unverified');return
    for jid in session['job_ids']:
        job=byid[jid];attempt=job['attempts'][-1];result=case_results.get(jid)
        complete=not remaining(job,rows)
        attempted=session.get('legacy') or result is not None or jid==last
        outcome='completed' if complete else (result or {}).get('status',receipt.get('status','failed')) if attempted else 'not_started'
        if action=='allocation_wait':outcome='allocation_failed'
        attempt.update(status=outcome,finished_utc=utc());job['status']='completed' if complete else 'queued'
        charged=sum(a.get('source_commit')==session['source_commit'] and a.get('status') not in ('not_started','allocation_failed','reserved','running') for a in job['attempts'])
        if not complete and attempted and charged>=3:job.update(status='failed',reason='Three attempts exhausted for this source; all evidence preserved')
    state.pop('active_session');state.pop('active_job',None)
    if action=='shared_setup':state.update(status='blocked_shared_setup',reason='Shared setup/cache failure; repair and requalification required',blocking_run=session['run'])
    elif action=='allocation_wait':
        n=state.get('allocation_failures',0)+1;state.update(status='retry_wait',allocation_failures=n,next_attempt_after=time.time()+min(3600,120*2**min(5,n-1)))
    elif action=='model_blocked':
        for job in state['jobs']:
            if job['model']==session['model'] and remaining(job,rows):job.update(status='blocked',reason='Official checkpoint access unavailable')
        state['status']='queued'
    else:state.update(status='queued',allocation_failures=0,next_attempt_after=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--controller-python',required=True);a=p.parse_args()
    root=Path(os.environ['STRASSEN_PROJECT_ROOT']);campaign=a.campaign.resolve();queue=campaign/'queue_v001';statepath=queue/'state.json'
    lock=(queue/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=read(statepath);source=queue/config['source_dir'];source_archive=queue/config['source_archive'];children={}
    def journal(title,detail,state='working'):
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state',state,'--title',title,'--detail',detail,'--evidence',str(statepath.relative_to(root))],stdout=subprocess.DEVNULL)
    def commit(paths,message):
        paths=[str(p.relative_to(root)) for p in paths if p.exists()]
        for _ in range(6):
            r=subprocess.run(['git','add','--',*paths],cwd=root,capture_output=True)
            if not r.returncode:
                r=subprocess.run(['git','commit','--only','-m',message,'--',*paths],cwd=root,capture_output=True)
                if not r.returncode or b'nothing to commit' in r.stdout:return
            time.sleep(5)
        raise RuntimeError('Scoped archive commit failed')
    def launch(run,legacy=False,recover=False):
        script='run_llm_case_controller_v005.py' if legacy else 'run_llm_session_controller_v001.py'
        command=[a.controller_python,str(source/'tools'/script),'--run',str(run),'--controller-python',a.controller_python]
        if recover:command+=['--recover']
        with (run/'queue-controller.log').open('ab') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        children[child.pid]=child;save(run/'queue-controller-process.json',dict(pid=child.pid,utc=utc(),recover=recover,command=command));return child.pid
    def allocations():
        r=subprocess.run([a.controller_python,str(source/'runtime/colab_control_v003.py'),'list'],capture_output=True,text=True,timeout=150)
        if r.returncode:raise RuntimeError('Cannot reconcile Colab inventory')
        for line in r.stdout.splitlines():
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('kind')=='allocations':return d['assignments']
        raise RuntimeError('No allocation inventory')
    def release(run):
        allocation=read(run/'allocation.json')
        if not allocation:return True
        with (run/'supervisor-release.log').open('ab') as f:
            return subprocess.run([a.controller_python,str(source/'runtime/release_allocation_v003.py'),'--session',allocation['session'],'--expect-endpoint',allocation['endpoint']],stdout=f,stderr=subprocess.STDOUT,timeout=240).returncode==0
    while True:
        try:
            state=read(statepath)
            if state['status'] in ('paused','blocked_shared_setup','completed','finished_with_failures'):return
            rows=ledger(campaign/'completed_entries.jsonl')
            for child in children.values():child.poll()
            if not state.get('active_session'):
                old=next((j for j in state['jobs'] if j['status']=='running'),None)
                if old:
                    attempt=old['attempts'][-1]
                    state['active_session']=dict(run=attempt['run'],pid=attempt.get('pid'),model=old['model'],job_ids=[old['job_id']],source_commit=attempt['source_commit'],legacy=True,recoveries=attempt.get('recoveries',0))
            session=state.get('active_session')
            if session:
                run=campaign/session['run'];latest=read(run/'latest.json',{});remote=latest.get('status.json',{});reconcile(state,rows,remote)
                if running(run,session.get('pid')):state['status']='running'
                else:
                    receipt=read(run/'completion.json')
                    if not receipt:
                        if read(run/'allocation.json') and session.get('recoveries',0)<2:
                            session['pid']=launch(run,session.get('legacy',False),True);session['recoveries']=session.get('recoveries',0)+1
                            journal('Recovering LLM session controller','Reattaching to the same endpoint; completed groups remain saved.')
                        else:
                            receipt=dict(status='controller_failed',remote_summary=remote,release_exit_code=0 if release(run) else 1,finished_utc=utc());save(run/'completion.json',receipt)
                    if receipt:
                        if receipt.get('release_exit_code',0) and release(run):receipt['release_exit_code']=0;save(run/'completion.json',receipt)
                        finish(state,receipt,rows);save(statepath,state)
                        commit([run,campaign/'completed_entries.jsonl',statepath],'Archive LLM model session '+run.name)
            else:
                reconcile(state,rows)
                pending=next((j for j in state['jobs'] if j['status']=='queued'),None)
                if not pending:
                    state.update(status='completed' if len(rows)==state['expected_comparisons'] else 'finished_with_failures',finished_utc=utc());save(statepath,state)
                    commit([queue,campaign/'completed_entries.jsonl'],'Archive final LLM model-session queue outcome');return
                if time.time()<state.get('next_attempt_after',0):state['status']='retry_wait'
                elif allocations():state.update(status='blocked',reason='Existing allocation must be reconciled before another can be requested')
                else:
                    selected=[j for j in state['jobs'] if j['model']==pending['model'] and j['status']=='queued' and remaining(j,rows)]
                    cases=[dict(j,output_dtypes=remaining(j,rows)) for j in selected]
                    job=dict(cases[0],session_jobs=cases)
                    name='session-'+pending['model']+'-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:4]
                    run=campaign/name;run.mkdir();save(run/'job.json',job);shutil.copyfile(source_archive,run/'source.tar')
                    raw=json.dumps(job).encode()
                    with tarfile.open(run/'source.tar','a') as t:item=tarfile.TarInfo('job.json');item.size=len(raw);t.addfile(item,io.BytesIO(raw))
                    save(run/'frozen.json',dict(source_commit=state['source_commit'],source_sha256=hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),created_utc=utc(),scope=job))
                    for j in selected:j['status']='reserved';j['attempts'].append(dict(run=name,status='reserved',source_commit=state['source_commit'],started_utc=utc()))
                    selected[0]['status']='running';state['active_session']=dict(run=name,model=pending['model'],job_ids=[j['job_id'] for j in selected],source_commit=state['source_commit'],legacy=False,recoveries=0)
                    state['status']='running';state.pop('reason',None);save(statepath,state)
                    save(campaign/'active-run.json',dict(run_name=name,scope='Cached model session; checkpointed full-forward and resident measurements'))
                    state['active_session']['pid']=launch(run)
                    for j in selected:j['attempts'][-1]['pid']=state['active_session']['pid']
                    journal('Starting cached LLM model session',pending['model']+'; '+str(len(selected))+' remaining workloads share one runtime and checkpoint.')
            state['updated_utc']=utc();state['supervisor_pid']=os.getpid();reconcile(state,rows,read(campaign/state['active_session']['run']/'latest.json',{}).get('status.json',{}) if state.get('active_session') else None);save(statepath,state)
        except Exception as error:
            print(json.dumps(dict(kind='supervisor_error',utc=utc(),error_type=type(error).__name__,error=str(error))),flush=True)
        time.sleep(30)
if __name__=='__main__':main()
