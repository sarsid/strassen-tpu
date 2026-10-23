"""Persistent serial v6e queue: 70 workloads, 700 measured arm/store entries.

Adopts the current case without interference. Restarts dead controllers against
an existing endpoint; never creates another TPU until reconciliation succeeds.
"""
import argparse,collections,fcntl,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,uuid
from pathlib import Path
from llm_queue_state_v001 import utc,read,save,plan,ledger,remaining,failure_action


def running(run,pid=None):
    # Process identity, not PID reuse, determines ownership.
    args=['ps','-p',str(pid),'-o','pid=,command='] if pid else ['ps','-axo','pid=,command=']
    result=subprocess.run(args,text=True,capture_output=True)
    found=[]
    for line in result.stdout.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)==2 and 'run_llm_case_controller_v00' in fields[1] and '--run '+str(run) in fields[1]:found.append(int(fields[0]))
    if len(found)>1:raise RuntimeError('Multiple controllers found for one run')
    return found[0] if found else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--controller-python',required=True)
    p.add_argument('--init',action='store_true');p.add_argument('--adopt-run');p.add_argument('--adopt-pid',type=int);a=p.parse_args()
    root=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()
    campaign=a.campaign.resolve();queue=campaign/'queue_v001';queue.mkdir(exist_ok=True)
    lock=(queue/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=queue/'state.json';protocol=read(root/'configs/llm_v6e_v002.json')
    def journal(title,detail,kind='working'):
        subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--state',kind,'--title',title,'--detail',detail,'--evidence',str(statepath.relative_to(root))],stdout=subprocess.DEVNULL,check=False)
    def commit(paths,message):
        paths=[str(Path(x).relative_to(root)) for x in paths if Path(x).exists()]
        for attempt in range(6):
            result=subprocess.run(['git','add','--',*paths],cwd=root,capture_output=True)
            if result.returncode==0:
                result=subprocess.run(['git','commit','--only','-m',message,'--',*paths],cwd=root,capture_output=True)
                if result.returncode==0 or b'nothing to commit' in result.stdout:return
            time.sleep(5)
        raise RuntimeError('Scoped result commit failed; evidence retained on disk')
    if a.init:
        if statepath.exists():raise FileExistsError('Queue already initialized')
        jobs=plan(protocol)
        if len(jobs)!=70 or sum(len(g) for j in jobs for g in j['groups'].values())!=700:raise ValueError('Unexpected campaign size')
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        paths=['src','runtime','configs/llm_v6e_v002.json','tools/check_fused_tpu_v004.py','tools/fusion_oracle_v001.py','tools/fusion_oracle_v002.py','tools/check_llm_stream_v002.py','tools/prepare_llm_case_v002.py','tools/load_gemma_text_v001.py','tools/run_llm_case_v002.py','tools/run_llm_case_controller_v004.py','tools/llm_queue_state_v001.py','tools/run_llm_queue_v001.py','tools/watch_llm_queue_v001.py','tools/stage_mlsys_auth_v002.py']
        subprocess.run(['git','diff','--exit-code','HEAD','--',*paths],cwd=root,check=True)
        with (queue/'source.tar').open('xb') as f:subprocess.run(['git','archive',revision,'--',*paths],cwd=root,stdout=f,check=True)
        with tarfile.open(queue/'source.tar') as t:t.extractall(queue/'source',filter='data')
        state=dict(schema_version=1,status='queued',created_utc=utc(),source_commit=revision,source_sha256=hashlib.sha256((queue/'source.tar').read_bytes()).hexdigest(),expected_comparisons=700,jobs=jobs,allocation_failures=0,next_attempt_after=0)
        if a.adopt_run:
            run=campaign/a.adopt_run
            if run.parent!=campaign or not running(run,a.adopt_pid):raise ValueError('Adopted controller is not live')
            job=jobs[0];job['status']='running';job['attempts'].append(dict(run=run.name,pid=a.adopt_pid,adopted=True,status='running',recoveries=0))
            save(run/'job.json',dict(job,output_dtypes=list(job['groups'])))
        save(statepath,state);commit([queue,campaign/a.adopt_run/'job.json'] if a.adopt_run else [queue],'Queue all 700 v6e LLM comparisons')
        print(json.dumps(dict(status='initialized',jobs=70,comparisons=700,source_commit=revision)),flush=True);return 0
    if not statepath.exists():raise FileNotFoundError('Initialize queue first')
    source=queue/'source';children={}
    def launch(run,recover=False):
        command=[a.controller_python,str(source/'tools/run_llm_case_controller_v004.py'),'--run',str(run),'--controller-python',a.controller_python]
        if recover:command+=['--recover']
        with (run/'queue-controller.log').open('ab') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(root),PYTHONDONTWRITEBYTECODE='1'))
        children[child.pid]=child
        save(run/'queue-controller-process.json',dict(pid=child.pid,utc=utc(),recover=recover,command=command));return child.pid
    def allocations():
        result=subprocess.run([a.controller_python,str(source/'runtime/colab_control_v003.py'),'list'],capture_output=True,text=True,timeout=150)
        if result.returncode:raise RuntimeError('Cannot verify Colab allocation inventory')
        for line in result.stdout.splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('kind')=='allocations':return row['assignments']
        raise RuntimeError('No allocation inventory returned')
    def release(run,allocation):
        command=[a.controller_python,str(source/'runtime/release_allocation_v002.py'),'--session',allocation['session'],'--expect-endpoint',allocation['endpoint']]
        with (run/'supervisor-release.log').open('ab') as log:return subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode==0
    last_notice=None
    while True:
        try:
            state=read(statepath);rows=ledger(campaign/'completed_entries.jsonl')
            for child in list(children.values()):child.poll()
            for job in state['jobs']:
                if not remaining(job,rows) and job['status']!='running':job['status']='completed'
            active=next((j for j in state['jobs'] if j['status']=='running'),None)
            if active:
                attempt=active['attempts'][-1];run=campaign/attempt['run'];pid=running(run,attempt.get('pid')) or running(run)
                if pid:
                    attempt['pid']=pid;state['status']='running';state['active_job']=active['job_id']
                else:
                    receipt=read(run/'completion.json')
                    if not receipt:
                        allocation=read(run/'allocation.json')
                        if allocation and attempt.get('recoveries',0)<2:
                            attempt['pid']=launch(run,True);attempt['recoveries']=attempt.get('recoveries',0)+1
                            journal('Recovering the LLM controller','Reattaching to '+run.name+' on its existing endpoint; no additional TPU requested.')
                        else:
                            released=release(run,allocation) if allocation else True
                            receipt=dict(status='controller_failed',error='Controller stopped without completion receipt',release_exit_code=0 if released else 1,finished_utc=utc());save(run/'completion.json',receipt)
                    if receipt:
                        if receipt.get('release_exit_code',0)!=0:
                            allocation=read(run/'allocation.json')
                            if allocation and release(run,allocation):receipt['release_exit_code']=0;save(run/'completion.json',receipt)
                        action=failure_action(receipt)
                        if action=='release_blocked':
                            state['status']='blocked';state['reason']='Owned TPU release not verified; next allocation is blocked.'
                        else:
                            attempt.update(status=receipt['status'],finished_utc=utc());active['status']='queued'
                            if not remaining(active,rows):active['status']='completed'
                            elif action=='allocation_wait':
                                state['allocation_failures']+=1
                                state['next_attempt_after']=time.time()+min(3600,120*2**min(5,state['allocation_failures']-1))
                                state['status']='retry_wait';state['reason']='Colab allocation unavailable; bounded backoff. No workload marked complete.'
                            elif action=='model_blocked':
                                for job in state['jobs']:
                                    if job['model']==active['model'] and remaining(job,rows):job.update(status='blocked',reason='Official checkpoint access unavailable')
                            else:
                                state['allocation_failures']=0
                                attempts=sum(x.get('status')!='allocation_failed' for x in active['attempts'])
                                if active['status']!='completed' and attempts>=3:active.update(status='failed',reason='Three workload attempts exhausted; saved evidence retained')
                            commit([run,campaign/'completed_entries.jsonl'], 'Archive queued v6e LLM workload '+run.name)
                            state.pop('active_job',None)
            else:
                pending=next((j for j in state['jobs'] if j['status']=='queued'),None)
                if not pending:
                    state['status']='completed' if len(rows)==700 else 'finished_with_failures'
                    state['finished_utc']=utc();state['measured_comparisons']=len(rows);save(statepath,state)
                    commit([queue,campaign/'completed_entries.jsonl'],'Archive final v6e LLM queue outcome')
                    journal('LLM queue finished',str(len(rows))+' / 700 comparisons measured. Failed or blocked workloads remain explicitly recorded. No further experiment is queued.','complete' if len(rows)==700 else 'blocked');return 0
                if time.time()<state.get('next_attempt_after',0):state['status']='retry_wait'
                elif allocations():
                    state['status']='blocked';state['reason']='An existing Colab allocation must be reconciled before another can be requested.'
                else:
                    name='case-'+pending['model']+'-b'+str(pending['batch'])+'s'+str(pending['sequence'])+'-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:4]
                    run=campaign/name;run.mkdir()
                    job=dict(pending,output_dtypes=remaining(pending,rows));save(run/'job.json',job)
                    shutil.copyfile(queue/'source.tar',run/'source.tar')
                    raw=json.dumps(job).encode()
                    with tarfile.open(run/'source.tar','a') as t:
                        item=tarfile.TarInfo('job.json');item.size=len(raw);t.addfile(item,io.BytesIO(raw))
                    save(run/'frozen.json',dict(source_commit=state['source_commit'],source_sha256=hashlib.sha256((run/'source.tar').read_bytes()).hexdigest(),created_utc=utc(),scope=job))
                    pending['status']='running';pending['attempts'].append(dict(run=name,status='running',recoveries=0,started_utc=utc()))
                    state['active_job']=pending['job_id'];state['status']='running';state.pop('reason',None);save(statepath,state)
                    save(campaign/'active-run.json',dict(run_name=name,scope='Full 700-comparison queue; current matched workload'))
                    pending['attempts'][-1]['pid']=launch(run)
                    journal('Starting the next queued LLM workload',pending['model']+' B'+str(pending['batch'])+' S'+str(pending['sequence'])+'; '+str(len(rows))+' / 700 saved.')
            state['updated_utc']=utc();state['supervisor_pid']=os.getpid();state['measured_comparisons']=len(rows)
            state['counts']=dict(collections.Counter(j['status'] for j in state['jobs']));save(statepath,state)
            notice=(state['status'],state.get('reason'))
            if notice!=last_notice and state['status'] in ('blocked','retry_wait'):
                journal('LLM queue waiting',state.get('reason','Waiting for retry'),'blocked');last_notice=notice
        except Exception as error:
            # Keep the supervisor alive. Persisted running-job ownership still
            # prevents an extra allocation on the next reconciliation pass.
            print(json.dumps(dict(utc=utc(),kind='supervisor_error',error_type=type(error).__name__,error=str(error))),flush=True)
            journal('LLM supervisor recovery',type(error).__name__+': '+str(error)+'. Reconciliation will retry in 30 seconds.','blocked')
        time.sleep(30)

if __name__=='__main__':raise SystemExit(main())
