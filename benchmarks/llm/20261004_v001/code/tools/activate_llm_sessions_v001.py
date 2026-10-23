"""Freeze a qualified session bundle and adopt the unchanged active legacy case."""
import argparse,hashlib,io,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
from llm_queue_state_v002 import read,save,utc,ledger,remaining
from reorder_llm_queue_v001 import process
from llm_bundle_v002 import validate_archive


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--controller-python',required=True);a=p.parse_args()
    root=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve();campaign=a.campaign.resolve();queue=campaign/'queue_v001';statepath=queue/'state.json'
    destination=queue/'optimized_20261003_v001';destination.mkdir(exist_ok=False)
    paths=['src','tools','runtime','configs/llm_v6e_v002.json'];subprocess.run(['git','diff','--exit-code','HEAD','--',*paths],cwd=root,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();archive=destination/'source.tar'
    with archive.open('xb') as f:subprocess.run(['git','archive',revision,'--',*paths],cwd=root,stdout=f,check=True)
    with tarfile.open(archive) as t:t.extractall(destination/'source',filter='data')
    state=read(statepath);rows=ledger(campaign/'completed_entries.jsonl')
    selected=[dict(j,output_dtypes=remaining(j,rows)) for j in state['jobs'] if j['model']=='qwen3_8b' and j['status']=='queued' and remaining(j,rows)]
    if not selected:raise RuntimeError('Expected pending Qwen3-8B jobs')
    job=dict(selected[0],session_jobs=selected);check_archive=destination/'launch-preflight.tar';check_archive.write_bytes(archive.read_bytes());raw=json.dumps(job).encode()
    with tarfile.open(check_archive,'a') as t:info=tarfile.TarInfo('job.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
    save(destination/'bundle-validation.json',validate_archive(check_archive,job,a.controller_python));check_archive.unlink()
    watchdog=read(queue/'watchdog-process.json')['pid'];supervisor=read(queue/'supervisor-process.json')['pid']
    for pid,script in [(watchdog,'watch_llm_queue_v004.py'),(supervisor,'run_llm_queue_v004.py')]:
        if script not in process(pid) or str(campaign) not in process(pid):raise RuntimeError('Expected old queue identity differs')
    active=next(j for j in state['jobs'] if j['status']=='running');controller=active['attempts'][-1]['pid'];run=campaign/active['attempts'][-1]['run']
    if active['model']!='qwen3_14b' or 'run_llm_case_controller_v005.py' not in process(controller) or str(run) not in process(controller):raise RuntimeError('Active legacy run not verified')
    ledger_hash=hashlib.sha256((campaign/'completed_entries.jsonl').read_bytes()).hexdigest()
    parked=False;retired=False
    try:
        os.kill(watchdog,signal.SIGSTOP);parked=True
        for _ in range(50):
            if process(watchdog).startswith('T'):break
            time.sleep(.1)
        else:raise RuntimeError('Watchdog did not park')
        os.kill(supervisor,signal.SIGTERM)
        for _ in range(50):
            s=process(supervisor)
            if not s or s.startswith('Z'):break
            time.sleep(.1)
        else:raise RuntimeError('Supervisor did not stop')
        state=read(statepath);save(destination/'state-before.json',state)
        state.setdefault('source_history',[]).append({k:state[k] for k in ('source_dir','source_archive','source_commit','source_sha256')})
        state.update(source_dir='optimized_20261003_v001/source',source_archive='optimized_20261003_v001/source.tar',source_commit=revision,source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),orchestration_version='model-session-v001')
        save(statepath,state);save(destination/'state-after.json',state)
        os.kill(watchdog,signal.SIGTERM);os.kill(watchdog,signal.SIGCONT);retired=True
        for _ in range(50):
            s=process(watchdog)
            if not s or s.startswith('Z'):break
            time.sleep(.1)
        else:raise RuntimeError('Old watchdog did not exit')
        command=[a.controller_python,str(destination/'source/tools/watch_llm_queue_v005.py'),'--campaign',str(campaign),'--controller-python',a.controller_python]
        with (destination/'watchdog.log').open('ab') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(root),PYTHONDONTWRITEBYTECODE='1'))
        save(destination/'activation.json',dict(utc=utc(),source_commit=revision,command=command,watchdog_pid=child.pid,retired_watchdog_pid=watchdog,retired_supervisor_pid=supervisor,unchanged_controller_pid=controller,active_run=run.name,ledger_sha256=ledger_hash))
        print(json.dumps(read(destination/'activation.json')),flush=True)
    finally:
        if parked and not retired:os.kill(watchdog,signal.SIGCONT)
if __name__=='__main__':main()
