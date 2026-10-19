"""Observe one foreground repair on the existing owned TPU; never allocate."""
import argparse
from datetime import datetime,timezone
import fcntl,hashlib,json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser()
    for name in ('run-id','session','endpoint','source-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=ROOT/'runs'/a.run_id
    if root.parent!=ROOT/'runs' or not root.is_dir():raise ValueError('Prepared local run required')
    if (root/'retained-starter-start-v001.json').exists():raise RuntimeError('Already launched; reconcile existing processes')
    lock=(root/'retained-starter.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    control=[sys.executable,str(ROOT/'runtime/colab_control_v004.py')]
    listed=subprocess.run(control+['list'],capture_output=True,text=True,check=True,timeout=100)
    save(root/'prelaunch-assignment-v001.json',json.loads(listed.stdout))
    matches=[r for r in json.loads(listed.stdout)['assignments']if r['endpoint']==a.endpoint and r['name']==a.session]
    if len(matches)!=1:raise RuntimeError('Owned retained assignment absent or ambiguous')
    expected=json.loads((root/'source-manifest.json').read_text())
    if expected['sha256']!=a.source_sha256 or hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest()!=a.source_sha256:raise ValueError('Prepared bundle digest mismatch')
    launcher=root/'source/runtime/accuracy_query_block_recovery_v001.py'
    if hashlib.sha256(launcher.read_bytes()).hexdigest()!=expected['files']['runtime/accuracy_query_block_recovery_v001.py']:raise ValueError('Frozen launcher changed')
    command=control+['exec-file','--session',a.session,'--expect-endpoint',a.endpoint,
        '--file',str(launcher),'--timeout','172800',
        '--script-arg=--archive','--script-arg=/content/Strassen_MM_Focus/'+a.run_id+'-source.tar.gz',
        '--script-arg=--sha256','--script-arg='+a.source_sha256,'--script-arg=--run-id','--script-arg='+a.run_id,
        '--script-arg=--previous-run','--script-arg=/content/Strassen_MM_Focus/runs/20261019-accuracy-study-v006']
    with (root/'foreground-execution-v002.jsonl').open('x')as log:
        foreground=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    observer_command=[sys.executable,str(ROOT/'runtime/observe_accuracy_study_v002.py'),'--run-id',a.run_id,'--session',a.session,'--endpoint',a.endpoint]
    def observe():
        with (root/'observer-v002.log').open('a')as log:return subprocess.Popen(observer_command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    observer=observe()
    guard=None
    if sys.platform=='darwin':
        ac=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout
        if 'AC Power' in ac:guard=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    save(root/'retained-starter-start-v001.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),foreground_pid=foreground.pid,observer_pid=observer.pid,endpoint=a.endpoint,session=a.session,command=command,observer_command=observer_command,sleep_guard_pid=guard.pid if guard else None))
    restarts=0
    while foreground.poll() is None or observer.poll() is None:
        if observer.poll() is not None and not (root/'observer-completion-v002.json').exists() and restarts<3:
            restarts+=1;observer=observe()
            save(root/f'observer-restart-{restarts:03d}.json',dict(pid=observer.pid,utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(10)
    completion=json.loads((root/'observer-completion-v002.json').read_text()) if (root/'observer-completion-v002.json').exists() else {}
    save(root/'retained-starter-completion-v001.json',dict(finished_utc=datetime.now(timezone.utc).isoformat(),foreground_returncode=foreground.returncode,observer_returncode=observer.returncode,remaining_work=completion.get('remaining_work',True)))

if __name__=='__main__':main()
