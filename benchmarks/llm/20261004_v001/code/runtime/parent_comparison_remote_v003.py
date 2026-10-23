"""One bounded current-stack tuning/comparison worker on a fresh v6e."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
BASE=Path('/content/Strassen_MM_Focus')
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(run):
    from parent_replication_remote_v001 import PINS,execute
    source=run/'source';out=run/'artifacts';out.mkdir()
    private=BASE/'.runtime_private'/run.name;private.mkdir(parents=True)
    lock=(BASE/'active.lock').open('ab');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(started_utc=datetime.now(timezone.utc).isoformat(),stages=[],status='running')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith('PIP_')}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(source/'src'),JAX_PLATFORMS='tpu',
               HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    save(run/'status.json',status)
    try:
        if shutil.disk_usage(BASE).free<85*1024**3:raise RuntimeError('Less than 85 GiB free for the public checkpoint')
        target=private/'current';python=target/'bin/python'
        status['active']='current-setup';save(run/'status.json',status)
        code=execute([sys.executable,'-m','venv','--without-pip',str(target)],out/'current-venv.log',env,120,source)
        if not code:
            code=execute([sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                          '--report',str(out/'current-installation.json'),'jax==0.11.2','jaxlib==0.11.2','libtpu==0.0.48',*PINS],out/'current-install.log',env,900,source)
        status['stages'].append(dict(name='current-setup',returncode=code));save(run/'status.json',status)
        if code:raise RuntimeError('Current-stack setup failed')
        execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'],out/'current-freeze.txt',env,60,source)
        phases=[('tune',[str(source/'tools/tune_parent_fused_v004.py')],5400)]
        for dtype in ('bfloat16','float32'):
            for family in ('s1','s2'):
                for stage in ('streamed','quality'):
                    phases.append((family+'-'+dtype+'-'+stage,[str(source/'tools/compare_parent_fused_v001.py'),
                        '--profile',str(out/'tune/profile.json'),'--dtype',dtype,'--family',family,'--stage',stage],2400))
        for title,command,timeout in phases:
            status['active']=title;save(run/'status.json',status)
            started=time.monotonic()
            code=execute([str(python),*command,'--output',str(out/title),'--cache',str(private/'checkpoint')],
                         out/(title+'.log'),env,timeout,source)
            status['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started))
            save(run/'status.json',status)
            if title=='tune' and (not (out/'tune/profile.json').exists() or code):
                raise RuntimeError('Tuning did not complete; no evaluation uses an unfinished profile')
        status['status']='completed' if all(s['returncode']==0 for s in status['stages']) else 'partial_or_failed'
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();status.pop('active',None)
        save(run/'status.json',status);save(out/'summary.json',status)
        archive=run.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as package:
            package.add(out,arcname=run.name+'/artifacts');package.add(run/'launch.json',arcname=run.name+'/launch.json')
        save(run/'archive-ready.json',dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size))
        lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--run-id');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:return worker(a.worker)
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id or ''):raise ValueError('Invalid run ID')
    if not a.archive.resolve().is_relative_to(BASE) or digest(a.archive)!=a.sha256:raise ValueError('Invalid source archive')
    run=BASE/'runs'/a.run_id;run.mkdir(parents=True,exist_ok=False);source=run/'source';source.mkdir()
    with tarfile.open(a.archive) as package:
        for member in package.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):raise ValueError('Unsafe source member')
        package.extractall(source,filter='data')
    script=source/'runtime/parent_comparison_remote_v003.py'
    save(run/'launch.json',dict(source_sha256=a.sha256,command=[sys.executable,str(script),'--worker',str(run)]))
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(script),'--worker',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='parent_comparison_launched',run=str(run),pid=child.pid)),flush=True)


if __name__=='__main__':main()
