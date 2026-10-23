"""Queued complete-model case; archive and release through its controller."""
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
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(run):
    from parent_replication_remote_v001 import PINS,execute
    source=run/'source';out=run/'artifacts';out.mkdir()
    private=BASE/'.runtime_private'/run.name;private.mkdir(parents=True,exist_ok=True);private.chmod(0o700)
    job=json.loads((source/'job.json').read_text())
    protocol=json.loads((source/'configs/llm_v6e_v002.json').read_text())
    jobs=job['session_jobs']
    if not jobs or any(j['model']!=job['model'] or j['revision']!=job['revision'] for j in jobs):raise ValueError('A session must contain one pinned model')
    if job['model'] not in {m['id'] for m in protocol['models']} or dict(batch=job['batch'],sequence=job['sequence']) not in protocol['prefill']:raise ValueError('Unplanned workload')
    lock=(BASE/'active.lock').open('ab');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(started_utc=datetime.now(timezone.utc).isoformat(),stages=[],status='running',cases=[])
    session_started=time.monotonic()
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith('PIP_')}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(source/'src'),JAX_PLATFORMS='tpu',
               HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    save(run/'status.json',status)
    try:
        sizes={'qwen3_8b':8,'qwen3_14b':14,'qwen3_32b':33,'mistral_7b':7,'mistral_24b':24,'gemma3_12b':12,'gemma3_27b':27}
        size=sizes[job['model']]
        if shutil.disk_usage(BASE).free<(2.1*size+25)*1024**3:raise RuntimeError('Insufficient disk for pinned checkpoint and full-logit references')
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))}
        if mem['MemAvailable']<(3*size+12)*1024**3:raise RuntimeError('Insufficient host RAM for streamed weights and official oracle')
        save(out/'resource-preflight.json',dict(disk_free=shutil.disk_usage(BASE).free,**mem))
        target=private/'current';python=target/'bin/python'
        status['active']='current-setup';save(run/'status.json',status)
        code=execute([sys.executable,'-m','venv','--without-pip',str(target)],out/'current-venv.log',env,120,source)
        if not code:
            code=execute([sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                          '--report',str(out/'current-installation.json'),'jax==0.11.2','jaxlib==0.11.2','libtpu==0.0.48',*PINS],out/'current-install.log',env,900,source)
        status['stages'].append(dict(name='current-setup',returncode=code));save(run/'status.json',status)
        if code:raise RuntimeError('Current-stack setup failed')
        execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'],out/'current-freeze.txt',env,60,source)
        code=execute([sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cpu','torch==2.8.0'],out/'torch-install.log',env,600,source)
        if code:raise RuntimeError('Official CPU oracle setup failed')
        execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'],out/'oracle-freeze.txt',env,60,source)
        phases=[('kernel-qualification',[str(source/'tools/check_fused_tpu_v004.py'),'--output-dir',str(out/'kernel-qualification')],1200,env),
                ('stream-semantics',[str(source/'tools/check_llm_stream_v002.py'),'--output',str(out/'stream-semantics')],900,dict(env,JAX_PLATFORMS='cpu'))]
        for title,command,timeout,stage_env in phases:
            status['active']=title;save(run/'status.json',status);started=time.monotonic()
            code=execute([str(python),*command],out/(title+'.log'),stage_env,timeout,source)
            status['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started));save(run/'status.json',status)
            if code:
                status['failure_class']='shared_setup';raise RuntimeError('Shared qualification failed: '+title)
        for index,case in enumerate(jobs):
            # Bound runtime lifetime. Completed groups have already been offered
            # for download; untouched workloads requeue on a new allocation.
            if time.monotonic()-session_started>20*3600:break
            caseout=out/'cases'/case['job_id'];caseout.mkdir(parents=True)
            status['current_job']=case;status['case_index']=index;status['active_case_path']=str(caseout.relative_to(out))
            phases=[('checkpoint-inputs',[str(source/'tools/prepare_llm_case_v003.py'),'--protocol',str(source/'configs/llm_v6e_v002.json'),'--model',case['model'],'--batch',str(case['batch']),'--sequence',str(case['sequence']),'--private',str(private/'model'),'--output',str(caseout/'checkpoint-inputs')],3600,dict(env,JAX_PLATFORMS='cpu')),
                    ('main-prefill',[str(source/'tools/run_llm_case_v003.py'),'--protocol',str(source/'configs/llm_v6e_v002.json'),'--private',str(private/'model'),'--inputs',str(caseout/'checkpoint-inputs'),'--output',str(caseout/'main-prefill'),'--output-dtypes',*case['output_dtypes'],*(['--pilot'] if index==0 else [])],36000,env)]
            result=dict(job_id=case['job_id'],status='running',stages=[])
            for title,command,timeout,stage_env in phases:
                status['active']=title;save(run/'status.json',status);started=time.monotonic()
                budget=min(timeout,max(1,int(22*3600-(time.monotonic()-session_started))))
                code=execute([str(python),*command],caseout/(title+'.log'),stage_env,budget,source)
                result['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started))
                if code:
                    result.update(status='failed',failed_stage=title,failure_tail=(caseout/(title+'.log')).read_text(errors='replace')[-4000:])
                    if any(s in result['failure_tail'] for s in ("can't open file",'ModuleNotFoundError','unrecognized arguments','FileExistsError','Cached full-model outputs differ','Cached weight checksum mismatch')):result['failure_class']='shared_setup'
                    break
            else:result['status']='completed'
            save(caseout/'case-summary.json',result);status['cases'].append(result);save(run/'status.json',status)
            # A complete case archive also preserves diagnostics for failed
            # cases. Model/checkpoint/layout caches live outside this tree.
            package=caseout.with_suffix('.tar.gz')
            with tarfile.open(package,'w:gz') as tar:
                tar.add(caseout,arcname='case')
            save(caseout/'case-ready.json',dict(job_id=case['job_id'],path=str(package),sha256=digest(package),status=result['status']))
            for pattern in ('native-logits-*','pilot-original-*'):
                for path in (private/'model').glob(pattern):
                    if path.is_dir():shutil.rmtree(path)
            if result.get('failure_class')=='shared_setup':
                status['failure_class']='shared_setup';raise RuntimeError('Shared session implementation failure')
        status['status']='completed' if len(status['cases'])==len(jobs) and all(c['status']=='completed' for c in status['cases']) else 'partial_or_failed'
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();status.pop('active',None)
        save(run/'status.json',status);save(out/'summary.json',status)
        archive=run.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as package:
            package.add(out,arcname=run.name+'/artifacts',filter=lambda info:None if info.name.endswith('.tar.gz') else info);package.add(run/'launch.json',arcname=run.name+'/launch.json')
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
    script=source/'runtime/llm_model_session_v001.py'
    save(run/'launch.json',dict(source_sha256=a.sha256,command=[sys.executable,str(script),'--worker',str(run)]))
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(script),'--worker',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='llm_case_launched',run=str(run),pid=child.pid)),flush=True)


if __name__=='__main__':main()
