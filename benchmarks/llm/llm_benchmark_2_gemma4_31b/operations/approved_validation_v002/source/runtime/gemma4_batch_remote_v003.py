"""Detached Gemma4 experiment: oracle barrier, scoped stages, immutable exports."""
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
from qwen32_batch_remote_v005 import save,package,BASE
from parent_exact_remote_v001 import execute

PINS=['jax==0.11.2','jaxlib==0.11.2','libtpu==0.0.48','transformers==5.5.0','tokenizers==0.22.2','huggingface-hub==1.33.0','safetensors==0.6.2','numpy==2.2.6','ml_dtypes==0.6.0','datasets==4.1.1','pyarrow==21.0.0','sentencepiece==0.2.1','protobuf==6.32.1','requests==2.32.5']


def worker(run):
    source=run/'source';artifacts=run/'artifacts';artifacts.mkdir()
    lock=(BASE/'active.lock').open('ab');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    private=BASE/'.runtime_private'/run.name;private.mkdir(parents=True)
    plan=json.loads((source/'configs/gemma4_parent_batch_v003.json').read_text())
    state=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',stages=[],batches=[],batch_archives=[],qualification='not_started')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith(('PIP_','QWEN3_'))}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',JAX_PLATFORMS='tpu',HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    save(run/'status.json',state)
    def progress(stage,label,**kw):save(run/'progress.json',dict(status='running',stage=stage,stage_label=label,test=label,utc=datetime.now(timezone.utc).isoformat(),**kw))
    try:
        if shutil.disk_usage(BASE).free<90*1024**3:raise RuntimeError('Need 90 GiB free for environment and checkpoint cache')
        state['active']='parent-setup';save(run/'status.json',state);progress('setup','Installing pinned Gemma environment')
        target=private/'parent';python=target/'bin/python'
        code=execute([sys.executable,'-m','venv','--without-pip',str(target)],artifacts/'parent-venv.log',env,120,source)
        pip=[sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:']
        if not code:code=execute(pip+['--report',str(artifacts/'installation.json'),*PINS],artifacts/'parent-install.log',env,900,source)
        if not code:code=execute(pip+['--index-url','https://download.pytorch.org/whl/cpu','--report',str(artifacts/'torch-installation.json'),'torch==2.8.0'],artifacts/'torch-install.log',env,900,source)
        if code:raise RuntimeError('Pinned environment setup failed')
        execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'],artifacts/'freeze.txt',env,60,source)
        shutil.copy2(source/'configs/gemma4_parent_batch_v003.json',artifacts/'plan.json')
        def stage_run(stage,batch,entries,profile=None):
            for attempt in range(plan['per_stage_retry_limit']+1):
                name=f'b{batch}-{stage}-a{attempt}' if stage!='oracle' else f'oracle-a{attempt}'
                output=artifacts/name;state.update(active=name,active_batch=batch if stage!='oracle' else None);save(run/'status.json',state)
                progress(stage,'Starting '+stage,batch=batch if stage!='oracle' else None)
                cmd=[str(python),str(source/'tools/gemma4_batch_stage_v003.py'),'--stage',stage,'--batch',str(batch),'--sequence',str(plan['sequence']),'--output',str(output),'--cache',str(private/'checkpoint'),'--plan',str(source/'configs/gemma4_parent_batch_v003.json'),'--progress',str(run/'progress.json')]
                if profile:cmd+=['--profile',str(profile),'--profile-sha256',hashlib.sha256(profile.read_bytes()).hexdigest()]
                start=time.monotonic();code=execute(cmd,artifacts/(name+'.log'),env,plan['stage_timeouts_seconds'][stage],source)
                completion=json.loads((output/'completion.json').read_text()) if (output/'completion.json').exists() else {}
                row=dict(name=name,batch=batch if stage!='oracle' else None,stage=stage,attempt=attempt,returncode=code,seconds=time.monotonic()-start)
                state['stages'].append(row);save(run/'status.json',state)
                entries.append((artifacts/(name+'.log'),name+'.log'))
                if output.exists():entries.append((output,name))
                if not code:return output
                if any(token in completion.get('error','') for token in ('Scientific','Nonfinite','Unregistered','source mismatch')):break
            return None
        entries=[];state['qualification']='running';save(run/'status.json',state)
        qualified=stage_run('oracle',1,entries)
        state['qualification']='passed' if qualified else 'failed';save(run/'status.json',state)
        if not qualified:raise RuntimeError('Official checkpoint qualification failed; batch timings blocked')
        for batch in plan['batches']:
            entries=[];profile=None;successful=True;quality_pass=True
            for stage in plan['stages']:
                output=stage_run(stage,batch,entries,profile)
                if output is None:successful=False;break
                if stage=='tune':profile=output/'profile.json'
                if stage in ('confirm','quality'):
                    summary=json.loads((output/'summary.json').read_text());quality_pass &= bool(summary['all_pass'])
            result=dict(batch=batch,status='completed' if successful else 'partial_or_failed',quality_pass=quality_pass if successful else None)
            summary=artifacts/f'b{batch}-summary.json';save(summary,result);entries.append((summary,summary.name))
            progress('archive','Saving batch artifacts',batch=batch)
            exported=package(run/f'b{batch}-results.tar.gz',entries)
            state['batch_archives'].append(dict(batch=batch,**exported));state['batches'].append(result);save(run/'status.json',state)
        state['status']='completed' if all(b['status']=='completed' for b in state['batches']) else 'completed_with_failures'
    except BaseException as e:state.update(status='failed',error_type=type(e).__name__,error=str(e)[-2000:])
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();state.pop('active',None)
        save(run/'status.json',state);save(artifacts/'summary.json',state)
        save(run/'archive-ready.json',package(run.with_suffix('.tar.gz'),[(artifacts,run.name+'/artifacts'),(run/'launch.json',run.name+'/launch.json')]))
        lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--run-id');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:return worker(a.worker)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',a.run_id or ''):raise ValueError('Invalid run ID')
    if not a.archive.resolve().is_relative_to(BASE) or hashlib.sha256(a.archive.read_bytes()).hexdigest()!=a.sha256:raise ValueError('Invalid source archive')
    run=BASE/'runs'/a.run_id;run.mkdir(parents=True,exist_ok=False);source=run/'source';source.mkdir()
    with tarfile.open(a.archive) as t:t.extractall(source,filter='data')
    script=source/'runtime/gemma4_batch_remote_v003.py';save(run/'launch.json',dict(source_sha256=a.sha256,command=[sys.executable,str(script),'--worker',str(run)]))
    with (run/'worker.log').open('x') as log:child=subprocess.Popen([sys.executable,str(script),'--worker',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='gemma4_batch_launched',run=str(run),pid=child.pid)),flush=True)


if __name__=='__main__':main()
