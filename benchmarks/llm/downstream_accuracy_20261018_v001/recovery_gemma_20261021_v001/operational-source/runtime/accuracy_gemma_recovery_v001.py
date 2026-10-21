"""Resume Gemma only, retaining completed Qwen archives and frozen numerical sources."""
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
import threading
import time

BASE=Path('/content/Strassen_MM_Focus')
MODELS=('gemma4_31b',)


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def archive(root,path):
    if not path.exists():
        temp=path.with_suffix('.partial')
        with tarfile.open(temp,'w:gz') as out:out.add(root,arcname=root.name)
        temp.rename(path)
    value=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(path.with_suffix('.ready.json'),value);return value


def worker(run,previous):
    source=run/'source';artifacts=run/'artifacts';artifacts.mkdir(exist_ok=True)
    sys.path.insert(0,str(source/'runtime'))
    from parent_exact_remote_v001 import execute
    lock=(BASE/'active.lock').open('ab');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    private=BASE/'.runtime_private'/previous.name
    if not private.is_dir():raise RuntimeError('Retained private environment is absent')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith(('PIP_','QWEN3_'))}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',JAX_PLATFORMS='tpu',HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',STRASSEN_SOURCE_SHA256=json.loads((run/'launch.json').read_text())['source_sha256'])
    state=dict(status='running',scope='three-model paired downstream subsets',started_utc=datetime.now(timezone.utc).isoformat(),completed_models=['qwen3_14b','qwen3_32b'],test_evaluations_completed=6,test_evaluations_total=9,release_attempted=False)
    stop=threading.Event()
    def stage(name,**extra):state.update(stage=name,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);save(run/'status.json',state)
    def observe():
        selected=None;offset=0;context={}
        while not stop.wait(2):
            model=state.get('model');paths=sorted((artifacts/state.get('event_dir',model)).glob('events*.jsonl')) if model else []
            if not paths:continue
            path=paths[-1]
            if selected!=path:selected=path;offset=0;context={'model':model,'completed_tasks':[]}
            try:
                with path.open() as f:
                    f.seek(offset)
                    while True:
                        old=f.tell();line=f.readline()
                        if not line or not line.endswith('\n'):f.seek(old);break
                        row=json.loads(line);context['latest_event']=row
                        if row['kind']=='task_batch_started':context.update({k:row[k] for k in ('task','batch','batches','completed_examples','total_examples')})
                        if row['kind']=='task_batch_complete':context.update(completed_examples=row['completed_examples'])
                        if row['kind']=='task_complete' and row['task'] not in context['completed_tasks']:context['completed_tasks'].append(row['task'])
                    offset=f.tell()
                save(run/'progress.json',dict(context,observed_utc=datetime.now(timezone.utc).isoformat()))
                # Persist actual admission/preflight filenames for the contents-only
                # observer. Scientific event/receipt contents are never rewritten.
                output=artifacts/model
                metadata=[str(p.relative_to(run)) for p in sorted(output.glob('admission-*.json'))]
                metadata += [str(p.relative_to(run)) for p in sorted(output.glob('query-block-preflight-*.json'))]
                if metadata:save(run/'runtime-metadata-ready.json',dict(model=model,files=metadata))
            except (OSError,ValueError):pass
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    try:
        stage('checking_retained_environment');python=private/'environment/bin/python'
        if not python.exists():raise RuntimeError('Retained Python environment is absent')
        for name in ('installation.json','freeze.txt'):
            old=previous/'artifacts'/name
            if old.exists():shutil.copy2(old,artifacts/name)
        for model in MODELS:
            cache=private/('weights-'+model)
            if shutil.disk_usage(BASE).free<(1 if (cache/'canonical').is_dir() else 135)*1024**3:raise RuntimeError('Insufficient private checkpoint disk capacity')
            output=artifacts/model
            cmd=[str(python),str(source/'tools/accuracy_study_v002.py'),'--model',model,
                '--inputs',str(source/f'plans/llm_accuracy_20261018_v001/subset_{model}_v002'),
                '--pilot',str(source/f'plans/llm_accuracy_20261018_v001/pilot_{model}_v001.json'),
                '--output',str(output),'--cache',str(cache)]
            save(artifacts/(model+'-command.json'),dict(command=cmd,source_sha256=env['STRASSEN_SOURCE_SHA256']))
            for attempt in range(2):
                preflight=artifacts/f'{model}-preflight-{attempt}'
                stage('training_validation',model=model,attempt=attempt,event_dir=preflight.name)
                check=[str(python),str(source/'tools/accuracy_query_block_preflight_v001.py'),'--model',model,
                    '--pilot',str(source/f'plans/llm_accuracy_20261018_v001/pilot_{model}_v001.json'),
                    '--output',str(preflight),'--cache',str(cache)]
                save(artifacts/f'{model}-preflight-{attempt}-command.json',dict(command=check))
                code=execute(check,artifacts/f'{model}-preflight-{attempt}.log',env,3600,source)
                if code or not (preflight/'summary.json').exists():
                    stage('training_validation_failed',model=model,attempt=attempt,returncode=code)
                    continue
                passed=json.loads((preflight/'summary.json').read_text())
                if passed['status']!='passed':raise RuntimeError('Preflight did not pass')
                output.mkdir(exist_ok=True)
                shutil.copy2(preflight/'summary.json',output/f'query-block-preflight-{attempt}.json')
                stage('evaluating',model=model,attempt=attempt,event_dir=model)
                code=execute(cmd,artifacts/f'{model}-attempt-{attempt}.log',env,24*3600,source)
                if code==0 and (output/'summary.json').exists():break
                stage('retrying_from_completed_batches',model=model,attempt=attempt,returncode=code)
            else:raise RuntimeError('Model failed after bounded restart; TPU retained for diagnosis')
            ready=archive(output,run/(model+'.tar.gz'));state['completed_models'].append(model)
            state['test_evaluations_completed']=3*len(state['completed_models']);stage('model_archived',model=model,model_archive=ready)
            # Only this disposable run/model weight cache is removed. Exact
            # checkpoint/canonical hashes remain in the public model archive.
            if cache.exists() and cache.parent==private:shutil.rmtree(cache)
        stage('complete',status='completed')
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        if type(error) is RuntimeError:state['error']=str(error)
    finally:
        stop.set();thread.join(timeout=5)
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save(run/'status.json',state)
        ready=archive(artifacts,run/'all-artifacts.tar.gz');save(run/'archive-ready.json',ready);lock.close()
