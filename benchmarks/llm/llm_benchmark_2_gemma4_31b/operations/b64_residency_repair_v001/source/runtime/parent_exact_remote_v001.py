"""Detached parent-only published-profile and retuning worker; original stack only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

BASE = Path('/content/Strassen_MM_Focus')
PINS = ['transformers==4.56.2', 'tokenizers==0.22.0', 'huggingface-hub==0.34.4',
        'safetensors==0.6.2', 'numpy==2.2.6', 'datasets==4.1.1', 'pyarrow==21.0.0',
        'sentencepiece==0.2.1', 'protobuf==6.32.1', 'requests==2.32.5']


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, log, env, timeout, cwd):
    with log.open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, cwd=cwd, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=20)
            return -124


def worker(run):
    import fcntl
    source = run/'source'
    artifacts = run/'artifacts'
    artifacts.mkdir()
    private = BASE/'.runtime_private'/run.name
    private.mkdir(parents=True)
    lock = (BASE/'active.lock').open('ab')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), stages=[], status='running')
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith('PIP_')}
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', JAX_PLATFORMS='tpu',
               HF_HOME=str(private/'huggingface'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    save(run/'status.json', status)
    try:
        if shutil.disk_usage(BASE).free < 90 * 1024**3:
            raise RuntimeError('Less than 90 GiB free for the public model cache and isolated environments')
        for name, jax_version, tpu_version in [('parent','0.7.2','0.0.21.1')]:
            target = private/name
            python = target/'bin/python'
            status['active'] = name+'-setup'; save(run/'status.json', status)
            setup = execute([sys.executable,'-m','venv','--without-pip',str(target)], artifacts/(name+'-venv.log'), env, 120, source)
            if setup == 0:
                setup = execute([sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                                 '--report',str(artifacts/(name+'-installation.json')),
                                 'jax=='+jax_version,'jaxlib=='+jax_version,'libtpu=='+tpu_version,*PINS],
                                artifacts/(name+'-install.log'), env, 900, source)
            status['stages'].append(dict(name=name+'-setup',returncode=setup))
            save(run/'status.json',status)
            if setup != 0:
                continue
            execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'], artifacts/(name+'-freeze.txt'),env,60,source)
            plan = [('published-'+s, s, False) for s in ['layer','streamed','quality']]
            plan += [('gate-tune','gate-tune',False), ('site-tune','site-tune',False)]
            plan += [('retuned-'+s, s, True) for s in ['layer','streamed','quality']]
            profile = artifacts/'retuned-profile.json'
            for title, stage, retuned in plan:
                if retuned and not profile.exists():
                    sys.path.insert(0, str(source/'tools'))
                    from parent_exact_stage_v001 import freeze_profile
                    freeze_profile(artifacts/'gate-tune/events.jsonl', artifacts/'site-tune/events.jsonl', profile)
                status['active'] = title; save(run/'status.json',status)
                started = time.monotonic()
                command = [str(python),str(source/'tools/parent_exact_stage_v001.py'),
                           '--stage',stage,'--output',str(artifacts/title),
                           '--cache',str(private/'checkpoint')]
                if retuned:
                    command += ['--profile',str(profile)]
                timeout = {'gate-tune':3600, 'site-tune':5400, 'streamed':2700,
                           'layer':1200, 'quality':1800}[stage]
                code = execute(command, artifacts/(title+'.log'),env,timeout,source)
                status['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started))
                save(run/'status.json',status)
                if code != 0:
                    raise RuntimeError('Parent stage failed: '+title+' (dependent phases were not run)')
        status['status'] = 'completed' if len(status['stages'])==9 and all(s['returncode']==0 for s in status['stages']) else 'partial_or_failed'
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        status.pop('active',None)
        save(run/'status.json',status)
        save(artifacts/'summary.json',status)
        archive=run.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as package:
            package.add(artifacts,arcname=run.name+'/artifacts')
            package.add(run/'launch.json',arcname=run.name+'/launch.json')
        save(run/'archive-ready.json',dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size))
        lock.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--run-id')
    p.add_argument('--worker',type=Path)
    a=p.parse_args()
    if a.worker:
        return worker(a.worker)
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id or ''):
        raise ValueError('Invalid run ID')
    if not a.archive.resolve().is_relative_to(BASE) or digest(a.archive)!=a.sha256:
        raise ValueError('Invalid source archive')
    run=BASE/'runs'/a.run_id
    run.mkdir(parents=True,exist_ok=False)
    source=run/'source';source.mkdir()
    with tarfile.open(a.archive) as package:
        for member in package.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe source archive member')
        package.extractall(source,filter='data')
    script=source/'runtime/parent_exact_remote_v001.py'
    save(run/'launch.json',dict(source_sha256=a.sha256,command=[sys.executable,str(script),'--worker',str(run)]))
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(script),'--worker',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='parent_replication_launched',run=str(run),pid=child.pid)),flush=True)


if __name__=='__main__':main()
