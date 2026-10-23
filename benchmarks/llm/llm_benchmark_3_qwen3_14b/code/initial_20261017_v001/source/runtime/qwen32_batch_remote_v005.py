"""Large-batch retry worker: reused profiles, preflight, immutable new exports."""
import argparse
from datetime import datetime, timezone
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
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def package(path,entries):
    with tarfile.open(path,'x:gz') as out:
        for file,name in entries: out.add(file,arcname=name)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def retry_stages(plan):
    if plan['batches'] != [64] or plan['stages'] != ['qualify', 'confirm', 'quality', 'prefill']:
        raise ValueError('Only B64 frozen-profile confirmation stages are registered')
    return tuple(plan['stages'])


def retry_profile(path, plan, batch):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['reused_profile']['sha256']:
        raise ValueError('Recovered profile hash mismatch')
    data = json.loads(path.read_text())
    if (data['batch'], data['sequence']) != (batch, plan['sequence']):
        raise ValueError('Profile workload mismatch')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def worker(run):
    from parent_exact_remote_v001 import execute, PINS
    source=run/'source'; artifacts=run/'artifacts'; artifacts.mkdir()
    lock=(BASE/'active.lock').open('ab'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    private=BASE/'.runtime_private'/run.name; private.mkdir(parents=True)
    plan=json.loads((source/'configs/qwen32_b64_profile_reuse_v007.json').read_text())
    state=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',stages=[],batches=[],batch_archives=[])
    save(run/'status.json',state)
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith(('PIP_','QWEN3_'))}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',JAX_PLATFORMS='tpu',HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    try:
        if shutil.disk_usage(BASE).free<90*1024**3: raise RuntimeError('Need 90 GiB free checkpoint/environment space')
        state['active']='parent-setup'; save(run/'status.json',state)
        target=private/'parent'; python=target/'bin/python'
        code=execute([sys.executable,'-m','venv','--without-pip',str(target)],artifacts/'parent-venv.log',env,120,source)
        if code==0:
            code=execute([sys.executable,'-m','pip','--isolated','--python',str(python),'install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:','--report',str(artifacts/'installation.json'),'jax==0.11.2','jaxlib==0.11.2','libtpu==0.0.48',*PINS],artifacts/'parent-install.log',env,900,source)
        if code: raise RuntimeError('Current-stack environment setup failed')
        execute([sys.executable,'-m','pip','--isolated','--python',str(python),'freeze','--all'],artifacts/'freeze.txt',env,60,source)
        shutil.copy2(source/'configs/qwen32_b64_profile_reuse_v007.json',artifacts/'plan.json')
        for batch in plan['batches']:
            outcomes=[]; entries=[]
            profile,profile_sha256=retry_profile(source/plan['reused_profile']['path'],plan,batch)
            copied=artifacts/f'b{batch}-frozen-profile.json'; shutil.copy2(profile,copied)
            entries.append((copied,copied.name))
            save(artifacts/'profile-reuse.json',plan['reused_profile'])
            state['profile_reuse']=plan['reused_profile']; save(run/'status.json',state)
            for stage in retry_stages(plan):
                success=False
                for attempt in range(plan['per_stage_retry_limit']+1):
                    title=f'b{batch}-{stage}-a{attempt}'; output=artifacts/title
                    state['active']=title; state['active_batch']=batch; save(run/'status.json',state)
                    command=[str(python),str(source/'tools/qwen32_batch_stage_v004.py'),'--stage',stage,'--batch',str(batch),'--sequence',str(plan['sequence']),'--output',str(output),'--cache',str(private/'checkpoint'),'--plan',str(source/'configs/qwen32_b64_profile_reuse_v007.json')]
                    if profile is not None: command+=['--profile',str(profile),'--profile-sha256',profile_sha256]
                    start=time.monotonic()
                    code=execute(command,artifacts/(title+'.log'),env,plan['stage_timeouts_seconds'][stage],source)
                    row=dict(name=title,batch=batch,stage=stage,attempt=attempt,returncode=code,seconds=time.monotonic()-start)
                    outcomes.append(row); state['stages'].append(row); save(run/'status.json',state)
                    entries.append((artifacts/(title+'.log'),title+'.log'))
                    if output.exists(): entries.append((output,title))
                    if code==0:
                        success=True; break
                    completion=output/'completion.json'
                    if completion.exists() and 'Nonfinite' in json.loads(completion.read_text()).get('error',''):
                        break
                if not success: break
            result=dict(batch=batch,status='completed' if success and stage=='prefill' else 'partial_or_failed',attempts=outcomes)
            summary=artifacts/f'b{batch}-summary.json'; save(summary,result); entries.append((summary,summary.name))
            exported=package(run/f'b{batch}-results.tar.gz',entries)
            state['batch_archives'].append(dict(batch=batch,**exported)); state['batches'].append(result)
            save(run/'status.json',state)
        state['status']='completed' if all(b['status']=='completed' for b in state['batches']) else 'completed_with_failures'
    except BaseException as e:
        state.update(status='failed',error_type=type(e).__name__,error=str(e)[-2000:])
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat(); state.pop('active',None)
        save(run/'status.json',state); save(artifacts/'summary.json',state)
        save(run/'archive-ready.json',package(run.with_suffix('.tar.gz'),[(artifacts,run.name+'/artifacts'),(run/'launch.json',run.name+'/launch.json')]))
        lock.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path); p.add_argument('--sha256'); p.add_argument('--run-id'); p.add_argument('--worker',type=Path); a=p.parse_args()
    if a.worker: return worker(a.worker)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',a.run_id or ''): raise ValueError('Invalid run ID')
    if not a.archive.resolve().is_relative_to(BASE) or hashlib.sha256(a.archive.read_bytes()).hexdigest()!=a.sha256: raise ValueError('Invalid source archive')
    run=BASE/'runs'/a.run_id; run.mkdir(parents=True,exist_ok=False); source=run/'source'; source.mkdir()
    with tarfile.open(a.archive) as t: t.extractall(source,filter='data')
    script=source/'runtime/qwen32_batch_remote_v005.py'
    save(run/'launch.json',dict(source_sha256=a.sha256,command=[sys.executable,str(script),'--worker',str(run)]))
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(script),'--worker',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='qwen32_batch_launched',run=str(run),pid=child.pid)),flush=True)


if __name__=='__main__': main()
