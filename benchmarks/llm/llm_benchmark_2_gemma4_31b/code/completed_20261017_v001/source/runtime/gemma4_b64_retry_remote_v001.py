"""B64-only recovery on the retained Gemma TPU; no allocation or release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

BASE = Path('/content/Strassen_MM_Focus')


def preflight(base, spec):
    predecessor = base / 'runs' / spec['predecessor_run']
    state = json.loads((predecessor / 'status.json').read_text())
    if state.get('status') != 'completed_with_failures' or not (predecessor / 'archive-ready.json').exists():
        raise RuntimeError('Predecessor is not finished and archived; do not run concurrently')
    batches = {r['batch']: r for r in state['batches']}
    if set(batches) != {1, 4, 8, 16, 32, 64}:
        raise RuntimeError('Unexpected predecessor batch set')
    if any(batches[b]['status'] != 'completed' or batches[b]['quality_pass'] is not True for b in (1, 4, 8, 16, 32)):
        raise RuntimeError('Resolve other failed batches before B64-only recovery')
    if batches[64]['status'] != 'partial_or_failed':
        raise RuntimeError('B64 recovery is not needed')
    required = {'b64-tune-a0', 'b64-qualify-a0'}
    if required != {r['name'] for r in state['stages'] if r['name'] in required and r['returncode'] == 0}:
        raise RuntimeError('Completed B64 tuning and qualification are required')
    profile = predecessor / 'artifacts/b64-tune-a0/profile.json'
    if hashlib.sha256(profile.read_bytes()).hexdigest() != spec['profile_sha256']:
        raise RuntimeError('Saved B64 profile hash changed')
    p = json.loads(profile.read_text())
    if (p['batch'], p['sequence'], p['model'], p['revision']) != (64, 1024, spec['model'], spec['revision']):
        raise RuntimeError('Saved profile identity mismatch')
    private = base / '.runtime_private' / spec['predecessor_run']
    if not (private / 'parent/bin/python').is_file() or not (private / 'checkpoint').is_dir():
        raise RuntimeError('Existing environment and checkpoint cache are required')
    return profile, private


def worker(run):
    import fcntl
    import shutil
    from qwen32_batch_remote_v005 import save, package
    from parent_exact_remote_v001 import execute
    source = run / 'source'
    spec = json.loads((source / 'configs/gemma4_b64_metric_retry_v001.json').read_text())
    profile, private = preflight(BASE, spec)
    lock = (BASE / 'active.lock').open('ab')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    artifacts = run / 'artifacts'; artifacts.mkdir()
    plan = json.loads((source / 'configs/gemma4_parent_batch_v003.json').read_text())
    shutil.copy2(profile, artifacts / 'reused-profile.json')
    profile = artifacts / 'reused-profile.json'
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 predecessor=spec['predecessor_run'], qualification='not_started',
                 stages=[], batches=[], batch_archives=[], profile_sha256=spec['profile_sha256'])
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith(('PIP_','QWEN3_'))}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',JAX_PLATFORMS='tpu',HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    python = private / 'parent/bin/python'
    entries = [(profile, profile.name)]
    save(run / 'status.json', state)
    try:
        quality_pass = True
        for stage in ('oracle', 'qualify', 'confirm', 'quality', 'prefill'):
            succeeded = False
            for attempt in range(plan['per_stage_retry_limit'] + 1):
                name = f'oracle-a{attempt}' if stage == 'oracle' else f'b64-{stage}-a{attempt}'
                output = artifacts / name
                state.update(active=name, active_batch=None if stage == 'oracle' else 64)
                save(run / 'status.json', state)
                save(run / 'progress.json', dict(status='running',stage=stage,batch=None if stage=='oracle' else 64,stage_label='B64 recovery: '+stage,utc=datetime.now(timezone.utc).isoformat()))
                cmd = [str(python),str(source/'tools/gemma4_batch_stage_v004.py'),'--stage',stage,'--batch','64','--sequence','1024','--output',str(output),'--cache',str(private/'checkpoint'),'--plan',str(source/'configs/gemma4_parent_batch_v003.json'),'--progress',str(run/'progress.json')]
                if stage != 'oracle':cmd += ['--profile',str(profile),'--profile-sha256',spec['profile_sha256']]
                start = time.monotonic()
                code = execute(cmd,artifacts/(name+'.log'),env,plan['stage_timeouts_seconds'][stage],source)
                state['stages'].append(dict(name=name,batch=None if stage=='oracle' else 64,stage=stage,attempt=attempt,returncode=code,seconds=time.monotonic()-start))
                save(run / 'status.json',state)
                entries.append((artifacts/(name+'.log'),name+'.log'))
                if output.exists():entries.append((output,name))
                if code == 0:succeeded=True;break
                completion=json.loads((output/'completion.json').read_text()) if (output/'completion.json').exists() else {}
                if any(v in completion.get('error','') for v in ('Scientific','Nonfinite','Unregistered','source mismatch')):break
            if not succeeded:raise RuntimeError('B64 recovery stage failed: '+stage)
            if stage == 'oracle':state['qualification']='passed'
            if stage in ('confirm','quality'):
                quality_pass &= bool(json.loads((output/'summary.json').read_text())['all_pass'])
        state['batches'] = [dict(batch=64,status='completed',quality_pass=quality_pass)]
        state['status'] = 'completed' if quality_pass else 'quality_failed'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error)[-2000:])
        if not state['batches']:state['batches']=[dict(batch=64,status='partial_or_failed',quality_pass=None)]
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();state.pop('active',None)
        save(artifacts/'b64-summary.json',state);entries.append((artifacts/'b64-summary.json','b64-summary.json'))
        exported=package(run/'b64-results.tar.gz',entries)
        state['batch_archives'].append(dict(batch=64,**exported))
        save(run/'status.json',state);save(artifacts/'summary.json',state)
        save(run/'archive-ready.json',package(run.with_suffix('.tar.gz'),[(artifacts,run.name+'/artifacts'),(run/'launch.json',run.name+'/launch.json')]))
        lock.close()


def launch(archive, sha256, run_id, base=BASE):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',run_id):raise ValueError('Invalid run ID')
    archive=Path(archive).resolve();base=Path(base).resolve()
    if not archive.is_relative_to(base) or hashlib.sha256(archive.read_bytes()).hexdigest()!=sha256:
        raise ValueError('Invalid source archive')
    run=base/'runs'/run_id
    if run.exists():raise FileExistsError('Recovery run already exists; do not relaunch')
    with tarfile.open(archive) as handle:
        spec=json.load(handle.extractfile('configs/gemma4_b64_metric_retry_v001.json'))
    if run_id!=spec['run_id']:raise ValueError('Unregistered recovery identity')
    preflight(base,spec)
    run.mkdir(parents=True,exist_ok=False);source=run/'source';source.mkdir()
    with tarfile.open(archive) as handle:handle.extractall(source,filter='data')
    script=source/'runtime/gemma4_b64_retry_remote_v001.py'
    receipt=dict(source_sha256=sha256,predecessor=spec['predecessor_run'],profile_sha256=spec['profile_sha256'],command=[sys.executable,str(script),'--worker',str(run)])
    (run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen(receipt['command'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='gemma4_b64_retry_launched',run=str(run),pid=child.pid)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path);parser.add_argument('--sha256');parser.add_argument('--run-id');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:launch(args.archive,args.sha256,args.run_id)


if __name__=='__main__':main()
