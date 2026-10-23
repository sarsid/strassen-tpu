"""After the main study, audit down VMEM caps on its still-owned runtime/cache."""
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/content/Strassen_MM_Focus')
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(run,mainrun):
    from parent_replication_remote_v001 import execute
    source=run/'source';out=run/'artifacts';out.mkdir();private=BASE/'.runtime_private'/mainrun.name
    status=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',active='await-main-completion',stages=[])
    save(run/'status.json',status);lock=None
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith('PIP_')}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(source/'src'),JAX_PLATFORMS='tpu',
               HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    try:
        deadline=time.monotonic()+900
        while not (mainrun/'archive-ready.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Main study did not finish')
            time.sleep(5)
        main_status=json.loads((mainrun/'status.json').read_text())
        if main_status['status']!='completed':raise RuntimeError('Main study must complete before the supplemental audit')
        lock=(BASE/'active.lock').open('ab')
        for attempt in range(20):
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.25)
        else:raise RuntimeError('Main worker still holds runtime lock')
        old_profile=mainrun/'artifacts/refine/profile.json';old_events=mainrun/'artifacts/tune/events.jsonl'
        for stage,name,actual in [('refine','profile.json',old_profile),('tune','events.jsonl',old_events)]:
            frozen=source/'runs'/mainrun.name/'stage-backups'/stage/name
            if digest(frozen)!=digest(actual):raise RuntimeError('Frozen audit input mismatch')
        save(out/'input-provenance.json',dict(main_run=str(mainrun),profile_sha256=digest(old_profile),events_sha256=digest(old_events)))
        # One declared unchanged-profile repeat diagnoses the observed 197 ms host-timed stall.
        # It is separate evidence and does not replace or delete the original pass.
        status['active']='s1-bfloat16-repeat';save(run/'status.json',status);started=time.monotonic()
        code=execute([str(private/'current/bin/python'),str(source/'tools/compare_parent_fused_v001.py'),
                      '--profile',str(old_profile),'--dtype','bfloat16','--family','s1','--stage','streamed',
                      '--output',str(out/'s1-bfloat16-repeat'),'--cache',str(private/'checkpoint')],
                     out/'s1-bfloat16-repeat.log',env,900,source)
        status['stages'].append(dict(name='s1-bfloat16-repeat',returncode=code,wall_seconds=time.monotonic()-started));save(run/'status.json',status)
        status['active']='budget';save(run/'status.json',status);started=time.monotonic()
        code=execute([str(private/'current/bin/python'),str(source/'tools/tune_parent_budget_v001.py'),
                      '--screen-events',str(old_events),'--screen-profile',str(old_profile),
                      '--output',str(out/'budget'),'--cache',str(private/'checkpoint')],out/'budget.log',env,1800,source)
        status['stages'].append(dict(name='budget',returncode=code,wall_seconds=time.monotonic()-started));save(run/'status.json',status)
        if code:raise RuntimeError('Budget audit failed; current profiles remain the final complete comparison')
        previous=json.loads(old_profile.read_text());new=json.loads((out/'budget/profile.json').read_text())
        changed=[]
        for dtype in ('bfloat16','float32'):
            for family in ('s1','s2'):
                if any(new['profiles'][dtype][f]!=previous['profiles'][dtype][f] for f in ('cubic',family)):
                    changed.append(dict(dtype=dtype,family=family))
        save(out/'changed-profiles.json',dict(changed=changed,rule='Only repeat model evaluations whose candidate or cubic policy changed; Native unchanged'))
        for item in changed:
            for stage in ('streamed','quality'):
                title=item['family']+'-'+item['dtype']+'-'+stage;status['active']=title;save(run/'status.json',status);started=time.monotonic()
                code=execute([str(private/'current/bin/python'),str(source/'tools/compare_parent_fused_v001.py'),
                              '--profile',str(out/'budget/profile.json'),'--dtype',item['dtype'],'--family',item['family'],'--stage',stage,
                              '--output',str(out/title),'--cache',str(private/'checkpoint')],out/(title+'.log'),env,900,source)
                status['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started));save(run/'status.json',status)
        status['status']='completed' if all(s['returncode']==0 for s in status['stages']) else 'partial_or_failed'
    except BaseException as e:status.update(status='failed',error_type=type(e).__name__,error=str(e))
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();status.pop('active',None);save(run/'status.json',status);save(out/'summary.json',status)
        archive=run.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as package:
            package.add(out,arcname=run.name+'/artifacts');package.add(run/'launch.json',arcname=run.name+'/launch.json')
        save(run/'archive-ready.json',dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size))
        if lock is not None:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--main-run-id',required=True)
    p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--worker',type=Path);a=p.parse_args()
    if any(not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',v) for v in (a.run_id,a.main_run_id)):raise ValueError('Invalid run ID')
    if a.worker:return worker(a.worker,BASE/'runs'/a.main_run_id)
    if not a.archive.resolve().is_relative_to(BASE) or digest(a.archive)!=a.sha256:raise ValueError('Invalid source archive')
    run=BASE/'runs'/a.run_id;run.mkdir(exist_ok=False);source=run/'source';source.mkdir()
    with tarfile.open(a.archive) as package:
        for m in package.getmembers():
            if Path(m.name).is_absolute() or '..' in Path(m.name).parts or not(m.isfile() or m.isdir()):raise ValueError('Unsafe archive member')
        package.extractall(source,filter='data')
    script=source/'runtime/parent_budget_remote_v002.py'
    save(run/'launch.json',dict(source_sha256=a.sha256,main_run=a.main_run_id))
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(script),'--run-id',a.run_id,'--main-run-id',a.main_run_id,'--worker',str(run)],
                               stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(kind='budget_audit_launched',pid=child.pid,run=str(run))),flush=True)


if __name__=='__main__':main()
