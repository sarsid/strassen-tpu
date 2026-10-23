"""Audited handoff after v004 tuning; reuse the owned runtime and private cache.

The original supervisor is suspended while its isolated tuning child finishes.
Only after the child's successful summary exists and it has exited is the
supervisor retired. A separately frozen source archive runs refinement and the
eight evaluations. The existing local controller still downloads and releases.
"""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/content/Strassen_MM_Focus')


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def command(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError,ProcessLookupError):return []


def find(script,argument):
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        args=command(path.name)
        if len(args)>1 and args[1]==str(script) and str(argument) in args:matches.append(int(path.name))
    if len(matches)!=1:raise RuntimeError(f'Expected one exact process match for {script.name}, found {len(matches)}')
    return matches[0]


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ')[1].split()[0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False


def worker(run,extension):
    sys.path.insert(0,str(extension/'source/runtime'))
    from parent_replication_remote_v001 import execute
    receipt=json.loads((extension/'handoff.json').read_text())
    supervisor=receipt['old_supervisor_pid'];child=receipt['tuning_pid']
    out=run/'artifacts';source=extension/'source';private=BASE/'.runtime_private'/run.name
    status=json.loads((run/'status.json').read_text());lock=None
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','LIBTPU_INIT_ARGS','XLA_FLAGS','JAX_PLATFORMS') and not k.startswith('PIP_')}
    env.update(PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(source/'src'),JAX_PLATFORMS='tpu',
               HF_HOME=str(private/'huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    try:
        deadline=time.monotonic()+5400
        while alive(child):
            if time.monotonic()>deadline:raise TimeoutError('Original tuning did not finish within handoff wait budget')
            time.sleep(2)
        summary=json.loads((out/'tune/summary.json').read_text())
        if summary['status']!='completed':raise RuntimeError('Original tuning summary is incomplete')
        expected=str(run/'source/runtime/parent_comparison_remote_v003.py')
        args=command(supervisor)
        if len(args)<2 or args[1]!=expected or str(run) not in args:raise RuntimeError('Supervisor identity changed')
        os.kill(supervisor,signal.SIGKILL)
        lock=(BASE/'active.lock').open('ab')
        for attempt in range(20):
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.25)
        else:raise RuntimeError('Original runtime lock was not released')
        receipt.update(state='refining',original_tuning_completed=True,handoff_utc=datetime.now(timezone.utc).isoformat(),
                       original_profile_sha256=hashlib.sha256((out/'tune/profile.json').read_bytes()).hexdigest())
        save(extension/'handoff.json',receipt);save(out/'refinement-handoff.json',receipt)
        status['stages'].append(dict(name='tune',returncode=0,completion_evidence='tune/summary.json; isolated child exited before handoff'))
        phases=[('refine',[str(source/'tools/tune_parent_fused_v005.py'),
                    '--screen-events',str(out/'tune/events.jsonl'),'--screen-profile',str(out/'tune/profile.json')],2700)]
        for dtype in ('bfloat16','float32'):
            for family in ('s1','s2'):
                for stage in ('streamed','quality'):
                    phases.append((family+'-'+dtype+'-'+stage,[str(source/'tools/compare_parent_fused_v001.py'),
                        '--profile',str(out/'refine/profile.json'),'--dtype',dtype,'--family',family,'--stage',stage],2400))
        for title,argv,timeout in phases:
            status['active']=title;save(run/'status.json',status)
            started=time.monotonic()
            code=execute([str(private/'current/bin/python'),*argv,'--output',str(out/title),'--cache',str(private/'checkpoint')],
                         out/(title+'.log'),env,timeout,source)
            status['stages'].append(dict(name=title,returncode=code,wall_seconds=time.monotonic()-started))
            save(run/'status.json',status)
            if title=='refine' and (code or not (out/'refine/profile.json').exists()):raise RuntimeError('Refinement did not complete; no partial profile is evaluated')
        status['status']='completed' if all(s['returncode']==0 for s in status['stages']) else 'partial_or_failed'
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        # Both identities were captured from exact owned commands before stopping.
        if alive(child) and command(child)==receipt['tuning_command']:
            os.killpg(child,signal.SIGTERM)
        if command(supervisor)==receipt['supervisor_command']:os.kill(supervisor,signal.SIGKILL)
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();status.pop('active',None)
        save(run/'status.json',status);save(out/'summary.json',status)
        archive=run.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as package:
            package.add(out,arcname=run.name+'/artifacts')
            package.add(run/'launch.json',arcname=run.name+'/launch.json')
            package.add(extension,arcname=run.name+'/refinement-extension')
        save(run/'archive-ready.json',dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
        if lock is not None:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--archive',type=Path)
    p.add_argument('--sha256');p.add_argument('--worker',type=Path);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid run ID')
    run=BASE/'runs'/a.run_id
    if a.worker:return worker(run,a.worker)
    if not a.archive.resolve().is_relative_to(BASE) or hashlib.sha256(a.archive.read_bytes()).hexdigest()!=a.sha256:raise ValueError('Invalid extension archive')
    if json.loads((run/'status.json').read_text()).get('active')!='tune':raise RuntimeError('Handoff only allowed during the original tune phase')
    supervisor=find(run/'source/runtime/parent_comparison_remote_v003.py',run)
    child=find(run/'source/tools/tune_parent_fused_v004.py',run/'artifacts/tune')
    if os.getpgid(child)!=child:raise RuntimeError('Tuning child is not an isolated process group')
    extension=run/'refinement-extension';extension.mkdir(exist_ok=False);source=extension/'source';source.mkdir()
    with tarfile.open(a.archive) as archive:
        for member in archive.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or not(member.isfile() or member.isdir()):raise ValueError('Unsafe member')
        archive.extractall(source,filter='data')
    receipt=dict(state='waiting_for_tune',old_supervisor_pid=supervisor,tuning_pid=child,
                 supervisor_command=command(supervisor),tuning_command=command(child),source_sha256=a.sha256,
                 requested_utc=datetime.now(timezone.utc).isoformat(),
                 reason='Preserve complete v004 screen; confirm shortlist without noisy Native normalization before model evaluation')
    save(extension/'handoff.json',receipt)
    script=source/'runtime/refine_handoff_v001.py'
    suspended=False
    try:
        os.kill(supervisor,signal.SIGSTOP);suspended=True
        with (extension/'worker.log').open('x') as log:
            helper=subprocess.Popen([sys.executable,str(script),'--run-id',a.run_id,'--worker',str(extension)],
                                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(kind='refinement_handoff_armed',helper_pid=helper.pid,**receipt)),flush=True)
    except BaseException:
        if suspended:os.kill(supervisor,signal.SIGCONT)
        raise


if __name__=='__main__':main()
