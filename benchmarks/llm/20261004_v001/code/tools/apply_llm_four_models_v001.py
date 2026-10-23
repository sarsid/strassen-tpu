"""Adopt the tested four-model scheduler without signalling the TPU controller."""
import argparse, hashlib, json, os, shutil, signal, subprocess, tarfile, time
from pathlib import Path
from llm_queue_state_v002 import read, save, ledger, utc
from llm_model_scope_v001 import revise, planned_entries
from run_llm_queue_v008 import reconcile


def process(pid):
    return subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True).stdout.strip()


def until(predicate, message):
    for _ in range(50):
        if predicate():return
        time.sleep(.1)
    raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--validation-dir',type=Path,required=True);parser.add_argument('--controller-python',required=True)
    args=parser.parse_args();campaign=args.campaign.resolve();queue=campaign/'queue_v001'
    validation=args.validation_dir.resolve();root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
    if read(validation/'completion.json').get('status')!='completed':raise ValueError('Validation did not pass')
    if read(validation/'artifacts/summary.json').get('status')!='completed':raise ValueError('Browser validation did not pass')
    destination=queue/'four_models_20261003_v001';destination.mkdir()
    shutil.copyfile(validation/'source.tar',destination/'source.tar')
    (destination/'source').mkdir()
    with tarfile.open(destination/'source.tar') as package:package.extractall(destination/'source',filter='data')
    manifest=read(validation/'source-manifest.json');source=destination/'source'
    digest=hashlib.sha256((destination/'source.tar').read_bytes()).hexdigest()
    if digest!=manifest['archive_sha256']:raise ValueError('Frozen source digest mismatch')
    statepath=queue/'state.json';before=read(statepath)
    watch=read(queue/'watchdog-process.json')['pid'];supervisor=read(queue/'supervisor-process.json')['pid']
    for pid,script in ((watch,'watch_llm_queue_v007.py'),(supervisor,'run_llm_queue_v007.py')):
        command=process(pid)
        if script not in command or str(campaign) not in command or command.startswith('T'):raise RuntimeError('Expected live orchestration process absent')
    controller=before['active_session']['pid'];run=campaign/before['active_session']['run']
    if 'run_llm_session_controller_v001.py' not in process(controller) or str(run) not in process(controller):raise RuntimeError('Active controller not verified')
    parked=False;applied=False
    try:
        os.kill(watch,signal.SIGSTOP);parked=True
        until(lambda:process(watch).startswith('T'),'Watchdog did not park')
        os.kill(supervisor,signal.SIGTERM)
        until(lambda:not process(supervisor) or process(supervisor).startswith('Z'),'Supervisor did not exit')
        before=read(statepath)
        if before['active_session']['pid']!=controller or before['status']!='running':raise RuntimeError('Active session changed')
        save(destination/'state-before.json',before)
        protocol=read(source/'configs/llm_v6e_v002.json')
        state=revise(before,protocol,read(campaign/'inputs_gemma4_v001/models.json')[0])
        old=[json.loads(line) for line in (campaign/'planned_grid_v001/planned_entries.jsonl').read_text().splitlines()]
        plan=planned_entries(state,old,protocol);target=campaign/state['active_plan'];target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in plan))
        state.setdefault('source_history',[]).append({k:before[k] for k in ('source_dir','source_archive','source_commit','source_sha256')})
        state.update(source_dir=str((destination/'source').relative_to(queue)),source_archive=str((destination/'source.tar').relative_to(queue)),source_commit=manifest['source_commit'],source_sha256=digest,scope_changed_utc=utc(),updated_utc=utc())
        reconcile(state,ledger(campaign/'completed_entries.jsonl'),read(run/'latest.json',{}).get('status.json',{}))
        save(statepath,state);save(destination/'state-after.json',state);applied=True
        os.kill(watch,signal.SIGTERM);os.kill(watch,signal.SIGCONT);parked=False
        until(lambda:not process(watch) or process(watch).startswith('Z'),'Old watchdog did not exit')
        command=[args.controller_python,str(source/'tools/watch_llm_queue_v008.py'),'--campaign',str(campaign),'--controller-python',args.controller_python]
        with (destination/'watchdog.log').open('ab') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        until(lambda:read(queue/'watchdog-process.json').get('pid')==child.pid,'New watchdog did not register')
        if not process(controller) or str(run) not in process(controller):raise RuntimeError('Controller liveness lost during migration')
        summary=dict(status='completed',utc=utc(),source_commit=manifest['source_commit'],validation=str(validation.relative_to(root)),controller_preserved=controller,retired_watchdog=watch,retired_supervisor=supervisor,new_watchdog=child.pid,model_order=state['model_order'],active_workloads=40,active_comparisons=200,deferred_workloads=len(state['deferred_jobs']),measured=state['measured_comparisons'],no_tpu_operation_issued=True)
        save(destination/'migration.json',summary);print(json.dumps(summary))
    finally:
        if parked:os.kill(watch,signal.SIGCONT)
        if applied and not (destination/'migration.json').exists():
            save(destination/'migration-incomplete.json',dict(utc=utc(),note='State applied; inspect orchestration before retrying. Controller was not signalled.'))

if __name__=='__main__':main()
