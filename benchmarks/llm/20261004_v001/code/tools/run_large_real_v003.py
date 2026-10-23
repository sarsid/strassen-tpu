"""Detached serial v5e campaign controller; checkpoint every immutable execution.

The Mac must stay powered and online. A caffeinate child prevents idle sleep.
Remote phase workers survive loss of this controller, but automatic phase
advancement and retrieval require the controller. Never retry an uncertain launch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import run_region_cohort_v001 as archive

ROOT=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):archive.write(Path(path),value)


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def commit(paths,message):
    relative=[str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','commit','--only','-m',message,'--',*relative],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


class StageArchive(archive.CohortArchive):
    def finish(self,path,status,**details):
        super().finish(path,status,**details)
        return commit([path,self.cohort/(self.phase+'-started.json'),self.cohort/(self.phase+'-finished.json')],
                      f'Archive MLSys {self.phase} {status}')


def stage_spec(cohort,stage_id):
    return next(s for s in read(cohort/'plan.json')['stages'] if s['id']==stage_id)


def checked_plan(cohort):
    frozen=archive.verify_frozen(cohort);plan=read(cohort/'plan.json')
    if plan!=read(cohort/'source/plan.json'):raise ValueError('Plan differs from frozen source')
    for key in ('session','allocation_id'):
        if plan.get(key)!=frozen.get(key):raise ValueError('Plan differs from frozen '+key)
    source=(cohort/'source/src').resolve()
    # Refuse a previously imported working-tree runner; do not silently reuse it.
    for name,module in tuple(sys.modules.items()):
        if name=='strassen_mm' or name.startswith('strassen_mm.'):
            path=getattr(module,'__file__',None)
            if path is None or not Path(path).resolve().is_relative_to(source):
                raise ValueError('Runner already imported outside frozen source: '+name)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    return frozen,plan


def phase(cohort,stage):
    frozen,plan=checked_plan(cohort);spec=next(s for s in plan['stages'] if s['id']==stage)
    transport_file=cohort/'source'/plan.get('transport','tools/run_phase_v004.py')
    modspec=importlib.util.spec_from_file_location('mlsys_transport',transport_file)
    transport=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(transport)
    transport.archive=StageArchive(cohort,stage)
    transport.emit=lambda *a,**kw:print(json.dumps(kw),flush=True)
    args=['run_phase_v004.py','--phase',stage,'--session',frozen['session'],'--endpoint',frozen['allocation_id'],
          '--expected-identity',plan['remote_identity'],'--controller-python',plan['controller_python'],
          '--campaign-relative',spec['campaign'],'--benchmark-module',spec['module'],
          '--timeout-seconds',str(spec['timeout_seconds'])]
    extra=list(spec.get('runner_args',[]))
    if spec.get('selection_stage'):
        prior=read(cohort/(spec['selection_stage']+'-finished.json'))
        if prior['status']!='completed':raise ValueError('Selection stage has not completed')
        extra += ['--selection','/content/Strassen_MM_Focus/runs/'+prior['run_id']+'/artifacts/'+spec.get('selection_file','selections.json')]
    for item in spec.get('input_stages',[]):
        prior=read(cohort/(item['stage']+'-finished.json'))
        if prior['status']!='completed':raise ValueError('Required input phase has not completed')
        extra += [item['flag'],'/content/Strassen_MM_Focus/runs/'+prior['run_id']+'/artifacts/'+item['file']]
    args += ['--runner-arg='+str(x) for x in extra]
    sys.argv=args
    return transport.main()


def read_new_rows(path,offset):
    """Only consume complete JSONL lines; a partial final line remains unread."""
    if not path.exists():return [],offset
    rows=[]
    with path.open('rb') as f:
        f.seek(offset)
        while True:
            start=f.tell();line=f.readline()
            if not line or not line.endswith(b'\n'):return rows,start
            rows.append(json.loads(line));offset=f.tell()


def counts(rows):
    terminal={}
    for r in rows:
        if r.get('event')=='case_result' and r.get('scope','call')=='call':
            terminal[(r.get('group_id'),r.get('seed'),r.get('arm_id'))]=r
    values=list(terminal.values())
    return dict(completed=len(values),succeeded=sum(r.get('status')=='ok' for r in values),
                failed=sum(r.get('status')!='ok' for r in values),
                measured=sum(r.get('timing',{}).get('sample_count',0)>0 for r in values))


def verify_phase(cohort,stage):
    receipt=read(cohort/(stage+'-finished.json'));run=Path(receipt['run'])
    if not run.resolve().is_relative_to(cohort/'phases'):raise ValueError('Phase path escapes cohort')
    if archive.sha(run/'completion.json')!=receipt['completion_sha256']:raise ValueError('Phase completion hash mismatch')
    completion=read(run/'completion.json')
    for relative,expected in read(run/'artifact-manifest.json')['sha256'].items():
        file=(run/relative).resolve()
        if not file.is_relative_to(run) or archive.sha(file)!=expected:raise ValueError('Phase artifact mismatch: '+relative)
    if receipt['status']!='completed' or completion.get('status')!='completed' or completion.get('remote_may_still_be_running') is not False:
        raise RuntimeError('Phase incomplete or remote execution uncertain; no automatic retry')
    summary=read(run/'artifacts/summary.json')
    if not summary.get('completed'):raise RuntimeError('Phase summary incomplete')
    return run,summary


def format_command(spec,cohort,plan):
    fields=dict(root=str(ROOT),source=str(cohort/'source'),cohort=str(cohort),
                controller_python=plan['controller_python'],analysis_python=plan['analysis_python'],
                session=plan['session'],endpoint=plan['allocation_id'],remote_identity=plan['remote_identity'])
    return [str(x).format(**fields) for x in spec['command']]


def expected_count(spec,cohort):
    if spec.get('module')!='strassen_mm.benchmark_mlsys_shapes_v001':return spec.get('expected')
    from strassen_mm.benchmark_mlsys_shapes_v001 import plan_groups
    cfg=read(cohort/'source'/spec['campaign']);shapes=read((cohort/'source'/spec['campaign']).parent/cfg['shape_manifest'])['shapes']
    selection=None
    if spec.get('selection_stage'):
        prior=read(cohort/(spec['selection_stage']+'-finished.json'))
        selection=read(Path(prior['run'])/'artifacts/selections.json')['selected']
    stage=spec['id'].rsplit('-',1)[-1]
    args=spec.get('runner_args',[])
    def number(flag,default):return int(args[args.index(flag)+1]) if flag in args else default
    groups=plan_groups(cfg,shapes,stage,selection,number('--shape-start',0),number('--shape-count',None))
    return sum(len(g['inputs'])*len(g['arms']) for g in groups)


def headline_results(rows,stage,phase_run):
    result={}
    for r in rows:
        if r.get('event')!='case_result' or r.get('scope','call')!='call':continue
        for role in r.get('headline_roles',[]):
            name={'native':'native_tuned','one_level':'strassen1','two_level':'strassen2'}.get(role,role)
            if name not in ('native_default','native_tuned','cubic','strassen1','strassen2'):continue
            meta=r.get('kernel_metadata') or {};k=(r.get('shape_id'),name)
            entry=result.setdefault(k,dict(shape_id=r.get('shape_id'),shape_mkn=r.get('shape_mkn') or meta.get('shape_mkn'),stage=stage,
                scope='call',selection_status='confirmed' if stage.endswith('confirm') else 'screen_selected',method=name,
                candidate_id=r['arm_id'],tile=r.get('tile',meta.get('tile_bm_bn_bk')),mean_ms=None,eligible=True,_weighted=0.,_count=0,
                evidence=[dict(label='Raw evidence',path=str((phase_run/'artifacts/results.jsonl').relative_to(ROOT)))]))
            timing=r.get('timing') or {};n=timing.get('sample_count',0)
            if n and timing.get('mean_ms') is not None:entry['_weighted']+=n*timing['mean_ms'];entry['_count']+=n
            entry['eligible']=entry['eligible'] and r.get('eligible_for_speedup_claim',False)
    for entry in result.values():
        entry['mean_ms']=entry['_weighted']/entry['_count'] if entry['_count'] else None
        del entry['_weighted'],entry['_count']
    return list(result.values())


def model_projection_results(rows,stage,phase_run,campaign_relative):
    """Display frozen real-projection choices, including absent/failed winners.

    The model journal does not attach headline_roles. Its immutable confirmation
    summaries supply those roles; original case records supply observed latency.
    Projection eligibility is separate from full-model qualification.
    """
    campaign_path=phase_run/'source'/campaign_relative;cfg=read(campaign_path)
    shapes={s['id']:s for s in read(campaign_path.parent/cfg['shape_manifest'])['shapes']}
    candidates={c['candidate_id']:c for family in cfg['candidate_families'].values() for c in family}
    role_names={'native_default':'native_default','native':'native_tuned','cubic':'cubic',
                'one_level':'strassen1','two_level':'strassen2'}
    expected_seeds=set(cfg['confirm_seeds']);repeats=cfg['timing']['confirm']['repeats'];result=[]
    for statistics in sorted((phase_run/'artifacts').glob('m*_confirmation_statistics.json')):
        for shape_id,summary in read(statistics)['by_shape'].items():
            shape=shapes[shape_id]
            for role,method in role_names.items():
                headline=summary['headline'].get(role,{});cid=headline.get('candidate_id')
                candidate=candidates[cid] if cid is not None else None
                observed=[r for r in rows if r.get('event')=='case_result' and r.get('scope','call')=='call'
                          and r.get('shape_id')==shape_id and r.get('group_id')==shape_id+'__confirm'
                          and r.get('arm_id')==cid]
                seed_complete=(len(observed)==len(expected_seeds) and {r.get('seed') for r in observed}==expected_seeds)
                valid=bool(headline.get('all_inputs_eligible') and headline.get('screen_selection_eligible') and seed_complete
                           and all(r.get('status')=='ok' and r.get('eligible_for_speedup_claim') is True
                                   and (r.get('timing') or {}).get('sample_count')==repeats for r in observed))
                weighted=0.;sample_count=0
                for row in observed:
                    timing=row.get('timing') or {};value=timing.get('mean_ms');n=timing.get('sample_count',0)
                    if type(value) in (float,int) and math.isfinite(value) and value>0 and type(n) is int and n>0:
                        weighted+=n*value;sample_count+=n
                metadata=next((r['kernel_metadata'] for r in observed if isinstance(r.get('kernel_metadata'),dict)),{})
                result.append(dict(shape_id=shape_id,shape_mkn=metadata.get('shape_mkn') or [shape[d] for d in ('m','k','n')],
                    stage=stage,scope='call',selection_status='confirmed' if headline.get('available') else 'unavailable',
                    method=method,candidate_id=cid,tile=metadata.get('tile_bm_bn_bk',candidate.get('tile') if candidate else None),
                    mean_ms=weighted/sample_count if sample_count else None,eligible=valid,
                    status='eligible_real_projection_only' if valid else 'ineligible_real_projection' if cid else 'no_frozen_winner',
                    sample_count=sample_count,fresh_input_count=len(observed),
                    interpretation='Layer-zero real checkpoint weights and recorded real activation windows; not full-model performance or quality.',
                    evidence=[dict(label='Frozen projection confirmation',path=str(statistics.relative_to(ROOT))),
                              dict(label='Raw real-operand cases',path=str((phase_run/'artifacts/results.jsonl').relative_to(ROOT)))]))
    return result


def live_detail(incoming):
    for row in reversed(incoming):
        kind=row.get('event');model=row.get('model_id') or 'Model'
        if kind=='resident_model_sample':
            return f"Resident {row.get('scope')}: {row.get('arm_id')} {row.get('elapsed_ms'):.3f} ms, round {row.get('repeat')}."
        if kind=='resident_layer_loaded':return f"Loading layer {row.get('layer')+1} into device memory, outside timing."
        if kind=='quality_window_complete':return f"Held-out text evaluation: {row.get('completed')}/{row.get('total')} windows."
        if kind=='quality_summary':return f"Held-out prediction/error metrics saved for {row.get('arm_id')}."
        if kind=='quality_layer_complete':
            return f"{model}: quality propagation completed layer {row.get('completed')}/{row.get('total')} across {row.get('windows')} fixed windows. MM attempt counts are unchanged."
        if kind=='native_qualification':
            return ('Independent Native model qualification passed; real-projection tuning follows.' if row.get('passed') is True else
                    'Independent Native qualification did not pass. Real-projection tuning continues; full-model quality/performance claims remain blocked.')
        if kind=='streamed_model_sample':
            return f"Streamed model timing: {row.get('arm_id')}, repeat {row.get('repeat')}; model timing is separate from MM candidate counts."
        if kind=='quality_window':return f"Quality metrics recorded for fixed window {row.get('window')}; MM candidate counts are unchanged."
        if kind=='real_operand_capture':return f"{model}: captured real layer-{row.get('layer')} operands from fixed window {row.get('window')}."
        if kind=='model_projection_compile':return 'Compiling model projection '+str(row.get('candidate_id'))+'; compilation does not add completed measurements.'
        if kind in ('preparation_start','preparation_complete','prep_step','preparation_step'):
            return 'Input preparation: '+str(row.get('step') or row.get('name') or kind)+'; excluded from measurement progress.'
    return None


def run(cohort):
    _,plan=checked_plan(cohort)
    supervised = plan.get('automatic_recovery', False)
    lock=(cohort/'controller.lock').open('ab');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (cohort/'controller-completion.json').exists():raise FileExistsError('Campaign is sealed')
    write(cohort/'controller-started.json',dict(pid=os.getpid(),started_utc=utc(),source_commit=read(cohort/'frozen.json')['baseline_commit']))
    sleep_guard=None
    if Path('/usr/bin/caffeinate').exists():
        sleep_guard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        write(cohort/'sleep-prevention.json',dict(pid=sleep_guard.pid,mode='Prevent idle sleep; lid must remain open, powered and online'))
    journal=(cohort/'controller-events.jsonl').open('a',buffering=1)
    def event(kind,**data):journal.write(json.dumps(dict(event=kind,utc=utc(),**data))+'\n')
    state=dict(schema_version=1,campaign_id=cohort.name,title=plan.get('title','Large LLM speed and accuracy'),cohort_path=str(cohort.relative_to(ROOT)),
               state='running',stage=None,updated_utc=utc(),heartbeat_utc=utc(),worker_last_data_utc=None,
               detail='Detached real-weight speed and accuracy campaign; both Strassen depths and Native controls.',next_step='',
               selected_results=[],stages=[dict(id=s['id'],label=s.get('label',s['id']),kind=s.get('kind','measurement'),state='not_started',
                   expected=s.get('expected'),completed=0,succeeded=0,failed=0,measured=0,detail='',evidence=[]) for s in plan['stages']])
    def publish():
        state.update(updated_utc=utc(),heartbeat_utc=utc());atomic(cohort/'progress.json',state)
    publish();event('controller_started',pid=os.getpid())
    stopped=None;issues=[];remote_uncertain=False;release_verified=False;child=None
    try:
        for index,spec in enumerate(plan['stages']):
            stage=state['stages'][index];stage.update(state='running',detail=spec.get('detail',''))
            failed_dependencies=[d for d in spec.get('requires',[]) if next(s for s in state['stages'] if s['id']==d)['state']!='succeeded']
            if failed_dependencies:
                stage.update(state='blocked',detail='Prerequisite failed: '+', '.join(failed_dependencies));issues.append(dict(stage=spec['id'],reason=stage['detail']))
                event('stage_blocked',stage=spec['id'],dependencies=failed_dependencies);publish();continue
            stage['expected']=expected_count(spec,cohort)
            state.update(stage=spec['id'],detail=spec.get('detail',spec.get('label',spec['id'])),next_step='Archive and verify this stage, then advance automatically.')
            operation=cohort/'operations'/spec['id'];operation.mkdir(parents=True,exist_ok=False)
            command=([plan['controller_python'],str(cohort/'source/tools/run_large_real_v002.py'),'phase','--cohort',str(cohort),'--stage',spec['id']]
                     if spec['operation']=='phase' else format_command(spec,cohort,plan))
            write(operation/'execution.json',dict(command=command,started_utc=utc(),spec=spec))
            event('stage_started',stage=spec['id']);publish()
            env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(ROOT),STRASSEN_EXECUTION_DIR=str(operation),
                     PYTHONPATH=str(cohort/'source/src'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (operation/'execution.log').open('x') as log:
                if spec['operation']=='phase':remote_uncertain=True
                child=subprocess.Popen(command,cwd=cohort/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(operation/'process.json',dict(pid=child.pid))
                offset=0;rows=[];started=time.monotonic()
                while child.poll() is None:
                    receipt=cohort/(spec['id']+'-started.json')
                    if receipt.exists():
                        phase_run=Path(read(receipt)['run']);incoming,offset=read_new_rows(phase_run/'results.jsonl',offset)
                        rows.extend(incoming);stage.update(counts(rows))
                        if incoming:
                            last=next((r.get('utc') for r in reversed(incoming) if r.get('utc')),None)
                            stage['worker_last_data_utc']=last;state['worker_last_data_utc']=last
                            detail=live_detail(incoming)
                            if detail:stage['detail']=detail;state['detail']=detail
                        stage['evidence']=[dict(label='Live phase log',path=str((phase_run/'benchmark.log').relative_to(ROOT)))]
                    if time.monotonic()-started>spec.get('controller_timeout_seconds',spec.get('timeout_seconds',3600)+1800):
                        # Stop advancing, but preserve a possibly active remote worker.
                        event('controller_step_timeout',stage=spec['id'],child_pid=child.pid)
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise TimeoutError('Controller step exceeded budget; inspect recorded child/remote state before recovery')
                    publish();time.sleep(5)
            code=child.returncode
            error=None;quality_blocked=False
            try:
                if spec['operation']=='phase' and (cohort/(spec['id']+'-finished.json')).exists():
                    finished=read(cohort/(spec['id']+'-finished.json'))
                    remote_uncertain=read(Path(finished['run'])/'completion.json').get('remote_may_still_be_running',True)
                if code:raise RuntimeError('Stage command exited '+str(code))
                if spec['operation']=='phase':
                    phase_run,summary=verify_phase(cohort,spec['id'])
                    complete_rows=[json.loads(x) for x in (phase_run/'artifacts/results.jsonl').read_text().splitlines()]
                    stage.update(counts(complete_rows))
                    if spec.get('progress_schema','standard_mm')=='standard_mm':
                        groups=read(phase_run/'artifacts/planned_cases.json')
                        stage['expected']=sum(len(g['inputs'])*len(g['arms']) for g in groups)
                        if stage['completed']!=stage['expected']:raise RuntimeError('Terminal case count differs from frozen plan')
                    elif spec.get('kind','measurement')=='measurement':
                        # Adaptive real-model finalists determine the eventual MM count.
                        # This is a terminal count after completion, never a forecast.
                        stage['expected']=stage['completed']
                        stage['detail']='Completed actual-model phase; MM count includes its realized screen and confirmation attempts.'
                    stage['evidence']=[dict(label='Phase summary',path=str((phase_run/'artifacts/summary.json').relative_to(ROOT)))]
                    write(operation/'verified-phase.json',dict(run=str(phase_run),summary=summary,verified_utc=utc()))
                    if spec.get('module')=='strassen_mm.benchmark_large_models_v001' or (spec.get('module')=='strassen_mm.benchmark_llm_tradeoff_v001' and 'tune' in spec.get('runner_args',[])):
                        state['selected_results'].extend(model_projection_results(complete_rows,spec['id'],phase_run,spec['campaign']))
                    else:
                        state['selected_results'].extend(headline_results(complete_rows,spec['id'],phase_run))
                    model=summary.get('model') or {}
                    if model.get('status')=='blocked_full_model_quality':
                        quality_blocked=True
                        reason=('Independent Native qualification passed, but an eligible Native model policy was unavailable'
                                if (model.get('qualification') or {}).get('passed') is True else 'Independent Native model qualification did not pass')
                        detail=reason+'; full-model quality/performance is blocked (quality_measured=false). Completed real-projection MM attempts and their numerical eligibility are retained.'
                        stage.update(detail=detail,quality_measured=False,model_status=model['status'])
                        state['detail']=detail
                        issues.append(dict(stage=spec['id'],reason=detail))
                if spec.get('releases_allocation'):
                    release_verified='"verified_absent": true' in (operation/'execution.log').read_text()
                    if not release_verified:raise RuntimeError('Release command returned without verified absence')
                stage['state']='attention' if quality_blocked else 'succeeded'
            except Exception as exc:
                error=dict(type=type(exc).__name__,message=str(exc));stage.update(state='failed',detail=str(exc));issues.append(dict(stage=spec['id'],**error))
            write(operation/'completion.json',dict(status=stage['state'],exit_code=code,error=error,finished_utc=utc()))
            event('stage_finished',stage=spec['id'],state=stage['state'],counts={k:stage[k] for k in ('completed','expected','succeeded','failed','measured')})
            publish();write(operation/'progress-snapshot.json',state)
            commit([operation,cohort/'controller-events.jsonl'],f'Checkpoint MLSys {spec["id"]} {stage["state"]}')
            if remote_uncertain:raise RuntimeError(spec['id']+' remote execution is uncertain; automatic advancement stopped')
            if error and not spec.get('continue_on_failure',False):raise RuntimeError(spec['id']+' failed; automatic advancement stopped')
        state['state']='attention' if issues else 'succeeded'
    except BaseException as exc:
        stopped=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        event('controller_stopped',**stopped);state.update(state='attention',detail=str(exc),next_step='Inspect preserved evidence before recovery; no automatic new allocation or duplicate execution.')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        # Release a known-idle runtime after failure; never discard uncertain work.
        if not release_verified and not remote_uncertain and not supervised:
            cleanup=cohort/'failure-cleanup';cleanup.mkdir(exist_ok=False)
            command=[plan['controller_python'],str(cohort/'source'/plan.get('release','runtime/release_allocation_v001.py')),
                     '--session',plan['session'],'--expect-endpoint',plan['allocation_id']]
            write(cleanup/'execution.json',dict(command=command,started_utc=utc()))
            with (cleanup/'execution.log').open('x') as log:
                try:
                    done=subprocess.run(command,cwd=cohort/'source',stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    release_verified=done.returncode==0 and '"verified_absent": true' in (cleanup/'execution.log').read_text()
                except Exception as exc:event('cleanup_error',type=type(exc).__name__,message=str(exc))
            write(cleanup/'completion.json',dict(released=release_verified,finished_utc=utc()))
            commit([cleanup],'Archive MLSys idle-allocation cleanup')
        status='completed' if state['state']=='succeeded' else 'attention'
        if not release_verified and not supervised:status='attention';state['state']='attention'
        state.update(allocation_released=release_verified,remote_execution_uncertain=remote_uncertain)
        write(cohort/'controller-completion.json',dict(status=status,finished_utc=utc(),error=stopped,issues=issues,
              allocation_released=release_verified,remote_execution_uncertain=remote_uncertain))
        event('controller_finished',status=status);publish();journal.close()
        commit([cohort/'controller-completion.json',cohort/'controller-started.json',cohort/'controller-events.jsonl',cohort/'progress.json'],f'Finish MLSys unattended controller {status}')
        if sleep_guard is not None:sleep_guard.terminate()
    return 0 if status=='completed' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['run','phase']);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--stage')
    args=p.parse_args();cohort=args.cohort.resolve()
    if not cohort.is_relative_to(ROOT/'runs'):raise ValueError('Cohort must be within project runs')
    return phase(cohort,args.stage) if args.operation=='phase' else run(cohort)


if __name__=='__main__':raise SystemExit(main())
