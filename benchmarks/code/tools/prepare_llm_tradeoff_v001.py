"""Freeze and launch the resident speed/accuracy campaign, with explicit hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
from run_region_cohort_v001 import write, read, sha, utc, verify_frozen
from run_large_real_v002 import commit
ROOT=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()


def build_plan(args):
    private='/content/Strassen_MM_Focus/.runtime_private/'+args.cohort.name
    cfg='configs/llm_tradeoff_'+args.hardware+'_v001/campaign.json'
    stages=[dict(id='MODEL-tools',label='Install isolated official reference tools',operation='phase',kind='preparation',
        campaign=cfg,module='strassen_mm.benchmark_large_prepare_v002',timeout_seconds=3600,progress_schema='application',
        runner_args=['--action','tools','--private-root',private])]
    titles={'qwen':'Qwen3-8B','mistral':'Mistral-7B-v0.3','gemma':'Gemma 3-12B text','qwen14':'Qwen3-14B'}
    for key in args.models:
        title=titles[key]
        if key=='gemma':
            stages.append(dict(id='gemma-access',label='Check authorized private Gemma access',operation='command',kind='preparation',
                timeout_seconds=600,continue_on_failure=True,command=['{controller_python}','{source}/tools/wait_large_gemma_access_v001.py','--cohort','{cohort}']))
        prep=key+'-inputs';tune=key+'-tune'
        stages.append(dict(id=prep,label=title+': exact weights, text and official reference',operation='phase',kind='preparation',
            campaign=cfg,module='strassen_mm.benchmark_large_prepare_v002',timeout_seconds=14400,progress_schema='application',
            requires=['MODEL-tools']+(['gemma-access'] if key=='gemma' else []),continue_on_failure=True,
            runner_args=['--action','inputs','--model-key',key,'--private-root',private]))
        for action in ('tune','quality','resident'):
            stage=dict(id=key+'-'+action,label=title+': '+action,operation='phase',kind='measurement',
                campaign=cfg,module='strassen_mm.benchmark_llm_tradeoff_v001',timeout_seconds=10800,
                progress_schema='application',requires=[prep] if action=='tune' else [tune],continue_on_failure=True,
                input_stages=[dict(stage=prep,flag='--model-input',file='prepared/input_bundle.json')],
                runner_args=['--action',action])
            if action!='tune':stage.update(selection_stage=tune,selection_file='policy_bundle.json')
            stages.append(stage)
    release='runtime/release_allocation_v002.py' if args.hardware=='v6e' else 'runtime/release_allocation_v001.py'
    stages.extend([
        dict(id='report',label='Assemble speed and error comparison',operation='command',kind='analysis',timeout_seconds=600,
             continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_llm_tradeoff_v001.py','--cohort','{cohort}',
                '--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release this allocation after result retrieval',operation='command',kind='preparation',
             timeout_seconds=180,releases_allocation=True,command=['{controller_python}','{source}/'+release,
                 '--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        private_root=private,models=args.models,hardware=args.hardware,release=release,
        transport='tools/run_phase_v005.py' if args.hardware=='v6e' else 'tools/run_phase_v004.py',
        title='Large LLM speed–accuracy tradeoffs ('+args.hardware+')',stages=stages,
        decisions=['Exact BF16 weights; FP32 MM accumulation; unchanged numerical and model-quality gates.',
            'M2048 tuning on windows0..3; quality on disjoint windows16..31, 32752 next-token targets.',
            'Separate tune/quality/resident processes and immutable per-stage result archives.',
            'Native default, tuned Native, cubic, Strassen1, Strassen2; full-JIT Native quality/performance control.',
            'Fully resident timing requires memory preflight; record a failure rather than silently stream weights.',
            'All policies receive two warmups and nine randomized paired rounds across three fixed held-out prompts.',
            'Resident prompt forward and actual teacher-forced scoring are separate; no production serving claim.',
            'Private Gemma token export already authorized for this checkpoint task; never put it in source/results.',
            'One allocation at a time, no automatic machine replacement or cross-session pooling.'])


def freeze(args):
    cohort=args.cohort.resolve();cohort.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=build_plan(args);write(cohort/'plan.json',plan)
    write(cohort/'cohort.json',dict(cohort_id=cohort.name,session=args.session,allocation_id=args.endpoint,source_commit=revision,created_utc=utc()))
    source=cohort/'source';source.mkdir()
    with (cohort/'baseline-source.tar').open('xb') as f:
        subprocess.run(['git','archive',revision,'--','src','tools','runtime','configs','status','tests','plans'],cwd=ROOT,stdout=f,check=True)
    with tarfile.open(cohort/'baseline-source.tar') as f:f.extractall(source,filter='data')
    write(source/'plan.json',plan)
    with tarfile.open(cohort/'source.tar','x') as f:
        for child in sorted(source.iterdir()):f.add(child,arcname=child.name)
    write(cohort/'frozen.json',dict(cohort_id=cohort.name,session=args.session,allocation_id=args.endpoint,baseline_commit=revision,
        archive_sha256=sha(cohort/'source.tar'),source_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}))
    write(cohort/'progress.json',dict(schema_version=1,campaign_id=cohort.name,title=plan['title'],cohort_path=str(cohort.relative_to(ROOT)),
        state='preparing',updated_utc=utc(),heartbeat_utc=utc(),detail='Frozen and awaiting launch; zero new model measurements',stages=[],selected_results=[]))
    commit([cohort],'Freeze larger LLM resident speed/accuracy experiment')
    print(json.dumps({'cohort':str(cohort),'stages':len(plan['stages']),'status':'frozen'}))


def launch(args):
    cohort=args.cohort.resolve();verify_frozen(cohort);plan=read(cohort/'plan.json')
    if (cohort/'controller-process.json').exists():raise FileExistsError('Already launched')
    env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(cohort/'source/src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with (cohort/'dashboard.log').open('x') as log:
        dashboard=subprocess.Popen([plan['controller_python'],str(cohort/'source/status/server_v011.py'),'--port',str(args.port),
            '--campaign-state',str(cohort/'progress.json')],env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    with (cohort/'controller.log').open('x') as log:
        child=subprocess.Popen([plan['controller_python'],str(cohort/'source/tools/run_large_real_v002.py'),'run','--cohort',str(cohort)],
            env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write(cohort/'controller-process.json',dict(pid=child.pid,dashboard_pid=dashboard.pid,launched_utc=utc(),dashboard='http://127.0.0.1:'+str(args.port)))
    print(json.dumps(read(cohort/'controller-process.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','launch']);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--hardware',choices=['v5e','v6e'],default='v6e');p.add_argument('--models',nargs='+',choices=['qwen','mistral','gemma','qwen14'],default=['qwen','mistral','gemma','qwen14'])
    for name in ['session','endpoint','identity','controller-python','analysis-python']:p.add_argument('--'+name)
    p.add_argument('--port',type=int,default=8769);a=p.parse_args()
    if not a.cohort.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Project run directory required')
    if a.action=='freeze':
        if not all([a.session,a.endpoint,a.identity,a.controller_python,a.analysis_python]):p.error('freeze requires allocation details')
        freeze(a)
    else:launch(a)
if __name__=='__main__':main()
