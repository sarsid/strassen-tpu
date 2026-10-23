"""Freeze the reviewed serial main-v5e campaign and launch detached local services."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from run_region_cohort_v001 import write,read,sha,utc,verify_frozen

ROOT=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()

def build_plan(cohort,session,endpoint,identity,controller,analysis):
    private='/content/Strassen_MM_Focus/.runtime_private/'+cohort.name
    real='configs/large_real_models_v001/campaign.json'
    stages=[dict(id='MODEL-tools',label='Install isolated CPU reference tools',operation='phase',kind='preparation',
        campaign=real,module='strassen_mm.benchmark_large_prepare_v001',timeout_seconds=3600,progress_schema='application',
        runner_args=['--action','tools','--private-root',private])]
    for model,title in [('qwen','Qwen3-8B'),('mistral','Mistral-7B-v0.3'),('gemma','Gemma3-12B text')]:
        if model=='gemma':
            stages.append(dict(id='gemma-access',label='Wait for authorized Gemma checkpoint access',operation='command',
                kind='preparation',timeout_seconds=14400,continue_on_failure=True,
                command=['{controller_python}','{source}/tools/wait_large_gemma_access_v001.py','--cohort','{cohort}']))
        prep=model+'-inputs'
        stages.append(dict(id=prep,label=title+': download actual checkpoint and official reference',operation='phase',kind='preparation',
            campaign=real,module='strassen_mm.benchmark_large_prepare_v001',timeout_seconds=14400,progress_schema='application',
            requires=['MODEL-tools']+(['gemma-access'] if model=='gemma' else []),continue_on_failure=True,
            runner_args=['--action','inputs','--model-key',model,'--private-root',private]))
        stages.append(dict(id=model+'-actual',label=title+': actual large projections and full-model evaluation',operation='phase',kind='measurement',
            campaign=real,module='strassen_mm.benchmark_large_models_v001',timeout_seconds=21600,progress_schema='application',
            requires=[prep],continue_on_failure=True,input_stages=[dict(stage=prep,flag='--model-input',file='prepared/input_bundle.json')]))
    stages.append(dict(id='report',label='Summarize actual model results and quality gates',operation='command',kind='analysis',timeout_seconds=1200,
        continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_large_real_v001.py','--cohort','{cohort}',
          '--output-dir','{cohort}/operations/report/artifacts']))
    stages.append(dict(id='release',label='Release this campaign TPU after evidence retrieval',operation='command',kind='preparation',
        timeout_seconds=180,releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v001.py',
        '--session','{session}','--expect-endpoint','{endpoint}']))
    return dict(schema_version=1,cohort_id=cohort.name,session=session,allocation_id=endpoint,remote_identity=identity,
        controller_python=controller,analysis_python=analysis,private_root=private,stages=stages,
        decisions=['Exact requested Qwen3-8B, Mistral7B-v0.3, Gemma3-12B checkpoint weights; no smaller substitutes.',
            'Serial single-v5e execution. Three actual activation projection sizes2048/8192/16384; full-model context2048.',
            'Native default, tuned Native, cubic, Strassen1 and Strassen2. Fixed numerical and quality gates.',
            'Gemma text backbone only; official global linear RoPE factor8, local default RoPE.',
            'Independent official CPU64-token qualification before any full-model conclusions.',
            'Full-model quality16x2048tokens, resident layers7rounds, streamed model3repeats; scopes stay separate.',
            'Gemma credential staging requires explicit approval; gate waits for private-stage receipt.',
            'Archive every stage and release after retrieval. No automatic allocation replacement.'])


def freeze(args):
    cohort=args.cohort.resolve();cohort.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=build_plan(cohort,args.session,args.endpoint,args.identity,args.controller_python,args.analysis_python)
    write(cohort/'plan.json',plan)
    write(cohort/'cohort.json',dict(cohort_id=cohort.name,session=args.session,allocation_id=args.endpoint,source_commit=head,created_utc=utc()))
    source=cohort/'source';source.mkdir()
    with (cohort/'baseline-source.tar').open('xb') as out:
        subprocess.run(['git','archive',head,'--','src','tools','runtime','configs','status','tests','plans'],cwd=ROOT,stdout=out,check=True)
    with tarfile.open(cohort/'baseline-source.tar') as packed:packed.extractall(source,filter='data')
    write(source/'plan.json',plan)
    with tarfile.open(cohort/'source.tar','x') as packed:
        for child in sorted(source.iterdir()):packed.add(child,arcname=child.name)
    write(cohort/'frozen.json',dict(cohort_id=cohort.name,session=args.session,allocation_id=args.endpoint,baseline_commit=head,
        archive_sha256=sha(cohort/'source.tar'),source_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}))
    write(cohort/'progress.json',dict(schema_version=1,campaign_id=cohort.name,title='Actual Qwen3-8B, Mistral7B and Gemma3-12B',cohort_path=str(cohort.relative_to(ROOT)),
        state='preparing',updated_utc=utc(),heartbeat_utc=utc(),detail='Frozen and awaiting detached launch',stages=[],selected_results=[]))
    from run_large_real_v001 import commit
    commit([cohort],'Freeze main MLSys v5e campaign source and complete execution plan')
    print(json.dumps({'cohort':str(cohort),'stage_count':len(plan['stages']),'status':'frozen'}))

def launch(args):
    cohort=args.cohort.resolve();verify_frozen(cohort);plan=read(cohort/'plan.json')
    if (cohort/'controller-process.json').exists():raise FileExistsError('Already launched')
    env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(cohort/'source/src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with (cohort/'dashboard.log').open('x') as log:
        dashboard=subprocess.Popen([plan['controller_python'],str(cohort/'source/status/server_v011.py'),'--port','8768',
            '--campaign-state',str(cohort/'progress.json')],env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    with (cohort/'controller.log').open('x') as log:
        child=subprocess.Popen([plan['controller_python'],str(cohort/'source/tools/run_large_real_v001.py'),'run','--cohort',str(cohort)],
            env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write(cohort/'controller-process.json',dict(pid=child.pid,dashboard_pid=dashboard.pid,launched_utc=utc(),dashboard='http://127.0.0.1:8768'))
    print(json.dumps(read(cohort/'controller-process.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','launch']);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--session');p.add_argument('--endpoint');p.add_argument('--identity');p.add_argument('--controller-python');p.add_argument('--analysis-python')
    args=p.parse_args()
    if not args.cohort.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Project cohort required')
    if args.action=='freeze':
        if not all([args.session,args.endpoint,args.identity,args.controller_python,args.analysis_python]):p.error('freeze requires allocation and interpreters')
        freeze(args)
    else:launch(args)

if __name__=='__main__':main()
