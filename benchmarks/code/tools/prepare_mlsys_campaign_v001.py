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
    stages=[];private='/content/Strassen_MM_Focus/.runtime_private/'+cohort.name
    def phase(name,label,campaign,timeout=7200,**extra):
        stages.append(dict(id=name,label=label,operation='phase',kind='measurement',campaign=campaign,
            module='strassen_mm.benchmark_mlsys_shapes_v001',timeout_seconds=timeout,**extra))
    def audit(name):
        stages.append(dict(id=name+'-audit',label='Verify '+name,operation='command',kind='analysis',timeout_seconds=1200,
            command=['{analysis_python}','{source}/tools/audit_mlsys_stage_v001.py','--cohort','{cohort}','--stage',name]))
    llm='configs/mlsys_llm_shapes_v001/campaign.json';main='configs/mlsys_shapes_v001/campaign.json'
    real='configs/mlsys_real_models_v001/campaign.json'
    phase('LLM-smoke','Large LLM shapes: qualification',llm,1800)
    audit('LLM-smoke')
    phase('LLM-screen','Large LLM shapes: 52 candidates on each of 18 shapes',llm,10800,expected=936)
    audit('LLM-screen')
    phase('LLM-confirm','Large LLM shapes: fresh confirmation of frozen selections',llm,7200,selection_stage='LLM-screen')
    audit('LLM-confirm')
    stages.append(dict(id='MODEL-tools',label='Prepare isolated CPU model tools',operation='phase',kind='preparation',campaign=real,
        module='strassen_mm.benchmark_mlsys_prepare_v001',timeout_seconds=3600,progress_schema='application',continue_on_failure=True,
        runner_args=['--action','tools','--private-root',private]))
    for model in ('qwen','mistral','gemma'):
        prep=model+'-inputs';actual=model+'-actual'
        stages.append(dict(id=prep,label=model.title()+': acquire real weights and independent reference',operation='phase',kind='preparation',
            campaign=real,module='strassen_mm.benchmark_mlsys_prepare_v001',timeout_seconds=10800,progress_schema='application',
            requires=['MODEL-tools'],continue_on_failure=True,runner_args=['--action','inputs','--model-key',model,'--private-root',private]))
        stages.append(dict(id=actual,label=model.title()+': actual projection tuning and full-model execution',operation='phase',kind='measurement',
            campaign=real,module='strassen_mm.benchmark_mlsys_models_v001',timeout_seconds=14400,progress_schema='application',
            requires=[prep],continue_on_failure=True,input_stages=[dict(stage=prep,flag='--model-input',file='prepared/input_bundle.json')]))
    phase('MAIN-smoke','168-shape suite: qualification',main,1800);audit('MAIN-smoke')
    for chunk in range(12):
        name=f'MAIN-{chunk+1:02d}';args=['--shape-start',str(chunk*14),'--shape-count','14']
        phase(name+'-screen',f'Shapes {chunk*14+1}–{chunk*14+14} of 168: tune',main,10800,runner_args=args,expected=728)
        audit(name+'-screen')
        phase(name+'-confirm',f'Shapes {chunk*14+1}–{chunk*14+14} of 168: confirm',main,7200,runner_args=args,selection_stage=name+'-screen')
        audit(name+'-confirm')
    stages.append(dict(id='release',label='Release the completed v5e allocation',kind='preparation',operation='command',timeout_seconds=180,
        releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v001.py','--session','{session}','--expect-endpoint','{endpoint}']))
    stages.append(dict(id='report',label='Compile all available results',kind='analysis',operation='command',timeout_seconds=1200,
        command=['{analysis_python}','{source}/tools/report_mlsys_campaign_v001.py','--cohort','{cohort}','--output-dir','{cohort}/report']))
    return dict(schema_version=1,cohort_id=cohort.name,session=session,allocation_id=endpoint,remote_identity=identity,
        controller_python=controller,analysis_python=analysis,private_root=private,stages=stages,
        decisions=['One v5e allocation, serial phases; no v6e and no automatic machine replacement.',
        'Large synthetic LLM projections precede actual-model study, then 168 unique generic shapes.',
        'Actual models: original supported Qwen3-0.6B, Mistral7B-v0.3, Gemma3-1B; large synthetic Qwen8B/Gemma12B are not actual checkpoint claims.',
        'All 168 are development shapes; later unseen shapes required for selector validation.',
        'Equal sixteen-candidate search per custom family plus four Native compiler settings; default Native is a separate headline.',
        'Shortlist top three plus within 5 percent, cap four per family; fresh confirmation never reselects headline winner.',
        'Three fresh seeds, 30 paired rounds, pointwise95percent intervals conditional on frozen selection.',
        'Exact BF16 inputs and FP32 output; FP64 host numerical reference on full or sampled outputs with all K.',
        'Main screen+confirmation in12chunks of14 shapes, source and allocation fixed; each execution archived and committed.',
        'Known-idle model preparation or qualification problems remain explicit and do not prevent generic shapes; uncertain remote work stops advancement.',
        'Native independent model qualification failure blocks full-model claims; streamed model timings include weight movement and are labeled accordingly.',
        'Controller prevents idle sleep, requires powered online open-lid laptop; current remote phase detached and checkpointed.',
        'No learned selector, two-level retuning after confirmation, AlphaTensor, or later generalization experiment in this campaign.'])

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
    write(cohort/'progress.json',dict(schema_version=1,campaign_id=cohort.name,title='MLSys main v5e campaign',cohort_path=str(cohort.relative_to(ROOT)),
        state='preparing',updated_utc=utc(),heartbeat_utc=utc(),detail='Frozen and awaiting detached launch',stages=[],selected_results=[]))
    from run_mlsys_unattended_v001 import commit
    commit([cohort],'Freeze main MLSys v5e campaign source and complete execution plan')
    print(json.dumps({'cohort':str(cohort),'stage_count':len(plan['stages']),'status':'frozen'}))

def launch(args):
    cohort=args.cohort.resolve();verify_frozen(cohort);plan=read(cohort/'plan.json')
    if (cohort/'controller-process.json').exists():raise FileExistsError('Already launched')
    env=dict(os.environ,STRASSEN_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(cohort/'source/src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with (cohort/'dashboard.log').open('x') as log:
        dashboard=subprocess.Popen([plan['controller_python'],str(cohort/'source/status/server_v011.py'),'--port','8767',
            '--campaign-state',str(cohort/'progress.json')],env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    with (cohort/'controller.log').open('x') as log:
        child=subprocess.Popen([plan['controller_python'],str(cohort/'source/tools/run_mlsys_unattended_v001.py'),'run','--cohort',str(cohort)],
            env=env,cwd=cohort/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write(cohort/'controller-process.json',dict(pid=child.pid,dashboard_pid=dashboard.pid,launched_utc=utc(),dashboard='http://127.0.0.1:8767'))
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
