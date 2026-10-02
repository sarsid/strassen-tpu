"""Twelve archived batches of the full 168-shape v6e study."""
import json
import prepare_llm_tradeoff_v002 as campaign
from strassen_mm.benchmark_v6e_suite_v001 import plan_groups

def build(args):
    if args.hardware!='v6e':raise ValueError('One v6e required')
    cfgpath='configs/v6e_suite_v001/campaign.json'
    cfg=json.loads((campaign.ROOT/cfgpath).read_text())
    shapes=json.loads((campaign.ROOT/'configs/v6e_suite_v001/shapes.json').read_text())['shapes']
    def phase(id,start=None,count=None):
        stage=id.rsplit('-',1)[-1]
        spec=dict(id=id,label='168-shape v6e '+id,operation='phase',kind='measurement',campaign=cfgpath,
            module='strassen_mm.benchmark_v6e_suite_v001',timeout_seconds=10800 if stage=='screen' else 7200,
            runner_args=[] if start is None else ['--shape-start',str(start),'--shape-count',str(count)])
        if stage!='confirm':spec['expected']=sum(len(g['arms'])*len(g['inputs'])for g in plan_groups(cfg,shapes,stage,start=start or 0,count=count))
        return spec
    stages=[phase('suite-smoke')]
    for i in range(12):
        prefix=f'suite-{i+1:02d}'
        screen=phase(prefix+'-screen',i*14,14);screen['requires']=[stages[-1]['id']]
        confirm=phase(prefix+'-confirm',i*14,14)
        confirm.update(requires=[screen['id']],selection_stage=screen['id'],selection_file='selections.json')
        stages.extend([screen,confirm])
    stages.extend([
        dict(id='device-analysis',label='Analyze separate device-module profiles',operation='command',kind='analysis',timeout_seconds=1800,continue_on_failure=True,
             command=['{analysis_python}','{source}/tools/analyze_v6e_opt_profiles_v002.py','--cohort','{cohort}','--output-dir','{cohort}/operations/device-analysis/artifacts']),
        dict(id='report',label='Compare all 168 shapes, errors, padding and both timing scopes',operation='command',kind='analysis',timeout_seconds=1800,continue_on_failure=True,
             command=['{analysis_python}','{source}/tools/report_v6e_suite_v001.py','--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release v6e after archiving',operation='command',kind='preparation',timeout_seconds=180,releases_allocation=True,
             command=['{controller_python}','{source}/runtime/release_allocation_v002.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='All 168 development shapes on v6e: Native, Strassen 1 and 2',stages=stages,
        decisions=[
            'All 168 shapes retained in original order, no timing-based shape filtering. Only depths1/2.',
            cfg['candidate_policy'],cfg['selection_policy'],
            'Qualification: five representative exact-integer shapes, seven implementations each; any error blocks screening.',
            'Five Native settings, revised S1 panel/deferred and S2 hybrid, original implementations on two offered tiles; cubic covers every offered tile.',
            'Three fresh confirmation inputs, 30 paired rotated timing rounds each. Same BF16 operands/preadds and FP32 output. No confirmation reselection.',
            'Device profiles after first confirmation input; never pool them with uninstrumented complete-call timings.',
            'Full/all-K sampled FP64 reference, full-output finiteness, unchanged numerical gates; all failures retained.',
            '12 archived screen/confirmation batches; no simultaneous timing, no allocation mixing; retrieve and release.',
            'Development shapes, not unseen selector validation or full-model prediction accuracy.'])

campaign.build_plan=build
if __name__=='__main__':campaign.main()
