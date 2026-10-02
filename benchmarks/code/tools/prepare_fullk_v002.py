"""Twelve serial shape batches with independent output-precision tuning."""
import json
import prepare_llm_tradeoff_v002 as campaign
from strassen_mm.benchmark_fullk_v002 import groups


def build(args):
    if args.hardware!='v6e':raise ValueError('v6e required')
    cfgpath='configs/fullk_v002/campaign.json'
    cfg=json.loads((campaign.ROOT/cfgpath).read_text())
    shapes=json.loads((campaign.ROOT/'configs/fullk_v002/shapes.json').read_text())['shapes']
    def phase(name,start=None):
        stage=name.rsplit('-',1)[-1]
        spec=dict(id=name,label='Full-K precision study: '+name,operation='phase',kind='measurement',
            campaign=cfgpath,module='strassen_mm.benchmark_fullk_v002',timeout_seconds=10800,
            runner_args=[] if start is None else ['--shape-start',str(start),'--shape-count','1'])
        if stage!='confirm':spec['expected']=sum(len(g['arms'])*len(g['inputs']) for g in groups(cfg,shapes,stage,start=start or 0,count=None if start is None else 1))
        return spec
    stages=[phase('fullk-smoke')]
    for i,s in enumerate(shapes):
        screen=phase(f'fullk-{i+1:02d}-screen',i);screen.update(label=s['id']+': FP32/BF16 screening',requires=[stages[-1]['id']])
        confirm=phase(f'fullk-{i+1:02d}-confirm',i);confirm.update(label=s['id']+': fresh FP32/BF16 confirmation',
            requires=[screen['id']],selection_stage=screen['id'],selection_file='selections.json')
        stages.extend([screen,confirm])
    stages.extend([
        dict(id='report',label='Separate precision results and full/short comparisons',operation='command',kind='analysis',
            timeout_seconds=1800,continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_fullk_v002.py',
                '--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release v6e after retrieval',operation='command',kind='preparation',timeout_seconds=180,releases_allocation=True,
            command=['{controller_python}','{source}/runtime/release_allocation_v002.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='12-shape full contraction: separate FP32 and BF16 output',stages=stages,
        decisions=[cfg['candidate_policy'],cfg['selection_policy'],
            'Same BF16 operands, FP32 accumulation/reconstruction and 112 MiB custom allowance. Only final output store differs by precision track.',
            'Exact CPU regression/final-cast checks and compiled sparse-integer smoke; measured full-output finiteness and sampled all-K FP64 errors.',
            'K sweep, swapped tall/wide orientations, small M/N, aligned non-power K, parent and large-cube anchors. No new padding or depths3/4.',
            'Single owned allocation and serial phases; all failures preserved, archives verified, automatic runtime release.',
            'Complete-call timing includes output conversion, excludes transfers/compilation. Fresh-seed confirmation is not unseen-shape selector validation.'])


campaign.build_plan=build
if __name__=='__main__':campaign.main()
