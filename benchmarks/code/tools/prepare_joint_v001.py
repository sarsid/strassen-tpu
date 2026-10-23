"""Freeze and launch the six-shape explainable joint MM tuner."""
import json
import prepare_llm_tradeoff_v002 as campaign
from strassen_mm.tuner_joint_v001 import groups


def build(args):
    if args.hardware != 'v6e':
        raise ValueError('v6e required')
    cfgpath = 'configs/joint_v001/campaign.json'
    cfg = json.loads((campaign.ROOT/cfgpath).read_text())
    shapes = json.loads((campaign.ROOT/'configs/joint_v001/shapes.json').read_text())['shapes']
    def phase(name, index=None):
        stage = name.rsplit('-',1)[-1]
        spec = dict(id=name, label='Joint MM tuning: '+name, operation='phase', kind='measurement',
            campaign=cfgpath, module='strassen_mm.benchmark_joint_v001', timeout_seconds=14400,
            runner_args=[] if index is None else ['--shape-start',str(index),'--shape-count','1'])
        if stage != 'confirm':
            spec['expected'] = sum(len(g['arms'])*len(g['inputs']) for g in groups(cfg,shapes,stage,start=index or 0,count=None if index is None else 1))
        return spec
    stages = [phase('joint-smoke')]
    for i, s in enumerate(shapes):
        screen = phase(f'joint-{i+1:02d}-screen',i)
        screen.update(label=s['id']+': joint search, separate outputs',requires=[stages[-1]['id']])
        confirm = phase(f'joint-{i+1:02d}-confirm',i)
        confirm.update(label=s['id']+': frozen choices and counterfactuals',requires=[screen['id']],
                       selection_stage=screen['id'],selection_file='selections.json')
        stages.extend([screen,confirm])
    stages.extend([
        dict(id='report',label='Results, per-choice explanations and tuner design',operation='command',kind='analysis',
            timeout_seconds=1800,continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_joint_v001.py',
                '--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release v6e after retrieval',operation='command',kind='preparation',timeout_seconds=180,
            releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v002.py',
                '--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='Explainable joint MM tuning: accumulator, tiles, K panels and buffers',stages=stages,
        decisions=[cfg['candidate_policy'],cfg['selection_policy'],
            'Six contrasting development geometries and two separate output contracts; no LLM fusion, padding or depths3/4.',
            'Every previous selected or replayed control is preserved and rerun on this allocation.',
            'Small randomized compilation batches include Native_default to normalize screening drift. Confirmation compares frozen finalists together.',
            'The report emits per-candidate exclusion/rank evidence, matched accumulator/buffer comparisons, a shape-specific research policy and design choices.',
            'Exact local tests and compiled integer smoke precede performance; no arbitrary-input numerical guarantee.',
            'Single allocation and serial timing, verified archives and automatic runtime release.'])


campaign.build_plan = build
if __name__ == '__main__':
    campaign.main()
