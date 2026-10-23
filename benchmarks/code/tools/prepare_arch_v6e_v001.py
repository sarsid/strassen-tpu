"""Freeze the full architecture-specific v6e study and its live controller."""
import json
import prepare_llm_tradeoff_v002 as campaign
from strassen_mm.tuner_arch_study_v001 import groups

def build(args):
    if args.hardware!='v6e':raise ValueError('Never reuse this v6e search on v5e')
    cfgpath='configs/arch_v6e_168_v001/campaign.json'
    cfg=json.loads((campaign.ROOT/cfgpath).read_text())
    shapes=json.loads((campaign.ROOT/'configs/arch_v6e_168_v001/shapes.json').read_text())['shapes']
    def phase(name,index=None):
        stage=name.rsplit('-',1)[-1]
        s=dict(id=name,label='v6e '+name,operation='phase',kind='measurement',campaign=cfgpath,
            module='strassen_mm.benchmark_arch_study_v001',timeout_seconds=14400,
            runner_args=[] if index is None else ['--shape-start',str(index),'--shape-count','1'])
        if stage!='confirm':s['expected']=sum(len(g['arms'])*len(g['inputs']) for g in groups(cfg,shapes,stage,start=index or 0,count=1 if index is not None else None))
        return s
    def export(name):
        return dict(id='export-'+name,label='Verify and commit dedicated results: '+name,operation='command',kind='analysis',requires=[name],timeout_seconds=1800,
            command=['{analysis_python}','{source}/tools/export_arch_study_v001.py','--cohort','{cohort}','--phase',name])
    stages=[phase('arch-smoke'),export('arch-smoke')]
    # Preflight a tiny, boundary and large shape before the remaining full grid.
    # Original manifest indices/IDs/seeds remain fixed despite this execution order.
    order=cfg['execution_shape_order']
    for index in order:
        screen=phase(f'arch-{index+1:03d}-screen',index);screen['requires']=[stages[-1]['id']]
        confirm=phase(f'arch-{index+1:03d}-confirm',index)
        confirm.update(requires=[screen['id']],selection_stage=screen['id'],selection_file='selections.json')
        stages.extend([screen,export(screen['id']),confirm,export(confirm['id'])])
    stages.extend([
        dict(id='report',label='Summarize all shapes, precision contracts, errors and tuning choices',operation='command',kind='analysis',timeout_seconds=3600,
            command=['{analysis_python}','{source}/tools/report_arch_study_v001.py','--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='export-final',label='Commit complete dedicated v6e results',operation='command',kind='analysis',requires=['report'],timeout_seconds=1800,
            command=['{analysis_python}','{source}/tools/export_arch_study_v001.py','--cohort','{cohort}','--final']),
        dict(id='release',label='Release v6e and verify absence',operation='command',kind='preparation',timeout_seconds=180,releases_allocation=True,
            command=['{controller_python}','{source}/runtime/release_allocation_v002.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='v6e: 168 shapes, FP32 and BF16, architecture-specific joint tuning',
        results_relative='results/v6e/168_shapes_joint_fp32_bf16_20260927_v001',stages=stages,
        decisions=[cfg['candidate_policy'],cfg['selection_policy'],
            'Only v6e joint kernels and v6e compiler/memory settings. Historical v5e code is a separate later study.',
            'Smoke must pass exact arithmetic for aligned, tiny and boundary cases in both output contracts.',
            'Native, independent cubic, S1 and S2; default Native retained. Three fresh confirmation seeds, no reselection.',
            'Archive and commit each phase into the dedicated results directory. Live progress counts are actual terminal measurements.',
            'Single allocation, serial timing. An uncertain launch stops advancement; no silent retries or cross-allocation timing pooling.'])

campaign.build_plan=build
if __name__=='__main__':campaign.main()
