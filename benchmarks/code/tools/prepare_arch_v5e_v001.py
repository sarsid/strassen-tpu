"""Historical v5e workflow, with independent FP32/BF16 output studies."""
import json
import prepare_llm_tradeoff_v002 as campaign
from strassen_mm.benchmark_mlsys_shapes_v001 import plan_groups

def build(args):
    if args.hardware!='v5e':raise ValueError('Historical v5e only')
    stages=[]
    for suffix in ('fp32','bf16'):
        cfgpath='configs/arch_v5e_168_'+suffix+'_v001/campaign.json'
        cfg=json.loads((campaign.ROOT/cfgpath).read_text());shapes=json.loads((campaign.ROOT/cfgpath).with_name('shapes.json').read_text())['shapes']
        def add(name,start=None,count=None):
            stage=name.rsplit('-',1)[-1]
            spec=dict(id=name,label='Historical v5e '+suffix+' '+name,operation='phase',kind='measurement',campaign=cfgpath,
                module='strassen_mm.benchmark_v5e_output_v001',timeout_seconds=14400,
                runner_args=[] if start is None else ['--shape-start',str(start),'--shape-count',str(count)])
            if stages:spec['requires']=[stages[-1]['id']]
            if stage=='confirm':spec.update(selection_stage=name.removesuffix('confirm')+'screen',selection_file='selections.json')
            else:spec['expected']=sum(len(g['arms'])*len(g['inputs']) for g in plan_groups(cfg,shapes,stage,start=start or 0,count=count))
            stages.append(spec)
            stages.append(dict(id='export-'+name,label='Verify and commit '+name,operation='command',kind='analysis',requires=[name],timeout_seconds=1800,
                command=['{analysis_python}','{source}/tools/export_arch_study_v001.py','--cohort','{cohort}','--phase',name]))
        add(suffix+'-smoke')
        for i in range(12):
            add(f'{suffix}-{i+1:02d}-screen',i*14,14);add(f'{suffix}-{i+1:02d}-confirm',i*14,14)
    stages.extend([
        dict(id='report',label='Both v5e output contracts: historical analysis',operation='command',kind='analysis',timeout_seconds=3600,
            command=['{analysis_python}','{source}/tools/report_arch_v5e_v001.py','--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='export-final',label='Commit complete v5e results',operation='command',kind='analysis',requires=['report'],timeout_seconds=1800,
            command=['{analysis_python}','{source}/tools/export_arch_study_v001.py','--cohort','{cohort}','--final']),
        dict(id='release',label='Release v5e and verify absence',operation='command',kind='preparation',timeout_seconds=180,releases_allocation=True,
            command=['{controller_python}','{source}/runtime/release_allocation_v001.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,remote_identity=args.identity,
        controller_python=args.controller_python,analysis_python=args.analysis_python,hardware='v5e',release='runtime/release_allocation_v001.py',
        transport='tools/run_phase_v004.py',title='v5e: historical 168-shape implementation, FP32 and BF16 output',
        results_relative='results/v5e/168_shapes_legacy_fp32_bf16_20260927_v001',stages=stages,
        decisions=['Exact historical kernel files, all 16 tiles per custom family and four Native settings retained.',
            'Original v5e software pins and 48 MiB custom VMEM limit; never import the v6e search.',
            'FP32 factory unchanged. BF16 adds exactly one final output conversion, included in time and error.',
            'Original seeds, ordering, timing, shortlist and fresh confirmation retained. Separate output comparisons.',
            'Each completed phase is verified, exported and committed; the live dashboard records actual measurements.'])

campaign.build_plan=build
if __name__=='__main__':campaign.main()
