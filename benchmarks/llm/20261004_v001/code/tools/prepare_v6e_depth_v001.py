"""Freeze and launch one bounded v6e recursion-depth experiment."""
import prepare_llm_tradeoff_v002 as campaign


def build(args):
    if args.hardware!='v6e':raise ValueError('v6e required')
    cfg='configs/v6e_depth_v001/campaign.json'
    stages=[]
    for action,timeout in [('smoke',1800),('screen',7200),('confirm',5400)]:
        stage=dict(id='depth-'+action,label='v6e recursion '+action,operation='phase',kind='measurement',
                   campaign=cfg,module='strassen_mm.benchmark_v6e_depth_v001',timeout_seconds=timeout)
        if action=='screen':stage['requires']=['depth-smoke']
        if action=='confirm':stage.update(requires=['depth-screen'],selection_stage='depth-screen',selection_file='selections.json')
        stages.append(stage)
    stages.extend([
        dict(id='report',label='Compare Native and four Strassen depths',operation='command',kind='analysis',
             timeout_seconds=600,continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_v6e_depth_v001.py',
             '--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release v6e after retrieving results',operation='command',kind='preparation',timeout_seconds=180,
             releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v002.py',
             '--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='v6e Native versus Strassen depths 1–4',stages=stages,
        decisions=['Recursion is within each BM/BN/BK tile, not over the entire matrix.',
            'Three shapes: 8192 square, 16384 square, and M8192 K8192 N28672.',
            'Three tiles independently screened per depth/shape; four Native settings including default. 48 screening attempts.',
            'Custom kernels use a frozen 112 MiB VMEM allowance. Depths1/2 preserve existing implementations; depths3/4 extend depth-first scratch reuse.',
            'BF16 inputs and preadds; FP32 leaf dot accumulation and reconstruction. 7^depth leaf dots per tile panel.',
            'CPU tests establish exact algebra; an aligned exact-integer TPU smoke is required before measurements.',
            'Reproducible synthetic Gaussian dense operands; no real-weight LLM claim.',
            'Same screening input per shape; frozen winners confirmed on three fresh seeds, 30 paired timing rounds each.',
            'Unchanged 2% relative L2 and max-error gate; numerical failures remain timed and visible but cannot count as valid speedups.',
            'Complete-call timings include device padding and crop, exclude host transfer and compilation. Report compilation separately.',
            'One bounded owned allocation, serial measurements, archive all failures and release automatically.'])


campaign.build_plan=build
if __name__=='__main__':campaign.main()
