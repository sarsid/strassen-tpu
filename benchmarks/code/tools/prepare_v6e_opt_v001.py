"""Freeze one bounded v6e schedule optimization on documented v5e winners."""
import prepare_llm_tradeoff_v002 as campaign


def build(args):
    if args.hardware!='v6e':raise ValueError('One v6e required')
    cfg='configs/v6e_opt_v001/campaign.json';stages=[]
    for action,timeout,expected in [('smoke',1800,10),('profile',1800,10),('screen',9000,136),('confirm',5400,84)]:
        stage=dict(id='opt-'+action,label='v6e optimization '+action,operation='phase',kind='measurement',
                   campaign=cfg,module='strassen_mm.benchmark_v6e_opt_v001',timeout_seconds=timeout,expected=expected)
        if stages:stage['requires']=[stages[-1]['id']]
        if action=='confirm':stage.update(selection_stage='opt-screen',selection_file='selections.json')
        stages.append(stage)
    stages.extend([
        dict(id='report',label='Compare optimized kernels with original kernels and both Native controls',
             operation='command',kind='analysis',timeout_seconds=600,continue_on_failure=True,
             command=['{analysis_python}','{source}/tools/report_v6e_opt_v001.py','--cohort','{cohort}',
                      '--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release v6e after artifact retrieval',operation='command',kind='preparation',
             timeout_seconds=180,releases_allocation=True,command=['{controller_python}',
             '{source}/runtime/release_allocation_v002.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='v6e kernel optimization on v5e winning shapes',stages=stages,
        decisions=['Only Native, classical cubic control and Strassen depths1/2; no depths3/4.',
            'Four shapes preselected from archived eligible v5e wins over both default and tuned Native; evidence frozen with configuration.',
            'New modes defer FP32 reconstruction until the last K panel, or reconstruct in SSA each panel. BF16 operand rounding unchanged; FP32 addition order changes.',
            'Five tiles per original depth; eight schedule/tile combinations per optimized depth; five Native controls and three cubic controls. 136 screen attempts.',
            'Custom VMEM allowance 112 MiB. Original kernels independently retuned on the same allocation.',
            'CPU exact algebra and compiled exact integer TPU smoke precede measurements.',
            'Baseline profiles precede screening; selected profiles follow confirmation measurements. Profiled calls never enter reported latency samples.',
            'Frozen winners receive 30 paired timing rounds on each of three fresh Gaussian seeds. No confirmation reselection.',
            'Same BF16 inputs/preadds, FP32 output, unchanged 2% relative-L2 and max-error gate; host all-K FP64 reference on a 128x128 output sample, full-output finiteness.',
            'Complete device-call timings include padding/cropping, exclude transfer and compilation. Synthetic operands match historical shape study, no new LLM claim.',
            'One owned allocation, serial execution, preserve failures, retrieve evidence and automatically release.'])


campaign.build_plan=build
if __name__=='__main__':campaign.main()
