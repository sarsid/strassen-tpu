"""Freeze a 13-shape boundary probe with the existing remote archive lifecycle."""
import prepare_llm_tradeoff_v002 as campaign


def build(args):
    if args.hardware!='v6e':raise ValueError('v6e required')
    cfg='configs/padding_v001/campaign.json'
    stages=[dict(id='padding-smoke',label='Compiled boundary algebra qualification',operation='phase',kind='measurement',
        campaign=cfg,module='strassen_mm.benchmark_padding_v001',timeout_seconds=3600,expected=28)]
    for i,(start,count) in enumerate(((0,4),(4,4),(8,5)),1):
        stage=dict(id=f'padding-{i:02d}-measure',label=f'Boundary batch {i}',operation='phase',kind='measurement',
            campaign=cfg,module='strassen_mm.benchmark_padding_v001',timeout_seconds=7200,expected=count*21,
            requires=[stages[-1]['id']],runner_args=['--shape-start',str(start),'--shape-count',str(count)])
        stages.append(stage)
    stages.extend([
        dict(id='report',label='Compare complete-call times and errors against existing S1/S2',operation='command',kind='analysis',
             timeout_seconds=600,continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_padding_v001.py',
                '--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release the owned v6e after artifact retrieval',operation='command',kind='preparation',
             timeout_seconds=180,releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v002.py',
                '--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='Boundary handling: S1/S2 versus padded edges and Native fringes',stages=stages,
        decisions=['13 frozen shapes: 10 boundary cases and 3 regular controls; only depths 1 and 2.',
          'Controls reuse historical per-shape selected pipeline tiles where present, otherwise the 8192-cube choice. New policies use exactly the same tile and interior implementation.',
          'No new tuning or winner selection. Every policy reported, with three fresh Gaussian seeds and 30 rotated paired rounds.',
          'Current S1/S2 are primary controls. Default Native is context only; no expanded padding-control campaign.',
          'S1_padded/S2_padded use smaller Strassen edge tiles; separately labelled Native-fringe variants use a Strassen core and conventional exact remainder terms.',
          'All slicing, padding, partial-sum additions, concatenation and cropping included in complete-call timings. Host transfer and compilation excluded.',
          'FP64 sampled reference uses all K and includes matrix boundaries; full-output finiteness. Exact full-output CPU tests and compiled integer TPU qualification precede timing.',
          'BF16 operands/preadds, FP32 accumulation/output; same historical numerical gates. No claim of best possible tuning or LLM accuracy.',
          'Full-contraction tuning and fused boundary loads are separate follow-ups, not silently mixed into the padding comparison.',
          'One owned allocation, serial phases, immutable archives, then release.'])


campaign.build_plan=build
if __name__=='__main__':campaign.main()
