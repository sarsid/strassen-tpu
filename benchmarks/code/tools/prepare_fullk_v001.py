"""Freeze serial full-contraction curves and separate parent BF16 reproduction."""
import json
import prepare_llm_tradeoff_v002 as campaign


def build(args):
    if args.hardware!='v6e':raise ValueError('v6e required')
    from strassen_mm.benchmark_fullk_v001 import arms_for,groups
    cfg='configs/fullk_v001/campaign.json'
    config=json.loads((campaign.ROOT/cfg).read_text())
    shapes=json.loads((campaign.ROOT/'configs/fullk_v001/shapes.json').read_text())['shapes']
    stages=[dict(id='fullk-smoke',label='Exact compiled full-K and multi-panel qualification',operation='phase',kind='measurement',
        campaign=cfg,module='strassen_mm.benchmark_fullk_v001',timeout_seconds=3600,
        expected=sum(len(g['arms']) for g in groups(config,shapes,True)))]
    for i,s in enumerate(shapes):
        stages.append(dict(id=f'fullk-{i+1:02d}-measure',label=s['id'],operation='phase',kind='measurement',
            campaign=cfg,module='strassen_mm.benchmark_fullk_v001',timeout_seconds=10800,
            expected=3*len(arms_for(s)),requires=[stages[-1]['id']],runner_args=['--shape-start',str(i),'--shape-count','1']))
    stages.extend([
        dict(id='report',label='Compare contraction curves, current controls, Native and errors',operation='command',kind='analysis',
             timeout_seconds=900,continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_fullk_v001.py',
                '--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release the owned v6e after artifact retrieval',operation='command',kind='preparation',
             timeout_seconds=180,releases_allocation=True,command=['{controller_python}','{source}/runtime/release_allocation_v002.py',
                '--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='Full contraction on v6e: current S1/S2 and pinned parent kernels',stages=stages,
        decisions=['Four aligned shapes: parent gate/up geometry, 8192 cube, 16384 cube, and 16384×8192×16384. Parent geometry also receives separate BF16-output reproduction.',
          'Two fixed output-tile geometries per shape, three divisor BK values including full K. No padding variants, no depths 3/4, no LLM fusion.',
          'Current deferred S1, hybrid S2, unmodified pinned parent S1, and matched blocked cubic; current historical S1/S2 choices replayed as controls in FP32.',
          'All fixed candidates measured on three fresh Gaussian inputs × 30 paired rotated rounds; no fastest-candidate inference without selection caveat.',
          'Native default and per-executable 32/48/64/96/112 MiB settings. No global scoped-VMEM override. Custom budget 112 MiB in FP32, upstream 104 MiB in BF16.',
          'Exact CPU and compiled integer qualification; sampled all-K FP64 references and full-output finiteness. BF16 rounding remains part of error, not subtracted.',
          'Record genuine compiler/VMEM failures without silently changing tiles. Complete-call timing excludes compile/transfer; all HLO and compiler memory analyses retained.',
          'Pinned parent source 95be1fb is byte-identical; newer JAX/libtpu and Gaussian inputs make this a kernel/geometry reproduction, not a bit-for-bit rerun of upstream historical evidence.',
          'One owned allocation, serial phases, immutable archive retrieval and automatic release.'])


campaign.build_plan=build
if __name__=='__main__':campaign.main()
