"""Freeze the bounded real-checkpoint diagnostic; reuse the serial controller."""
import prepare_llm_tradeoff_v002 as campaign

original=campaign.build_plan
def build(args):
    args.models=['gemma'];plan=original(args)
    stages=[s for s in plan['stages'] if s['id'] in ('MODEL-tools','gemma-access','gemma-inputs')]
    stages.append(dict(id='gemma-diagnose',label='Gemma Native official intermediate comparisons',operation='phase',kind='measurement',
        campaign='configs/llm_tradeoff_v6e_v001/campaign.json',module='strassen_mm.benchmark_gemma_diagnostic_v001',
        timeout_seconds=7200,progress_schema='application',requires=['gemma-inputs'],continue_on_failure=True,
        input_stages=[dict(stage='gemma-inputs',flag='--model-input',file='prepared/input_bundle.json')],
        runner_args=['--private-root',plan['private_root']]))
    stages.append(next(s for s in plan['stages'] if s['id']=='release'))
    plan.update(stages=stages,title='Gemma 3-12B Native diagnosis (v6e)',
        decisions=['Exact checkpoint and pinned official reference; unchanged qualification gates.',
            'Compare official common inputs and propagated states for composed, full-JIT and eager Native.',
            'Capture intermediate operations in first local, first global and last decoder layer.',
            'No Strassen performance inference; archive diagnostics and release owned runtime.'])
    return plan
campaign.build_plan=build
if __name__=='__main__':campaign.main()
