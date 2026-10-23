"""Run architecture diagnostics and retrieve/release before adaptive ten-shape run."""
import prepare_v6e_opt_v001 as previous

def build(args):
    plan=previous.build(args);plan['title']='v6e architecture and compiler diagnostic'
    stages=[]
    for name,cfg,action,timeout in [('diag-smoke','campaign.json','smoke',1800),('diag-profile','campaign.json','profile',4200),('llo-profile','llo.json','profile',2400)]:
        stage=dict(id=name,label=name,operation='phase',kind='measurement',campaign='configs/v6e_diagnostic_v001/'+cfg,
            module='strassen_mm.benchmark_v6e_diagnostic_v001',timeout_seconds=timeout)
        if stages:stage['requires']=[stages[-1]['id']]
        stages.append(stage)
    plan['stages']=stages+[plan['stages'][-1]]
    plan['decisions']=[
      'Architecture diagnostic precedes ten-shape screening. Separate controlled ablations on two shapes not in the ten-shape set.',
      'Pinned current JAX 0.11.2/jaxlib 0.11.2/libtpu 0.0.48 and xprof-nightly 2.24.2a20260922. Historical stack unchanged in archives.',
      'Compare original and reconstructed kernels with larger leaves, buffer counts 1/2/3, and M/N output traversal.',
      'Exact integer smoke before timings; unchanged BF16/FP32 and error gates.',
      'First profiles use no LLO instrumentation. Separate LLO-instrumented probe never used for speedup claims.',
      'LLO timing breakdowns include compiler cycle estimates; no v6e periodic hardware-counter claim.',
      'One allocation serial phases, immutable artifacts, automatic release after retrieval.']
    return plan
previous.campaign.build_plan=build
if __name__=='__main__':previous.campaign.main()
