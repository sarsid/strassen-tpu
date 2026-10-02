"""Same-shape fresh-allocation comparison of memory-reduced Strassen two."""
import prepare_v6e_opt_v001 as previous


def build(args):
    plan=previous.build(args)
    plan['title']='v6e outer-deferred Strassen 2 follow-up'
    plan['stages']=[s for s in plan['stages'] if s['id']!='opt-profile']
    for stage in plan['stages']:
        if stage['operation']!='phase':continue
        stage['campaign']='configs/v6e_opt_v002/campaign.json'
        stage['module']='strassen_mm.benchmark_v6e_opt_v002'
        stage.pop('expected',None)
        if stage['id']=='opt-screen':stage.update(requires=['opt-smoke'],expected=30)
    # First-round report remains preserved; the follow-up uses an accurate header.
    for stage in plan['stages']:
        if stage['id']=='report':stage['command'][1]='{source}/tools/report_v6e_opt_v002.py'
    plan['decisions']=[
        'Adaptive follow-up motivated by first-round VMEM failures. No depths3/4.',
        'Two long-K down-projection shapes selected from the four v5e winners; exact first-round per-shape original and optimized kernels frozen as controls.',
        'Six hybrid depth-two tiles; five Native settings retuned on this allocation. 30 screen attempts.',
        'Hybrid reconstructs the inner level per panel and defers outer reconstruction until K finishes, retaining 7 intermediate matrices.',
        'Four exact CPU algebra checks passed before TPU smoke. Compiled TPU exact smoke remains mandatory.',
        'Same precision and error gates; 3 new confirmation seeds, 30 paired rounds each. Synthetic MM only.',
        'New allocation and fresh controls; never pool timing samples across allocations. Finalist profiling is separate from normal latency.',
        'One owned allocation, serial execution, immutable evidence, automatic release after retrieval.']
    return plan


previous.campaign.build_plan=build
if __name__=='__main__':previous.campaign.main()
