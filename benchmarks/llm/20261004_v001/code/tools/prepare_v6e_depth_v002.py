"""Preserve the bounded protocol using compact deeper-recursion kernels."""
import prepare_v6e_depth_v001 as prior

original=prior.build


def build(args):
    plan=original(args)
    for stage in plan['stages']:
        if stage.get('operation')=='phase':stage['module']='strassen_mm.benchmark_v6e_depth_v002'
    plan['decisions'].append('Depths3/4 use scalar loops for outer recursion and expanded inner two levels; same 7^depth products, order and scratch reuse. Fully expanded depth4 CPU compilation was superseded and its interruption retained.')
    return plan


prior.campaign.build_plan=build
if __name__=='__main__':prior.campaign.main()
