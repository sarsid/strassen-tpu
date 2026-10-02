"""Fixed replay of prior pilot tiles against frozen broad-study selections."""
from . import benchmark_fullk_v001 as runner
from .benchmark_fullk_v002 import make_function


def groups(cfg,shapes,smoke=False):
    if smoke:raise ValueError('Use the qualified fullk_v002 smoke module')
    return [dict(group_id=s['id']+'__'+dtype,shape={**s,'id':s['id']+'__'+dtype,'output_dtype':dtype},
        tile=None,arms=arms,scopes=['call'],timing='measure',
        inputs=[dict(distribution='gaussian',seed=seed+s['seed_offset']) for seed in cfg['confirm_seeds']])
        for s in shapes for dtype,arms in s['arms_by_dtype'].items()]


runner.groups=groups
runner.make_function=make_function
if __name__=='__main__':raise SystemExit(runner.main())
