"""Second bounded round: first-round finalists plus outer-deferred depth two."""
from . import benchmark_v6e_opt_v001 as benchmark

original=benchmark.make_function


def make_function(arm,shape):
    if arm.get('implementation')!='hybrid':return original(arm,shape)
    from .kernels_v6e_v002 import make_matmul
    return make_matmul(shape,tuple(arm['tile']),order=arm['order'],vmem_limit_bytes=arm['vmem_limit_bytes'])


original_groups=benchmark.groups


def groups(cfg,shapes,stage,selection):
    # Frozen per-shape incumbents must never be used on a different shape.
    result=[]
    for shape in shapes:
        local=dict(cfg)
        local['candidate_families']={k:[c for c in v if c.get('only_shape',shape['id'])==shape['id']]
                                     for k,v in cfg['candidate_families'].items()}
        if stage=='smoke':
            # One representative of each implementation/mode for exact TPU algebra.
            local=cfg
        result.extend(original_groups(local,[shape],stage,selection))
    return result


benchmark.make_function=make_function
benchmark.groups=groups
if __name__=='__main__':raise SystemExit(benchmark.main())
