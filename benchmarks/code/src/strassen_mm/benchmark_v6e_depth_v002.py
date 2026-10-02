"""Use validated compact outer loops for recursion depths three and four."""
from . import benchmark_v6e_depth_v001 as benchmark

original=benchmark.make_function


def make_function(arm,shape):
    if arm.get('depth',0)<3:return original(arm,shape)
    from .kernels_recursive_v002 import make_matmul
    return make_matmul(shape,tuple(arm['tile']),arm['depth'],vmem_limit_bytes=arm['vmem_limit_bytes'])


benchmark.make_function=make_function
if __name__=='__main__':raise SystemExit(benchmark.main())
