"""Controlled v6e pipeline buffering and output traversal experiments.

Reuse unchanged BF16/FP32 Strassen arithmetic. Only BlockSpec pipeline buffer
counts and M/N traversal change. Depths 1 and 2 only; modern Pallas required.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v6e_v001 as prior
from . import kernels_v6e_v002 as hybrid
from . import kernels_v001 as base
from .kernels_two_level_v001 import ORDER


def make_matmul(shape,tile,depth,*,mode='deferred',buffers=2,traversal='mn',interpret=False,validate_tpu_alignment=True,vmem_limit_bytes=112*1024**2):
    if buffers not in (1,2,3) or traversal not in ('mn','nm'):raise ValueError('Invalid pipeline/traversal')
    if mode=='hybrid' and depth!=2:raise ValueError('Hybrid is depth 2')
    contract=prior.make_matmul(shape,tile,depth,mode='deferred' if mode=='hybrid' else mode,interpret=interpret,
        validate_tpu_alignment=validate_tpu_alignment,vmem_limit_bytes=vmem_limit_bytes)
    bm,bn,bk=tile;mp,kp,nn=contract.metadata['padded_shape_mkn']
    if mode=='hybrid':
        scratch=(7,bm//2,bn//2);body=functools.partial(hybrid.kernel,order=ORDER,nk=kp//bk)
    else:
        scratch=tuple(contract.metadata['scratch_shape']);body=functools.partial(prior.kernel,depth=depth,mode=mode,order=ORDER,nk=kp//bk)
    swapped=traversal=='nm'
    a_map=(lambda j,i,s:(i,s)) if swapped else (lambda i,j,s:(i,s))
    b_map=(lambda j,i,s:(s,j)) if swapped else (lambda i,j,s:(s,j))
    c_map=(lambda j,i,s:(i,j)) if swapped else (lambda i,j,s:(i,j))
    options={} if interpret else {'pipeline_mode':pl.Buffered(buffer_count=buffers)}
    call=pl.pallas_call(body,grid=(nn//bn,mp//bm,kp//bk) if swapped else (mp//bm,nn//bn,kp//bk),
        in_specs=[pl.BlockSpec((bm,bk),a_map,**options),pl.BlockSpec((bk,bn),b_map,**options)],
        out_specs=pl.BlockSpec((bm,bn),c_map),out_shape=jax.ShapeDtypeStruct((mp,nn),jnp.float32),
        scratch_shapes=[pltpu.VMEM(scratch,jnp.float32)],
        compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,name=f'v6e_s{depth}_{mode}_buf{buffers}_{traversal}')
    def prepared(a,b):base._check_inputs(a,b,(mp,kp,nn));return call(a,b)
    def complete(a,b):return contract.finish(prepared(*contract.prepare(a,b)))
    complete.prepare,complete.kernel,complete.finish=contract.prepare,prepared,contract.finish
    complete.metadata={**contract.metadata,'kernel_version':'kernels_v6e_v003','variant':mode,
        'pipeline_buffers':buffers,'traversal':traversal,'scratch_shape':list(scratch),
        'scratch_bytes':4*scratch[0]*scratch[1]*scratch[2]}
    return complete
