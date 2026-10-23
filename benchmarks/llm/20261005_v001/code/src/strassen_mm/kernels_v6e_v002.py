"""Depth-two hybrid: inner reconstruction per panel, outer reconstruction after K.

The first v6e optimization screen found that retaining all 49 leaf accumulators
can exceed VMEM, while panel reconstruction fits. This alternative retains seven
half-tile products across K, reconstructing their inner level in SSA per panel.
It uses 7/4 output-tile equivalents of FP32 scratch rather than 49/16.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from . import kernels_v6e_v001 as prior
from .kernels_two_level_v001 import quarters, operands, ORDER


def kernel(a_ref,b_ref,out_ref,outer_ref,*,order,nk):
    step=pl.program_id(2)
    @pl.when(step==0)
    def zero():outer_ref[...]=jnp.zeros(outer_ref.shape,jnp.float32)
    aq,bq=quarters(a_ref[...]),quarters(b_ref[...])
    for product in order:
        left,right=operands(aq,bq,product)
        outer_ref[product-1,...] += prior.panel_product(left,right,1,order)
    @pl.when(step==nk-1)
    def finish():
        products={p:outer_ref[p-1,...] for p in order}
        out_ref[...]=prior.join(prior.combine(products,order))


def make_matmul(shape,tile,*,order='interleaved',interpret=False,
                validate_tpu_alignment=True,vmem_limit_bytes=112*1024**2):
    # Reuse the checked public shape/precision/padding contract.
    contract=prior.make_matmul(shape,tile,2,mode='deferred',order=order,interpret=interpret,
        validate_tpu_alignment=validate_tpu_alignment,vmem_limit_bytes=vmem_limit_bytes)
    m,k,n=shape;bm,bn,bk=tile
    mp,kp,nn=contract.metadata['padded_shape_mkn']
    product_order=ORDER if order=='interleaved' else tuple(range(1,8))
    scratch=(7,bm//2,bn//2)
    call=pl.pallas_call(functools.partial(kernel,order=product_order,nk=kp//bk),
        grid=(mp//bm,nn//bn,kp//bk),
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))],
        out_specs=pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)),
        out_shape=jax.ShapeDtypeStruct((mp,nn),jnp.float32),
        scratch_shapes=[pltpu.VMEM(scratch,jnp.float32)],
        compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,name='v6e_s2_outer_deferred_'+order)
    def prepared(a,b):
        base._check_inputs(a,b,(mp,kp,nn));return call(a,b)
    def complete(a,b):return contract.finish(prepared(*contract.prepare(a,b)))
    complete.prepare,complete.kernel,complete.finish=contract.prepare,prepared,contract.finish
    complete.metadata={**contract.metadata,'kernel_version':'kernels_v6e_v002',
        'variant':'outer_deferred_'+order,'reconstruction_frequency':'inner every K panel; outer last K panel',
        'scratch_shape':list(scratch),'scratch_bytes':4*scratch[0]*scratch[1]*scratch[2]}
    return complete
