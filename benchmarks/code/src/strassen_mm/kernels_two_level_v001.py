"""Two Strassen levels inside a Pallas tile, with BF16 rounding at each level.

The outer seven products are each computed using seven smaller products. One
FP32 half-output-tile scratch buffer is reused for each outer product, then
added/subtracted into the persistent FP32 output tile. No HBM intermediate
products are materialized. Both levels use the existing [4,6,5,2,7,3,1] order.
The preserved one-level implementation is the comparison baseline.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from strassen_mm import kernels_v001 as base

ORDER=(4,6,5,2,7,3,1)
TARGETS={1:((0,1),(3,1)),2:((2,1),(3,-1)),3:((1,1),(3,1)),
         4:((0,1),(2,1)),5:((0,-1),(1,1)),6:((3,1),),7:((0,1),)}


def quarters(x):
    m,n=x.shape;hm,hn=m//2,n//2
    return x[:hm,:hn],x[:hm,hn:],x[hm:,:hn],x[hm:,hn:]


def operands(a,b,product):
    a0,a1,a2,a3=a;b0,b1,b2,b3=b
    def add(x,y):return (x+y).astype(jnp.bfloat16)
    def sub(x,y):return (x-y).astype(jnp.bfloat16)
    if product==1:return add(a0,a3),add(b0,b3)
    if product==2:return add(a2,a3),b0
    if product==3:return a0,sub(b1,b3)
    if product==4:return a3,sub(b2,b0)
    if product==5:return add(a0,a1),b3
    if product==6:return sub(a2,a0),add(b0,b1)
    if product==7:return sub(a1,a3),add(b2,b3)
    raise ValueError('Invalid Strassen product')


def locations(rows,cols):
    hm,hn=rows//2,cols//2
    return ((slice(0,hm),slice(0,hn)),(slice(0,hm),slice(hn,cols)),
            (slice(hm,rows),slice(0,hn)),(slice(hm,rows),slice(hn,cols)))


def update(target,parts,product,value):
    for index,sign in TARGETS[product]:
        if sign==1:target[parts[index]]+=value
        else:target[parts[index]]-=value


def kernel(a_ref,b_ref,out_ref,scratch_ref):
    step=pl.program_id(2)
    @pl.when(step==0)
    def zero():out_ref[...]=jnp.zeros(out_ref.shape,jnp.float32)
    a=quarters(a_ref[...]);b=quarters(b_ref[...])
    outer_locations=locations(*out_ref.shape)
    inner_locations=locations(*scratch_ref.shape)
    for outer in ORDER:
        left,right=operands(a,b,outer)
        la,rb=quarters(left),quarters(right)
        scratch_ref[...]=jnp.zeros(scratch_ref.shape,jnp.float32)
        for inner in ORDER:
            lhs,rhs=operands(la,rb,inner)
            product=base._dot(lhs,rhs)
            update(scratch_ref,inner_locations,inner,product)
        update(out_ref,outer_locations,outer,scratch_ref[...])


def make_matmul(shape,tile,*,interpret=False,vmem_limit_bytes=48*1024**2):
    shape=base._positive_triple(shape,'shape (M,K,N)')
    tile=base._positive_triple(tile,'tile (BM,BN,BK)')
    bm,bn,bk=tile;m,k,n=shape
    for value,alignment in zip(tile,(32,512,512)):
        if value%alignment:raise ValueError('Two levels require BM/BN/BK multiples of 32/512/512')
    if vmem_limit_bytes is not None and (type(vmem_limit_bytes) is not int or vmem_limit_bytes<=0):
        raise ValueError('Invalid VMEM budget')
    mp,kp,nn=[((s+t-1)//t)*t for s,t in zip(shape,(bm,bk,bn))]
    padded=(mp,kp,nn)
    call=pl.pallas_call(kernel,grid=(mp//bm,nn//bn,kp//bk),
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))],
        out_specs=pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)),
        out_shape=jax.ShapeDtypeStruct((mp,nn),jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm//2,bn//2),jnp.float32)],
        compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,name='strassen_two_level_v001')
    def prepare(a,b):
        base._check_inputs(a,b,shape)
        if shape==padded:return a,b
        return jnp.pad(a,((0,mp-m),(0,kp-k))),jnp.pad(b,((0,kp-k),(0,nn-n)))
    def prepared(a,b):
        base._check_inputs(a,b,padded)
        return call(a,b)
    def finish(c):
        if c.shape!=(mp,nn):raise ValueError('Incorrect padded output shape')
        return c[:m,:n]
    def complete(a,b):return finish(prepared(*prepare(a,b)))
    complete.prepare,complete.kernel,complete.finish=prepare,prepared,finish
    complete.metadata={'kernel_version':'kernels_two_level_v001','algorithm':'strassen','variant':'two_level_scratch',
        'shape_mkn':list(shape),'padded_shape_mkn':list(padded),'tile_bm_bn_bk':list(tile),
        'input_dtype':'bfloat16','output_dtype':'float32','accumulation_dtype':'float32','dot_precision':'DEFAULT',
        'strassen_combination_dtype':'bfloat16 at each of two levels','strassen_levels_per_tile':2,
        'leaf_products_per_panel':49,'leaf_shape_mkn':[bm//4,bk//4,bn//4],
        'outer_and_inner_product_order':list(ORDER),'scratch_shape':[bm//2,bn//2],
        'scratch_bytes':bm*bn,'padding_required':shape!=padded,'padded_volume_ratio':mp*kp*nn/(m*k*n),
        'interpret':interpret,'vmem_limit_bytes':vmem_limit_bytes,
        'complete_call_scope':'device_padding+two_level_matmul+crop; excludes host_transfer/compile'}
    return complete
