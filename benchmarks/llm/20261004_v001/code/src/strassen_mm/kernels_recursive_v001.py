"""One to four Strassen levels entirely inside each Pallas tile.

Depth-first traversal reuses one FP32 product buffer at each non-leaf level.
All seven products are statically expanded; there are 7**depth leaf dots per
K panel. Intermediate products stay in VMEM. BF16 pre-additions round at every
level, and output/reconstruction/K-panel accumulation use FP32.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from .kernels_two_level_v001 import ORDER, quarters, operands, locations, update


def accumulate(a, b, target, scratch, depth):
    aa, bb = quarters(a), quarters(b)
    parts = locations(*target.shape)
    for product in ORDER:
        left, right = operands(aa, bb, product)
        if depth == 1:
            value = base._dot(left, right)
        else:
            child = scratch[0]
            child[...] = jnp.zeros(child.shape, jnp.float32)
            accumulate(left, right, child, scratch[1:], depth - 1)
            value = child[...]
        update(target, parts, product, value)


def make_matmul(shape, tile, depth, *, interpret=False, vmem_limit_bytes=112*1024**2,
                validate_tpu_alignment=True):
    shape = base._positive_triple(shape, 'shape (M,K,N)')
    tile = base._positive_triple(tile, 'tile (BM,BN,BK)')
    if type(depth) is not int or not 1 <= depth <= 4:
        raise ValueError('Expected one to four recursion levels')
    if not validate_tpu_alignment and not interpret:
        raise ValueError('Small algebra-only tiles are restricted to CPU interpretation')
    factor = 2**depth
    alignment = (8*factor, 128*factor, 128*factor) if validate_tpu_alignment else (factor,)*3
    if any(t % a for t, a in zip(tile, alignment)):
        raise ValueError('Tile dimensions must be divisible by '+str(alignment))
    if type(vmem_limit_bytes) is not int or vmem_limit_bytes <= 0:
        raise ValueError('Invalid VMEM budget')
    bm, bn, bk = tile
    m, k, n = shape
    mp, kp, npad = [((s+t-1)//t)*t for s,t in zip(shape,(bm,bk,bn))]
    padded = (mp, kp, npad)
    scratch_shapes = [(bm//2**i, bn//2**i) for i in range(1, depth)]

    def kernel(a_ref, b_ref, out_ref, *scratch):
        @pl.when(pl.program_id(2) == 0)
        def zero():
            out_ref[...] = jnp.zeros(out_ref.shape, jnp.float32)
        accumulate(a_ref[...], b_ref[...], out_ref, scratch, depth)

    call = pl.pallas_call(kernel, grid=(mp//bm, npad//bn, kp//bk),
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)), pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))],
        out_specs=pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)),
        out_shape=jax.ShapeDtypeStruct((mp,npad),jnp.float32),
        scratch_shapes=[pltpu.VMEM(s,jnp.float32) for s in scratch_shapes],
        compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),
                                              vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret, name='strassen_depth_'+str(depth)+'_v001')

    def prepare(a,b):
        base._check_inputs(a,b,shape)
        return (a,b) if shape == padded else (jnp.pad(a,((0,mp-m),(0,kp-k))),jnp.pad(b,((0,kp-k),(0,npad-n))))
    def prepared(a,b):
        base._check_inputs(a,b,padded)
        return call(a,b)
    def finish(c):
        return c[:m,:n]
    def complete(a,b):
        return finish(prepared(*prepare(a,b)))
    complete.prepare, complete.kernel, complete.finish = prepare, prepared, finish
    complete.metadata = dict(kernel_version='kernels_recursive_v001', algorithm='strassen',
        variant='recursive_depth_first_unrolled', shape_mkn=list(shape), padded_shape_mkn=list(padded),
        tile_bm_bn_bk=list(tile), input_dtype='bfloat16', output_dtype='float32', accumulation_dtype='float32',
        dot_precision='DEFAULT', strassen_levels_per_tile=depth, leaf_products_per_panel=7**depth,
        leaf_shape_mkn=[bm//factor,bk//factor,bn//factor], scratch_shapes=scratch_shapes,
        scratch_bytes=sum(x*y*4 for x,y in scratch_shapes), vmem_limit_bytes=vmem_limit_bytes,
        product_order=list(ORDER), padding_required=shape!=padded, interpret=interpret,
        padded_volume_ratio=mp*kp*npad/(m*k*n), tpu_alignment_validated=validate_tpu_alignment,
        complete_call_scope='device padding + recursive tiled MM + crop; excludes host transfer and compilation')
    return complete
