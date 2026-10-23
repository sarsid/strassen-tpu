"""V6e experiments: defer reconstruction across K or reconstruct in SSA per panel.

Depth is restricted to 1 or 2. BF16 operands/combinations and FP32 products,
accumulation and output match the prior arithmetic contract. Deferred mode
changes FP32 addition order, so numerical eligibility must be measured anew.
No intermediate products are written to HBM. Compiler-managed Pallas input
pipelining remains enabled; this version changes accumulator lifetimes.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from .kernels_two_level_v001 import quarters, operands, locations, TARGETS, ORDER


def combine(products, order):
    result = [None] * 4
    for product in order:
        value = products[product]
        for q, sign in TARGETS[product]:
            signed = value if sign == 1 else -value
            result[q] = signed if result[q] is None else result[q] + signed
    return result


def join(parts):
    return jnp.concatenate((jnp.concatenate(parts[:2], axis=1),
                            jnp.concatenate(parts[2:], axis=1)), axis=0)


def panel_product(a, b, depth, order):
    if depth == 0:
        return base._dot(a, b)
    aq, bq = quarters(a), quarters(b)
    products = {}
    for product in order:
        left, right = operands(aq, bq, product)
        products[product] = panel_product(left, right, depth-1, order)
    return join(combine(products, order))


def kernel(a_ref, b_ref, out_ref, leaf_ref, *, depth, mode, order, nk):
    step = pl.program_id(2)
    if mode == 'panel':
        @pl.when(step == 0)
        def zero_output():
            out_ref[...] = jnp.zeros(out_ref.shape, jnp.float32)
        out_ref[...] += panel_product(a_ref[...], b_ref[...], depth, order)
        return

    @pl.when(step == 0)
    def zero_leaves():
        leaf_ref[...] = jnp.zeros(leaf_ref.shape, jnp.float32)

    def accumulate(a, b, remaining, index):
        if remaining == 0:
            leaf_ref[index, ...] += base._dot(a, b)
            return
        aq, bq = quarters(a), quarters(b)
        for product in order:
            left, right = operands(aq, bq, product)
            accumulate(left, right, remaining-1, index*7+product-1)

    accumulate(a_ref[...], b_ref[...], depth, 0)

    @pl.when(step == nk-1)
    def finish():
        def reconstruct(remaining, index):
            if remaining == 0:
                return leaf_ref[index, ...]
            products = {p: reconstruct(remaining-1, index*7+p-1) for p in order}
            return join(combine(products, order))
        out_ref[...] = reconstruct(depth, 0)


def make_matmul(shape, tile, depth, *, mode='deferred', order='interleaved',
                interpret=False, validate_tpu_alignment=True, vmem_limit_bytes=112*1024**2):
    if depth not in (1, 2):
        raise ValueError('Only depths 1 and 2 are in scope')
    if mode not in ('deferred', 'panel') or order not in ('interleaved', 'standard'):
        raise ValueError('Unknown reconstruction mode or product order')
    shape = base._positive_triple(shape, 'shape (M,K,N)')
    tile = base._positive_triple(tile, 'tile (BM,BN,BK)')
    bm, bn, bk = tile
    divisor = 2**depth
    alignment = (8*divisor, 128*divisor, 128*divisor) if validate_tpu_alignment else (divisor,)*3
    if any(v % a for v, a in zip(tile, alignment)):
        raise ValueError('Tile fails recursive leaf alignment')
    if not validate_tpu_alignment and not interpret:
        raise ValueError('Unaligned tiles permitted only for CPU algebra tests')
    if type(vmem_limit_bytes) is not int or vmem_limit_bytes <= 0:
        raise ValueError('Invalid VMEM allowance')
    m, k, n = shape
    mp, kp, nn = [((v+t-1)//t)*t for v, t in zip(shape, (bm,bk,bn))]
    padded = mp, kp, nn
    leaf_shape = (7**depth, bm//divisor, bn//divisor) if mode == 'deferred' else (1,8,128)
    product_order = ORDER if order == 'interleaved' else tuple(range(1,8))
    call = pl.pallas_call(
        functools.partial(kernel, depth=depth, mode=mode, order=product_order, nk=kp//bk),
        grid=(mp//bm, nn//bn, kp//bk),
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),
                  pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))],
        out_specs=pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)),
        out_shape=jax.ShapeDtypeStruct((mp,nn),jnp.float32),
        scratch_shapes=[pltpu.VMEM(leaf_shape,jnp.float32)],
        compiler_params=base._TPUCompilerParams(
            dimension_semantics=('parallel','parallel','arbitrary'), vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret, name=f'v6e_s{depth}_{mode}_{order}')
    def prepare(a,b):
        base._check_inputs(a,b,shape)
        if shape == padded:
            return a,b
        return jnp.pad(a,((0,mp-m),(0,kp-k))), jnp.pad(b,((0,kp-k),(0,nn-n)))
    def prepared(a,b):
        base._check_inputs(a,b,padded)
        return call(a,b)
    def finish(c):
        if c.shape != (mp,nn):
            raise ValueError('Incorrect padded output shape')
        return c[:m,:n]
    def complete(a,b):
        return finish(prepared(*prepare(a,b)))
    complete.prepare, complete.kernel, complete.finish = prepare, prepared, finish
    complete.metadata = dict(kernel_version='kernels_v6e_v001', algorithm='strassen',
        variant=mode+'_'+order, shape_mkn=list(shape), padded_shape_mkn=list(padded),
        tile_bm_bn_bk=list(tile), input_dtype='bfloat16', output_dtype='float32',
        accumulation_dtype='float32', dot_precision='DEFAULT',
        strassen_levels_per_tile=depth, leaf_products_per_panel=7**depth,
        leaf_shape_mkn=[bm//divisor,bk//divisor,bn//divisor],
        reconstruction_frequency='last_K_panel' if mode=='deferred' else 'every_K_panel',
        product_order=product_order, scratch_shape=list(leaf_shape),
        scratch_bytes=4*leaf_shape[0]*leaf_shape[1]*leaf_shape[2],
        vmem_limit_bytes=vmem_limit_bytes, padding_required=shape!=padded,
        padded_volume_ratio=mp*kp*nn/(m*k*n), interpret=interpret,
        complete_call_scope='device_padding+matmul+crop; excludes host_transfer/compile')
    return complete
