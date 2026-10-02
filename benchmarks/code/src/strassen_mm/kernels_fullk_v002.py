"""Aligned MM with explicit final FP32/BF16 stores and pipeline buffers.

S1 retains seven products across K. S2 retains seven outer products, using the
unchanged depth-one panel arithmetic internally. Only the final store rounds
to the requested output dtype; all reconstruction and accumulation stay FP32.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from . import kernels_v6e_v001 as arithmetic
from .kernels_two_level_v001 import ORDER, quarters, operands
from .kernels_fullk_v001 import parent_modules


def kernel(a_ref, b_ref, out_ref, acc_ref, *, depth, nk):
    step = pl.program_id(2)
    @pl.when(step == 0)
    def zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    aq, bq = quarters(a_ref[...]), quarters(b_ref[...])
    for product in ORDER:
        a, b = operands(aq, bq, product)
        value = base._dot(a, b) if depth == 1 else arithmetic.panel_product(a, b, 1, ORDER)
        acc_ref[product - 1, ...] += value
    @pl.when(step == nk - 1)
    def finish():
        value = arithmetic.join(arithmetic.combine({p: acc_ref[p - 1, ...] for p in ORDER}, ORDER))
        out_ref[...] = value.astype(out_ref.dtype)


def make_matmul(shape, tile=None, implementation='current', *, depth=0,
                output_dtype='float32', buffers=2, vmem_limit_bytes=112*1024**2,
                interpret=False):
    shape = base._positive_triple(shape, 'shape (M,K,N)'); m, k, n = shape
    dtype = jnp.dtype(output_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)):
        raise ValueError('Only FP32/BF16 output')
    if buffers not in (1, 2):
        raise ValueError('Only one or two pipeline buffers')
    metadata = {}
    if implementation == 'native':
        def fn(a, b):
            base._check_inputs(a, b, shape)
            return jnp.matmul(a, b, precision=jax.lax.Precision.DEFAULT,
                              preferred_element_type=jnp.float32).astype(dtype)
    else:
        tile = base._positive_triple(tile, 'tile (BM,BN,BK)'); bm, bn, bk = tile
        if m % bm or k % bk or n % bn:
            raise ValueError('Padding is outside this study')
        if implementation == 'incumbent' and dtype == jnp.dtype(jnp.float32):
            from .kernels_v6e_v003 import make_matmul as old
            fn = old(shape, tile, depth, mode='deferred' if depth == 1 else 'hybrid',
                     buffers=2, vmem_limit_bytes=vmem_limit_bytes, interpret=interpret)
            metadata = dict(fn.metadata)
        else:
            if implementation in ('current', 'incumbent'):
                if depth not in (1, 2):
                    raise ValueError('Only depths 1 and 2')
                divisor = 2**depth
                if bm % (8*divisor) or bn % (128*divisor) or bk % (128*divisor):
                    raise ValueError('Recursive leaf alignment')
                body = functools.partial(kernel, depth=depth, nk=k//bk)
                scratch = (7, bm//2, bn//2)
            elif implementation == 'cubic':
                sp, cubic = parent_modules(); sp.check_tiling(bm, bn, bk)
                body = functools.partial(cubic._cubic_blocked_kernel, nk=k//bk,
                                         dot_precision=jax.lax.Precision.DEFAULT)
                scratch = (4, bm//2, bn//2)
                metadata['parent_cubic_source'] = '95be1fb088656a89813b04492e1d77c66b36ccf9'
            else:
                raise ValueError('Unknown implementation')
            options = {} if interpret else {'pipeline_mode': pl.Buffered(buffer_count=buffers)}
            call = pl.pallas_call(body, grid=(m//bm, n//bn, k//bk),
                in_specs=[pl.BlockSpec((bm, bk), lambda i,j,s: (i,s), **options),
                          pl.BlockSpec((bk, bn), lambda i,j,s: (s,j), **options)],
                out_specs=pl.BlockSpec((bm, bn), lambda i,j,s: (i,j)),
                out_shape=jax.ShapeDtypeStruct((m,n), dtype),
                scratch_shapes=[pltpu.VMEM(scratch, jnp.float32)],
                compiler_params=base._TPUCompilerParams(
                    dimension_semantics=('parallel','parallel','arbitrary'), vmem_limit_bytes=vmem_limit_bytes),
                interpret=interpret, name=f'fullk_v002_{implementation}_{depth}_{output_dtype}_b{buffers}')
            def fn(a, b):
                base._check_inputs(a, b, shape)
                return call(a, b)
            metadata.update(scratch_shape=list(scratch), scratch_bytes=4*scratch[0]*scratch[1]*scratch[2])
    fn.metadata = {**metadata, 'kernel_version':'kernels_fullk_v002', 'implementation':implementation,
        'algorithm':implementation, 'shape_mkn':list(shape), 'padded_shape_mkn':list(shape),
        'padding_required':False, 'padded_volume_ratio':1., 'tile_bm_bn_bk':list(tile) if tile else None,
        'depth':depth, 'strassen_levels_per_tile':depth, 'input_dtype':'bfloat16',
        'pre_add_dtype':'bfloat16', 'accumulation_dtype':'float32', 'output_dtype':output_dtype,
        'output_rounding':'one final store after all FP32 accumulation and reconstruction',
        'pipeline_buffers':buffers if tile else None, 'vmem_limit_bytes':vmem_limit_bytes if tile else None,
        'k_panels':k//tile[2] if tile else None, 'full_contraction':bool(tile and tile[2] == k),
        'complete_call_scope':'aligned MM including final output conversion; excludes transfer and compilation'}
    return fn
