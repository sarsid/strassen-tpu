"""Joint accumulator/geometry/buffering study; BF16 inputs, FP32 arithmetic.

Products retains seven outer products across K. Outputs updates four output
quadrants per panel, using FP32 output directly or four FP32 scratch quadrants
for final BF16 output. S2 reconstructs each inner S1 product inside its panel.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from . import kernels_v6e_v001 as arithmetic
from .kernels_two_level_v001 import ORDER, quarters, operands
from .kernels_fullk_v002 import make_matmul as previous


CONTRIBUTIONS = {1: ((0, 1), (3, 1)), 2: ((2, 1), (3, -1)),
    3: ((1, 1), (3, 1)), 4: ((0, 1), (2, 1)),
    5: ((0, -1), (1, 1)), 6: ((3, 1),), 7: ((0, 1),)}


def output_kernel(a_ref, b_ref, out_ref, acc_ref, *, depth, nk, direct):
    step = pl.program_id(2)
    bm, bn = out_ref.shape
    hm, hn = bm // 2, bn // 2
    slices = ((slice(0, hm), slice(0, hn)), (slice(0, hm), slice(hn, bn)),
              (slice(hm, bm), slice(0, hn)), (slice(hm, bm), slice(hn, bn)))
    @pl.when(step == 0)
    def zero():
        if direct:
            out_ref[...] = jnp.zeros(out_ref.shape, jnp.float32)
        else:
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    aq, bq = quarters(a_ref[...]), quarters(b_ref[...])
    for product in ORDER:
        a, b = operands(aq, bq, product)
        value = base._dot(a, b) if depth == 1 else arithmetic.panel_product(a, b, 1, ORDER)
        for quadrant, sign in CONTRIBUTIONS[product]:
            target = out_ref if direct else acc_ref
            index = slices[quadrant] if direct else quadrant
            if sign == 1:
                target[index] += value
            else:
                target[index] -= value
    if not direct:
        @pl.when(step == nk - 1)
        def store():
            for quadrant, index in enumerate(slices):
                out_ref[index] = acc_ref[quadrant].astype(out_ref.dtype)


def make_matmul(shape, tile=None, *, depth=0, accumulator='products',
                output_dtype='float32', buffers=2, implementation='current',
                vmem_limit_bytes=112 * 1024**2, interpret=False):
    if implementation in ('native', 'cubic') or accumulator == 'products':
        fn = previous(shape, tile, implementation, depth=depth,
            output_dtype=output_dtype, buffers=buffers,
            vmem_limit_bytes=vmem_limit_bytes, interpret=interpret)
        fn.metadata = {**fn.metadata, 'joint_kernel_version': 'kernels_joint_v001',
                       'accumulator_strategy': accumulator if depth else None,
                       'accumulator_storage': 'seven_outer_products' if depth else implementation}
        return fn
    if accumulator != 'outputs' or implementation != 'current':
        raise ValueError('Unsupported accumulator/implementation')
    # Reuse the established alignment, dtype and no-padding guards.
    contract = previous(shape, tile, 'current', depth=depth,
        output_dtype=output_dtype, buffers=buffers,
        vmem_limit_bytes=vmem_limit_bytes, interpret=interpret)
    m, k, n = shape
    bm, bn, bk = tile
    dtype = jnp.dtype(output_dtype)
    direct = dtype == jnp.dtype(jnp.float32)
    scratch = (1,) if direct else (4, bm // 2, bn // 2)
    options = {} if interpret else {'pipeline_mode': pl.Buffered(buffer_count=buffers)}
    call = pl.pallas_call(functools.partial(output_kernel, depth=depth, nk=k // bk, direct=direct),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, s: (i, s), **options),
                  pl.BlockSpec((bk, bn), lambda i, j, s: (s, j), **options)],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, s: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_shapes=[pltpu.VMEM(scratch, jnp.float32)],
        compiler_params=base._TPUCompilerParams(
            dimension_semantics=('parallel', 'parallel', 'arbitrary'),
            vmem_limit_bytes=vmem_limit_bytes), interpret=interpret,
        name=f'joint_v001_outputs_s{depth}_{output_dtype}_b{buffers}')
    def fn(a, b):
        base._check_inputs(a, b, shape)
        return call(a, b)
    count = 1
    for extent in scratch:
        count *= extent
    fn.metadata = {**contract.metadata, 'joint_kernel_version': 'kernels_joint_v001',
        'accumulator_strategy': accumulator,
        'accumulator_storage': 'output_tile' if direct else 'four_quadrants',
        'scratch_shape': list(scratch), 'scratch_bytes': 4 * count,
        'reconstruction_frequency': 'outer contributions each K panel; inner S1 within panel'}
    return fn
