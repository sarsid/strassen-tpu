"""Study adapters; executed historical kernels remain unchanged.

The v6e adapter adds the established tile-multiple zero-pad/crop boundary
contract around the qualified joint kernel. The v5e adapter only adds a final
output cast to the original complete call; it does not change its arithmetic,
VMEM allowance, scheduling, padding, or tiling search.
"""
import math


def make_matmul(arm, shape, *, interpret=False):
    import jax.numpy as jnp
    dtype = arm['output_dtype']
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('Unsupported output contract')
    if arm.get('architecture') == 'v5e':
        from . import kernels_v002 as old
        if arm['family'] == 'two_level':
            from .kernels_two_level_v001 import make_matmul as old_s2
            fn = old_s2(shape, tuple(arm['tile']), vmem_limit_bytes=48*1024**2, interpret=interpret)
        else:
            fn = old.make_matmul(arm['algorithm'], shape, tuple(arm['tile']) if arm['tile'] else None,
                variant=arm['variant'], vmem_limit_bytes=None if arm['family']=='native' else 48*1024**2,
                interpret=interpret)
        if dtype == 'float32':
            return fn
        def converted(a, b):
            return fn(a, b).astype(jnp.bfloat16)
        converted.metadata = {**fn.metadata, 'output_dtype': dtype,
            'output_adapter': 'final BF16 cast of unchanged historical FP32 complete call',
            'complete_call_scope': 'historical padding+MM+crop+final BF16 cast; excludes transfer/compile'}
        return converted
    if arm.get('architecture') != 'v6e':
        raise ValueError('Explicit architecture required')
    if arm['implementation']=='cubic_full':
        from .kernels_v002 import make_matmul as full
        fn=full('cubic_full',shape,tuple(arm['tile']),variant='output_accumulator',
            vmem_limit_bytes=arm['vmem_limit_bytes'],interpret=interpret)
        if dtype=='float32':return fn
        def cast_cubic(a,b):return fn(a,b).astype(jnp.bfloat16)
        cast_cubic.metadata={**fn.metadata,'output_dtype':dtype,'output_adapter':'one final BF16 cast'}
        return cast_cubic
    from .kernels_joint_v001 import make_matmul as joint
    tile = tuple(arm['tile']) if arm['tile'] else None
    m, k, n = shape
    padded = tuple(math.ceil(x/t)*t for x,t in zip(shape, (tile[0],tile[2],tile[1]))) if tile else tuple(shape)
    fn = joint(padded, tile, implementation=arm['implementation'], depth=arm['depth'],
        accumulator=arm.get('accumulator','products'), output_dtype=dtype,
        buffers=arm['buffers'], vmem_limit_bytes=arm['vmem_limit_bytes'], interpret=interpret)
    if padded == tuple(shape):
        return fn
    from .kernels_v001 import _check_inputs
    mp, kp, nn = padded
    def complete(a, b):
        _check_inputs(a,b,tuple(shape))
        ap=jnp.pad(a,((0,mp-m),(0,kp-k)))
        bp=jnp.pad(b,((0,kp-k),(0,nn-n)))
        return fn(ap,bp)[:m,:n]
    complete.metadata = {**fn.metadata, 'boundary_adapter':'tile_multiple_zero_pad_crop_v001',
        'shape_mkn':list(shape), 'padded_shape_mkn':list(padded), 'padding_required':True,
        'padded_volume_ratio':math.prod(padded)/math.prod(shape),
        'full_contraction':bool(tile and tile[2]==kp),
        'exact_unpadded_full_contraction':bool(tile and tile[2]==k),
        'complete_call_scope':'device tile-multiple padding+MM+crop+final output conversion; excludes transfer/compile'}
    return complete
