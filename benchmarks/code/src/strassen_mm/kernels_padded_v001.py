"""Boundary policies around unchanged v6e Strassen kernels, depths 1 and 2.

All slices, zero padding, partial-result additions and output concatenations
belong to the compiled complete call. No values are dropped. This first probe
uses ordinary JAX operations between Pallas calls; it does not claim fused
boundary loads or an in-kernel K correction.
"""
import math
import jax
import jax.numpy as jnp
from . import kernels_v001 as base
from . import kernels_v6e_v003 as interior


def partition(size, tile):
    core = size // tile * tile
    return [(lo, hi) for lo, hi in ((0, core), (core, size)) if hi > lo]


def make_matmul(shape, tile, depth, *, policy='edge_tiles', mode=None,
                buffers=2, traversal='mn', interpret=False,
                validate_tpu_alignment=True, vmem_limit_bytes=112*1024**2):
    shape = base._positive_triple(shape, 'shape (M,K,N)')
    tile = base._positive_triple(tile, 'tile (BM,BN,BK)')
    if depth not in (1, 2) or policy not in ('edge_tiles', 'native_fringe'):
        raise ValueError('Expected depth 1/2 and a registered boundary policy')
    mode = mode or ('deferred' if depth == 1 else 'hybrid')
    kwargs = dict(mode=mode, buffers=buffers, traversal=traversal,
                  interpret=interpret, validate_tpu_alignment=validate_tpu_alignment,
                  vmem_limit_bytes=vmem_limit_bytes)
    original = interior.make_matmul(shape, tile, depth, **kwargs)
    m, k, n = shape
    bm, bn, bk = tile
    mc, kc, nc = m//bm*bm, k//bk*bk, n//bn*bn
    segments = [partition(x, t) for x, t in zip(shape, (bm, bk, bn))]
    # Regular shapes must use exactly the incumbent execution path.
    if (mc, kc, nc) == shape:
        def complete(a, b):
            return original(a, b)
        calls = [dict(shape_mkn=list(shape), tile_bm_bn_bk=list(tile),
                      method='strassen', padded_shape_mkn=list(shape))]
        padding_volume = m*k*n
        strassen_volume = m*k*n
        native_volume = 0
    elif policy == 'edge_tiles':
        divisor = 2**depth
        alignment = (8*divisor, 128*divisor, 128*divisor) if validate_tpu_alignment else (divisor,)*3
        calls, functions = [], {}
        padding_volume = 0
        strassen_volume, native_volume = m*k*n, 0
        for mi, (ml, mh) in enumerate(segments[0]):
            for ni, (nl, nh) in enumerate(segments[2]):
                for ki, (kl, kh) in enumerate(segments[1]):
                    block = (mh-ml, kh-kl, nh-nl)
                    local_tile = tuple(min(t, math.ceil(x/a)*a)
                        for x, t, a in zip((block[0], block[2], block[1]), tile, alignment))
                    fn = interior.make_matmul(block, local_tile, depth, **kwargs)
                    functions[mi, ni, ki] = fn
                    padded = fn.metadata['padded_shape_mkn']
                    padding_volume += math.prod(padded)
                    calls.append(dict(shape_mkn=list(block), tile_bm_bn_bk=list(local_tile),
                        padded_shape_mkn=padded, method='strassen', offsets_mkn=[ml, kl, nl]))

        def complete(a, b):
            base._check_inputs(a, b, shape)
            output_rows = []
            for mi, (ml, mh) in enumerate(segments[0]):
                output_columns = []
                for ni, (nl, nh) in enumerate(segments[2]):
                    value = None
                    for ki, (kl, kh) in enumerate(segments[1]):
                        product = functions[mi, ni, ki](a[ml:mh, kl:kh], b[kl:kh, nl:nh])
                        value = product if value is None else value + product
                    output_columns.append(value)
                output_rows.append(output_columns[0] if len(output_columns) == 1 else jnp.concatenate(output_columns, axis=1))
            return output_rows[0] if len(output_rows) == 1 else jnp.concatenate(output_rows, axis=0)
    else:
        calls = []
        core_fn = interior.make_matmul((mc, kc, nc), tile, depth, **kwargs) if min(mc, kc, nc) else None
        strassen_volume = mc*kc*nc
        native_volume = m*k*n-strassen_volume
        padding_volume = m*k*n  # Explicit padding only; Native internal padding is unknown.
        if core_fn:
            calls.append(dict(shape_mkn=[mc, kc, nc], tile_bm_bn_bk=list(tile),
                              padded_shape_mkn=[mc, kc, nc], method='strassen'))
        for ml, mh in segments[0]:
            for nl, nh in segments[2]:
                is_core = core_fn is not None and ml == 0 and mh == mc and nl == 0 and nh == nc
                kl = kc if is_core else 0
                if kl < k:
                    calls.append(dict(shape_mkn=[mh-ml, k-kl, nh-nl], method='native', offsets_mkn=[ml, kl, nl]))

        def native(a, b):
            return jnp.matmul(a, b, precision=jax.lax.Precision.DEFAULT,
                              preferred_element_type=jnp.float32)

        def complete(a, b):
            base._check_inputs(a, b, shape)
            output_rows = []
            for ml, mh in segments[0]:
                output_columns = []
                for nl, nh in segments[2]:
                    if core_fn is not None and ml == 0 and mh == mc and nl == 0 and nh == nc:
                        value = core_fn(a[:mc, :kc], b[:kc, :nc])
                        if kc < k:
                            value = value + native(a[:mc, kc:], b[kc:, :nc])
                    else:
                        value = native(a[ml:mh, :], b[:, nl:nh])
                    output_columns.append(value)
                output_rows.append(output_columns[0] if len(output_columns) == 1 else jnp.concatenate(output_columns, axis=1))
            return output_rows[0] if len(output_rows) == 1 else jnp.concatenate(output_rows, axis=0)

    complete.metadata = dict(original.metadata, kernel_version='kernels_padded_v001',
        algorithm='strassen_boundary', variant=policy, boundary_policy=policy,
        interior_version='kernels_v6e_v003', core_shape_mkn=[mc, kc, nc],
        padded_shape_mkn=list(shape), padded_volume_ratio=padding_volume/(m*k*n),
        padding_required=any(c.get('padded_shape_mkn', c['shape_mkn']) != c['shape_mkn'] for c in calls),
        boundary_calls=calls, explicit_multiplication_volume=padding_volume,
        strassen_useful_volume_fraction=strassen_volume/(m*k*n),
        native_useful_volume_fraction=native_volume/(m*k*n),
        native_internal_padding='unknown', aligned_fast_path=(mc, kc, nc)==shape,
        complete_call_scope='device slicing+padding+all MM+partial sums+concatenation+crop; excludes host transfer/compile')
    return complete
