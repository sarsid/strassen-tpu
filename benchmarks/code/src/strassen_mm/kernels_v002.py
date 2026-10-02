"""MM kernel v002: add a full-tile classical output-accumulator candidate.

All v001 algorithms/variants remain available through the same explicit
prepare/kernel/finish interface. The new classical variant accumulates into
the FP32 output tile, avoiding a separate full-tile scratch accumulator. It
uses exactly the same BF16 dot/FP32 accumulation contract and sequential K
panels. There is one product per full-tile panel, so a product-interleaving
variant would be inert and is deliberately not offered for full-tile cubic.

The frozen kernels_v001 module is reused without modifying its source.
"""
from __future__ import annotations

import functools

from strassen_mm import kernels_v001 as original


VERSION = "kernels_v002"
ALGORITHMS = original.ALGORITHMS
VARIANTS = original.VARIANTS
enable_qualified_mosaic_v7_compat = original.enable_qualified_mosaic_v7_compat


def _full_output_kernel(a_ref, b_ref, out_ref, unused_scratch_ref):
    del unused_scratch_ref
    step = original.pl.program_id(2)

    @original.pl.when(step == 0)
    def zero():
        out_ref[...] = original.jnp.zeros(out_ref.shape, original.jnp.float32)

    out_ref[...] += original._dot(a_ref[...], b_ref[...])


def make_matmul(algorithm, shape, tile=None, *, variant="plain", interpret=False,
                vmem_limit_bytes=None):
    if algorithm != "cubic_full" or variant != "output_accumulator":
        fn = original.make_matmul(algorithm, shape, tile, variant=variant,
                                  interpret=interpret, vmem_limit_bytes=vmem_limit_bytes)
        fn.metadata.update(kernel_version=VERSION, implementation_module="kernels_v001")
        return fn
    shape = original._positive_triple(shape, "shape (M, K, N)")
    tile = original.check_tile(algorithm, tile)
    if vmem_limit_bytes is not None and (type(vmem_limit_bytes) is not int or vmem_limit_bytes <= 0):
        raise ValueError("vmem_limit_bytes must be a positive integer or None")
    m, k, n = shape
    bm, bn, bk = tile
    mp, kp, nn = (((size + block - 1) // block) * block
                  for size, block in zip(shape, (bm, bk, bn)))
    padded_shape = (mp, kp, nn)
    padded_call = original.pl.pallas_call(
        _full_output_kernel, grid=(mp // bm, nn // bn, kp // bk),
        in_specs=[original.pl.BlockSpec((bm, bk), lambda i, j, s: (i, s)),
                  original.pl.BlockSpec((bk, bn), lambda i, j, s: (s, j))],
        out_specs=original.pl.BlockSpec((bm, bn), lambda i, j, s: (i, j)),
        out_shape=original.jax.ShapeDtypeStruct((mp, nn), original.jnp.float32),
        scratch_shapes=[original.pltpu.VMEM((1,), original.jnp.float32)],
        compiler_params=original._TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret, name=f"{VERSION}_cubic_full_output_accumulator",
    )

    def prepare(a, b):
        original._check_inputs(a, b, shape)
        if padded_shape == shape:
            return a, b
        return (original.jnp.pad(a, ((0, mp - m), (0, kp - k))),
                original.jnp.pad(b, ((0, kp - k), (0, nn - n))))

    def kernel(a, b):
        original._check_inputs(a, b, padded_shape)
        return padded_call(a, b)

    def finish(c):
        if c.shape != (mp, nn):
            raise ValueError(f"expected padded output shape {(mp, nn)}")
        return c[:m, :n]

    def complete(a, b):
        return finish(kernel(*prepare(a, b)))

    complete.prepare, complete.kernel, complete.finish = prepare, kernel, finish
    complete.metadata = {
        "kernel_version": VERSION, "implementation_module": VERSION,
        "algorithm": algorithm, "variant": variant,
        "shape_mkn": list(shape), "padded_shape_mkn": list(padded_shape),
        "tile_bm_bn_bk": list(tile), "padding_required": padded_shape != shape,
        "padded_volume_ratio": (mp * kp * nn) / (m * k * n),
        "input_dtype": "bfloat16", "output_dtype": "float32",
        "accumulation_dtype": "float32", "dot_precision": "DEFAULT",
        "strassen_combination_dtype": None, "strassen_levels_per_tile": 0,
        "accumulator_storage": "output_tile", "product_order": None,
        "interpret": bool(interpret), "vmem_limit_bytes": vmem_limit_bytes,
        "complete_call_scope": "device_padding+matmul+crop; excludes host_transfer/compile",
        "kernel_scope": "prepared_device_inputs_to_padded_output",
    }
    return complete
