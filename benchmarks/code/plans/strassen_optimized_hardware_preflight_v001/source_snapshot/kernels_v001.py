"""Version 001: standalone BF16 matrix multiplication for the MM study.

Adapted from this project's historical ``strassen-tpu/strassen_pallas.py``
(classical Strassen products and dependency-spaced product order), and
``strassen-tpu/experiments/qwen3/cubic_full_fused.py`` (full-tile classical
contraction). The new implementation has no model epilogues, dispatcher,
historical-project imports, or import-time environment changes.

Contract: shape is (M, K, N); tile is (BM, BN, BK). Inputs are BF16, every dot
uses DEFAULT precision with FP32 accumulation, and output is FP32. Strassen
operand additions/subtractions are rounded back to BF16 before the dot. Those
roundings are intrinsic to this implementation and must be measured by N4;
FP32 output does not mean that Strassen has FP32 input-combination accuracy.

``make_matmul`` returns an un-jitted function with explicit ``prepare``,
``kernel``, ``finish`` and ``metadata`` attributes. The complete function
includes device-side zero padding and output cropping. ``kernel`` alone
expects already padded device inputs and returns the padded output. Compile
and time these paths separately; do not report kernel-only timing as the
complete operation's cost. Host transfer and compilation are separate costs.

Every Pallas grid program handles one BM-by-BN output tile and one BK panel.
The K grid axis is sequential; accumulators are initialized on its first
panel and stored on its last. One Strassen level is applied inside each tile
product, not recursively to the whole input matrix.
"""

from __future__ import annotations

import functools
from importlib import metadata as package_metadata

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


VERSION = "kernels_v001"
ALGORITHMS = ("native", "cubic_full", "cubic_quadrant", "strassen")
VARIANTS = (
    "plain", "interleaved", "output_accumulator",
    "interleaved_output_accumulator",
)
_TPUCompilerParams = getattr(pltpu, "TPUCompilerParams", None)
if _TPUCompilerParams is None:
    _TPUCompilerParams = pltpu.CompilerParams


def enable_qualified_mosaic_v7_compat():
    """Opt in to the previously qualified Colab Mosaic serialization fix.

    Only JAX 0.7.2 with libtpu 0.0.21/0.0.21.1 is eligible. The caller must
    record this return value and run its own TPU preflight. This helper is
    idempotent and does not change compilation or memory flags. Adapted from
    historical ``strassen-tpu/mosaic_compat.py``.
    """
    try:
        libtpu = package_metadata.version("libtpu")
    except package_metadata.PackageNotFoundError:
        libtpu = None
    qualified = jax.__version__ == "0.7.2" and libtpu in ("0.0.21", "0.0.21.1")
    if qualified:
        from jax._src import tpu_custom_call

        original = getattr(
            tpu_custom_call, "_strassen_mm_original_get_ir_version",
            tpu_custom_call.get_ir_version,
        )
        tpu_custom_call._strassen_mm_original_get_ir_version = original

        def get_ir_version_v7(ctx):
            version = original(ctx)
            return 7 if version is None else min(version, 7)

        tpu_custom_call.get_ir_version = get_ir_version_v7
    return {
        "jax": jax.__version__, "libtpu": libtpu,
        "override_enabled": qualified,
        "target_ir_version": 7 if qualified else None,
    }


def _positive_triple(values, label):
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must contain three positive integers") from error
    if len(values) != 3 or any(type(x) is not int or x <= 0 for x in values):
        raise ValueError(f"{label} must contain three positive integers")
    return values


def check_tile(algorithm, tile):
    """Validate the TPU alignment contract, including interpreter tests."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm!r}")
    bm, bn, bk = _positive_triple(tile, "tile (BM, BN, BK)")
    alignments = (16, 256, 256) if algorithm in (
        "strassen", "cubic_quadrant"
    ) else (8, 128, 128)
    for name, value, alignment in zip(("BM", "BN", "BK"), (bm, bn, bk), alignments):
        if value % alignment:
            raise ValueError(f"{name}={value} must be a multiple of {alignment}")
    return bm, bn, bk


def _check_inputs(a, b, shape):
    m, k, n = shape
    if a.shape != (m, k) or b.shape != (k, n):
        raise ValueError(f"expected input shapes {(m, k)} and {(k, n)}")
    if a.dtype != jnp.dtype(jnp.bfloat16) or b.dtype != jnp.dtype(jnp.bfloat16):
        raise ValueError("both inputs must be BF16; input conversion is caller-owned")


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32,
                   precision=jax.lax.Precision.DEFAULT)


def _full_kernel(a_ref, b_ref, out_ref, acc_ref, *, nk):
    step = pl.program_id(2)

    @pl.when(step == 0)
    def zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    acc_ref[...] += _dot(a_ref[...], b_ref[...])

    @pl.when(step == nk - 1)
    def finish():
        out_ref[...] = acc_ref[...]


def _split_kernel(a_ref, b_ref, out_ref, acc_ref, *, nk, strassen,
                  interleaved, output_accumulator):
    step = pl.program_id(2)
    bm, bk = a_ref.shape
    bn = b_ref.shape[1]
    hm, hk, hn = bm // 2, bk // 2, bn // 2
    a0, a1 = a_ref[:hm, :hk], a_ref[:hm, hk:]
    a2, a3 = a_ref[hm:, :hk], a_ref[hm:, hk:]
    b0, b1 = b_ref[:hk, :hn], b_ref[:hk, hn:]
    b2, b3 = b_ref[hk:, :hn], b_ref[hk:, hn:]
    quadrants = (
        (slice(0, hm), slice(0, hn)), (slice(0, hm), slice(hn, bn)),
        (slice(hm, bm), slice(0, hn)), (slice(hm, bm), slice(hn, bn)),
    )

    @pl.when(step == 0)
    def zero():
        if output_accumulator:
            out_ref[...] = jnp.zeros(out_ref.shape, jnp.float32)
        else:
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    def update(index, product, subtract=False):
        target = out_ref if output_accumulator else acc_ref
        location = quadrants[index] if output_accumulator else index
        if subtract:
            target[location] -= product
        else:
            target[location] += product

    if strassen:
        # Explicit BF16 casts specify rounding of every input combination.
        def add(x, y):
            return (x + y).astype(jnp.bfloat16)

        def sub(x, y):
            return (x - y).astype(jnp.bfloat16)

        def p1():
            p = _dot(add(a0, a3), add(b0, b3))
            update(0, p)
            update(3, p)

        def p2():
            p = _dot(add(a2, a3), b0)
            update(2, p)
            update(3, p, subtract=True)

        def p3():
            p = _dot(a0, sub(b1, b3))
            update(1, p)
            update(3, p)

        def p4():
            p = _dot(a3, sub(b2, b0))
            update(0, p)
            update(2, p)

        def p5():
            p = _dot(add(a0, a1), b3)
            update(0, p, subtract=True)
            update(1, p)

        def p6():
            update(3, _dot(sub(a2, a0), add(b0, b1)))

        def p7():
            update(0, _dot(sub(a1, a3), add(b2, b3)))

        order = (p4, p6, p5, p2, p7, p3, p1) if interleaved else (
            p1, p2, p3, p4, p5, p6, p7
        )
        for product in order:
            product()
    else:
        # Exactly two half-K products per output quadrant. Product order is
        # the only difference between plain and interleaved cubic controls.
        products = (
            (0, a0, b0), (0, a1, b2), (1, a0, b1), (1, a1, b3),
            (2, a2, b0), (2, a3, b2), (3, a2, b1), (3, a3, b3),
        )
        order = (0, 2, 4, 6, 1, 3, 5, 7) if interleaved else range(8)
        for index in order:
            quadrant, lhs, rhs = products[index]
            update(quadrant, _dot(lhs, rhs))

    if not output_accumulator:
        @pl.when(step == nk - 1)
        def finish():
            for index, location in enumerate(quadrants):
                out_ref[location] = acc_ref[index]


def make_matmul(algorithm, shape, tile=None, *, variant="plain", interpret=False,
                vmem_limit_bytes=None):
    """Construct a pure-MM implementation with explicit preparation boundaries.

    Algorithms: ``native`` (native JAX/XLA dot), ``cubic_full`` (one full-tile
    classical product per K panel), ``cubic_quadrant`` (eight half-tile
    classical products), ``strassen`` (seven half-tile products).

    Split-kernel variants form a 2x2 ablation: plain/interleaved product order
    and scratch/output accumulators. ``plain`` retains four FP32 scratch
    quadrants; ``output_accumulator`` instead updates the FP32 output tile
    directly. This changes buffer scheduling, not algebra or output precision.
    The compiler is free to produce equivalent schedules; gains are empirical.

    No variant provides more than one Strassen level. The full cubic and
    native baselines support only ``plain``. Native uses original dimensions
    and ignores tile (its padding is the compiler's implementation detail).
    ``interpret=True`` is correctness-only CPU emulation, never a TPU timing.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if algorithm in ("native", "cubic_full") and variant != "plain":
        raise ValueError(f"{algorithm} supports only the plain variant")
    if vmem_limit_bytes is not None and (
        type(vmem_limit_bytes) is not int or vmem_limit_bytes <= 0
    ):
        raise ValueError("vmem_limit_bytes must be a positive integer or None")
    shape = _positive_triple(shape, "shape (M, K, N)")
    m, k, n = shape
    interleaved = variant.startswith("interleaved")
    output_accumulator = "output_accumulator" in variant

    if algorithm == "native":
        tile = None
        padded_shape = shape
        padded_call = _dot
    else:
        tile = check_tile(algorithm, tile)
        bm, bn, bk = tile
        mp, kp, np_ = (((size + block - 1) // block) * block
                       for size, block in zip(shape, (bm, bk, bn)))
        padded_shape = (mp, kp, np_)
        if algorithm == "cubic_full":
            body = functools.partial(_full_kernel, nk=kp // bk)
            scratch = pltpu.VMEM((bm, bn), jnp.float32)
        else:
            body = functools.partial(
                _split_kernel, nk=kp // bk, strassen=algorithm == "strassen",
                interleaved=interleaved, output_accumulator=output_accumulator,
            )
            scratch = pltpu.VMEM(
                (1,) if output_accumulator else (4, bm // 2, bn // 2),
                jnp.float32,
            )
        padded_call = pl.pallas_call(
            body, grid=(mp // bm, np_ // bn, kp // bk),
            in_specs=[pl.BlockSpec((bm, bk), lambda i, j, s: (i, s)),
                      pl.BlockSpec((bk, bn), lambda i, j, s: (s, j))],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, s: (i, j)),
            out_shape=jax.ShapeDtypeStruct((mp, np_), jnp.float32),
            scratch_shapes=[scratch],
            compiler_params=_TPUCompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=interpret, name=f"{VERSION}_{algorithm}_{variant}",
        )

    mp, kp, np_ = padded_shape

    def prepare(a, b):
        _check_inputs(a, b, shape)
        if padded_shape == shape:
            return a, b
        return (jnp.pad(a, ((0, mp - m), (0, kp - k))),
                jnp.pad(b, ((0, kp - k), (0, np_ - n))))

    def kernel(a, b):
        _check_inputs(a, b, padded_shape)
        return padded_call(a, b)

    def finish(c):
        if c.shape != (mp, np_):
            raise ValueError(f"expected padded output shape {(mp, np_)}")
        return c[:m, :n]

    def complete(a, b):
        return finish(kernel(*prepare(a, b)))

    complete.prepare = prepare
    complete.kernel = kernel
    complete.finish = finish
    complete.metadata = {
        "kernel_version": VERSION,
        "algorithm": algorithm, "variant": variant,
        "shape_mkn": list(shape), "padded_shape_mkn": list(padded_shape),
        "tile_bm_bn_bk": list(tile) if tile else None,
        "padding_required": padded_shape != shape,
        "padded_volume_ratio": (mp * kp * np_) / (m * k * n),
        "input_dtype": "bfloat16", "output_dtype": "float32",
        "accumulation_dtype": "float32", "dot_precision": "DEFAULT",
        "strassen_combination_dtype": "bfloat16" if algorithm == "strassen" else None,
        "strassen_levels_per_tile": 1 if algorithm == "strassen" else 0,
        "accumulator_storage": (
            "compiler_managed" if algorithm == "native" else
            "output_tile" if output_accumulator else
            "full_tile_scratch" if algorithm == "cubic_full" else "quadrant_scratch"
        ),
        "product_order": (
            [4, 6, 5, 2, 7, 3, 1] if interleaved else list(range(1, 8))
        ) if algorithm == "strassen" else None,
        "interpret": bool(interpret), "vmem_limit_bytes": vmem_limit_bytes,
        "complete_call_scope": "device_padding+matmul+crop; excludes host_transfer/compile",
        "kernel_scope": "prepared_device_inputs_to_padded_output",
    }
    return complete
