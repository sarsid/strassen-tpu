"""One-level classical Strassen GEMM for a single TPU v5e.

Each Pallas program loads one A and B tile into VMEM, forms the ten classical
Strassen input combinations, executes seven half-size MXU products instead of
eight, and accumulates four output quadrants in FP32. No Strassen intermediate
is written to HBM.

``STRASSEN_VMEM_PROFILE`` selects one of three measured deployment points:
``compact16`` (default), ``spatial34``, or ``max48``. Larger profiles install a
process-wide scoped-VMEM ceiling before JAX initializes unless the caller has
already supplied one.
"""

from __future__ import annotations

import functools
import os


_VMEM_PROFILES = {
    "compact16": {
        "scoped_kib": 16 * 1024,
        "tile": (1024, 1024, 512),
        "kernel_limit_bytes": None,
    },
    "spatial34": {
        "scoped_kib": 34 * 1024,
        "tile": (1024, 2048, 512),
        "kernel_limit_bytes": 32 * 1024 * 1024,
    },
    "max48": {
        "scoped_kib": 48 * 1024,
        "tile": (2048, 2048, 512),
        "kernel_limit_bytes": 47 * 1024 * 1024,
    },
}

VMEM_PROFILE = os.environ.get("STRASSEN_VMEM_PROFILE", "compact16")
if VMEM_PROFILE not in _VMEM_PROFILES:
    choices = ", ".join(sorted(_VMEM_PROFILES))
    raise ValueError(
        f"unknown STRASSEN_VMEM_PROFILE={VMEM_PROFILE!r}; choose {choices}"
    )

_profile = _VMEM_PROFILES[VMEM_PROFILE]
TUNED_SCOPED_VMEM_KIB = _profile["scoped_kib"]
TUNED_BM, TUNED_BN, TUNED_BK = _profile["tile"]
TUNED_VMEM_LIMIT_BYTES = _profile["kernel_limit_bytes"]
TUNED_SQUARE_CROSSOVER = 8192

# These are exact workload shapes with same-run wins. Avoid extrapolating the
# dispatcher to arbitrary rectangles until padding, layout, and fusion are
# measured end to end.
TUNED_PROJECTION_SHAPES = frozenset({
    (8192, 4096, 8192),
    (8192, 4096, 12288),
    (8192, 4096, 28672),
    (8192, 14336, 4096),
})

# Dependency-spaced product execution re-won these exact max48 shapes in a
# separate 60-sample comparison against the original P1..P7 order. Do not
# extrapolate it to other tiles or shapes without requalification.
TUNED_INTERLEAVED_SHAPES = frozenset({
    (8192, 8192, 8192),
    (8192, 4096, 8192),
    (8192, 4096, 12288),
    (8192, 4096, 28672),
})

_SCOPED_VMEM_OPTION = "--xla_tpu_scoped_vmem_limit_kib="
_libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
# Installing a process-wide scoped-VMEM ceiling affects every compilation in
# the process, including GEMMs that fall back to native XLA -- so it is an
# application-level decision, not something a library should do invisibly.
# It stays the default for the non-default profiles (which exist precisely to
# raise the ceiling, and whose measurements assume it), but is now named,
# suppressible via strassen_config.configure(manage_ceiling=False), and
# reported in MANAGES_PROCESS_CEILING / PROFILE_INFO. The default profile
# compact16 has never installed a ceiling.
MANAGE_CEILING = os.environ.get("STRASSEN_MANAGE_CEILING", "1") != "0"
MANAGES_PROCESS_CEILING = (
    MANAGE_CEILING
    and VMEM_PROFILE != "compact16"
    and _SCOPED_VMEM_OPTION not in _libtpu_args
)
if MANAGES_PROCESS_CEILING:
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(
        value
        for value in (
            _libtpu_args,
            f"{_SCOPED_VMEM_OPTION}{TUNED_SCOPED_VMEM_KIB}",
        )
        if value
    )

# Recorded by benchmarks so an executable's provenance names the profile it was
# compiled under, rather than leaving it implicit in the environment.
PROFILE_INFO = {
    "profile": VMEM_PROFILE,
    "tile": (TUNED_BM, TUNED_BN, TUNED_BK),
    "kernel_limit_bytes": TUNED_VMEM_LIMIT_BYTES,
    "scoped_vmem_kib": TUNED_SCOPED_VMEM_KIB,
    "installed_process_ceiling": MANAGES_PROCESS_CEILING,
    "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
}


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import mosaic_compat


LANE = 128
SUBLANE = 8
TPU_COMPILER_PARAMS = getattr(pltpu, "TPUCompilerParams", None)
if TPU_COMPILER_PARAMS is None:
    TPU_COMPILER_PARAMS = pltpu.CompilerParams


MOSAIC_IR_V7_COMPAT = mosaic_compat.enable_v7_compat()


def check_tiling(bm, bn, bk):
    """Validate v5e lane alignment after the one-level split."""
    requirements = (
        ("bm", bm, 2 * SUBLANE),
        ("bn", bn, 2 * LANE),
        ("bk", bk, 2 * LANE),
    )
    errors = [
        f"{name}={value} must be a multiple of {multiple}"
        for name, value, multiple in requirements
        if value % multiple
    ]
    if errors:
        raise ValueError("tile breaks TPU v5e alignment:\n  " + "\n  ".join(errors))


def _classical_strassen_kernel(
    a_ref,
    b_ref,
    o_ref,
    acc_ref,
    *,
    nk,
    compute_dtype,
    dot_precision,
    interleave_products,
    block_permutation,
    formula_variant,
    output_is_accumulator,
    classical_panels=(),
    panel_schedule_ref=None,
    epilogue=None,
    residual_ref=None,
    bias_ref=None,
    finalize_quadrant=None,
    product_aware_swiglu=False,
):
    step = pl.program_id(2)
    m, k = a_ref.shape
    _, n = b_ref.shape
    hm, hk, hn = m // 2, k // 2, n // 2

    original_a = (
        a_ref[:hm, :hk],
        a_ref[:hm, hk:],
        a_ref[hm:, :hk],
        a_ref[hm:, hk:],
    )
    original_b = (
        b_ref[:hk, :hn],
        b_ref[:hk, hn:],
        b_ref[hk:, :hn],
        b_ref[hk:, hn:],
    )
    swap_m = bool(block_permutation & 1)
    swap_k = bool(block_permutation & 2)
    swap_n = bool(block_permutation & 4)

    def transformed_index(row, column, swap_rows, swap_columns):
        return 2 * (row ^ swap_rows) + (column ^ swap_columns)

    a0, a1, a2, a3 = tuple(
        original_a[transformed_index(row, column, swap_m, swap_k)]
        for row in range(2)
        for column in range(2)
    )
    b0, b1, b2, b3 = tuple(
        original_b[transformed_index(row, column, swap_k, swap_n)]
        for row in range(2)
        for column in range(2)
    )
    output_indices = tuple(
        transformed_index(row, column, swap_m, swap_n)
        for row in range(2)
        for column in range(2)
    )

    dot = functools.partial(
        jnp.dot,
        preferred_element_type=jnp.float32,
        precision=dot_precision,
    )

    @pl.when(step == 0)
    def _zero_accumulators():
        if output_is_accumulator:
            o_ref[...] = jnp.zeros(o_ref.shape, dtype=o_ref.dtype)
        else:
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

    def update(index, value, subtract=False):
        index = output_indices[index]
        if not output_is_accumulator:
            if subtract:
                acc_ref[index] -= value
            else:
                acc_ref[index] += value
        elif index == 0:
            if subtract:
                o_ref[:hm, :hn] -= value
            else:
                o_ref[:hm, :hn] += value
        elif index == 1:
            if subtract:
                o_ref[:hm, hn:] -= value
            else:
                o_ref[:hm, hn:] += value
        elif index == 2:
            if subtract:
                o_ref[hm:, :hn] -= value
            else:
                o_ref[hm:, :hn] += value
        elif subtract:
            o_ref[hm:, hn:] -= value
        else:
            o_ref[hm:, hn:] += value

    def update_basis(index, value, subtract=False):
        """Update an encoded output basis without permuting its coordinates."""
        if subtract:
            acc_ref[index] -= value
        else:
            acc_ref[index] += value

    def store_transformed(index, value):
        """Store one decoded quadrant after undoing the block permutation."""
        index = output_indices[index]
        if index == 0:
            o_ref[:hm, :hn] = value.astype(o_ref.dtype)
        elif index == 1:
            o_ref[:hm, hn:] = value.astype(o_ref.dtype)
        elif index == 2:
            o_ref[hm:, :hn] = value.astype(o_ref.dtype)
        else:
            o_ref[hm:, hn:] = value.astype(o_ref.dtype)

    cd = compute_dtype

    if finalize_quadrant is not None and (
        not output_is_accumulator
        or formula_variant != "classical"
        or block_permutation != 0
        or classical_panels
        or panel_schedule_ref is not None
    ):
        raise ValueError(
            "quadrant finalization requires unpermuted classical Strassen "
            "with an output accumulator and no exact-panel policy")
    if product_aware_swiglu and (
        output_is_accumulator
        or formula_variant != "classical"
        or block_permutation != 0
        or classical_panels
        or panel_schedule_ref is not None
        or epilogue != "swiglu"
    ):
        raise ValueError(
            "product-aware SwiGLU requires unpermuted classical Strassen "
            "with quadrant accumulators and no exact-panel policy")

    def p1():
        # P1=(A0+A3)(B0+B3): C0 += P1, C3 += P1.
        product = dot(
            (a0.astype(cd) + a3.astype(cd)).astype(cd),
            (b0.astype(cd) + b3.astype(cd)).astype(cd),
        )
        update(0, product)
        update(3, product)

    def p2():
        # P2=(A2+A3)B0: C2 += P2, C3 -= P2.
        product = dot(
            (a2.astype(cd) + a3.astype(cd)).astype(cd), b0.astype(cd)
        )
        update(2, product)
        update(3, product, subtract=True)

    def p3():
        # P3=A0(B1-B3): C1 += P3, C3 += P3.
        product = dot(
            a0.astype(cd), (b1.astype(cd) - b3.astype(cd)).astype(cd)
        )
        update(1, product)
        update(3, product)

    def p4():
        # P4=A3(B2-B0): C0 += P4, C2 += P4.
        product = dot(
            a3.astype(cd), (b2.astype(cd) - b0.astype(cd)).astype(cd)
        )
        update(0, product)
        update(2, product)

    def p5():
        # P5=(A0+A1)B3: C0 -= P5, C1 += P5.
        product = dot(
            (a0.astype(cd) + a1.astype(cd)).astype(cd), b3.astype(cd)
        )
        update(0, product, subtract=True)
        update(1, product)

    def p6():
        # P6=(A2-A0)(B0+B1): C3 += P6.
        product = dot(
            (a2.astype(cd) - a0.astype(cd)).astype(cd),
            (b0.astype(cd) + b1.astype(cd)).astype(cd),
        )
        update(3, product)

    def p7():
        # P7=(A1-A3)(B2+B3): C0 += P7.
        product = dot(
            (a1.astype(cd) - a3.astype(cd)).astype(cd),
            (b2.astype(cd) + b3.astype(cd)).astype(cd),
        )
        update(0, product)

    # Dumas--Pernet--Sedoglavic's power-of-two rank-7 formula has a lower
    # Frobenius growth factor than classical Strassen (12.2034 vs 14.8284).
    # All non-unit input coefficients are exact binary halves/quarters.  The
    # four persistent accumulators use a sparse output basis that reduces the
    # formula's raw 18 product contributions to ten unscaled updates:
    #
    #   z0 = C3
    #   z1 = C0 - C1/2 + C2/2 - C3/4 = R6 + R7
    #   z2 = C0 + C1/2 - C2/2 - C3/4 = R2 - R4
    #   z3 = C2 - C3/2                 = R1 - R3 + R4 + R7
    #
    # Decoding happens once after the final K panel.  Each encoded operand is
    # formed directly with a balanced depth-two expression so no collection of
    # transformed panels remains live across MXU products.
    half = jnp.asarray(0.5, dtype=cd)
    quarter = jnp.asarray(0.25, dtype=cd)

    def r1():
        product = dot(
            (a2.astype(cd) - a1.astype(cd)).astype(cd),
            (b0.astype(cd) - b3.astype(cd)).astype(cd),
        )
        update_basis(3, product)

    def r2():
        encoded_a = (
            (a0.astype(cd) - a2.astype(cd) * half)
            + (a1.astype(cd) * half - a3.astype(cd) * quarter)
        ).astype(cd)
        encoded_b = (b0.astype(cd) + b1.astype(cd) * half).astype(cd)
        product = dot(encoded_a, encoded_b)
        update_basis(2, product)

    def r3():
        product = dot(
            (a2.astype(cd) - a3.astype(cd) * half).astype(cd),
            (b1.astype(cd) * half - b3.astype(cd)).astype(cd),
        )
        update_basis(0, product)
        update_basis(3, product, subtract=True)

    def r4():
        encoded_b = (
            (b0.astype(cd) * half - b2.astype(cd))
            + (b1.astype(cd) * quarter - b3.astype(cd) * half)
        ).astype(cd)
        product = dot(
            (a1.astype(cd) - a3.astype(cd) * half).astype(cd),
            encoded_b,
        )
        update_basis(2, product, subtract=True)
        update_basis(3, product)

    def r5():
        product = dot(
            (a2.astype(cd) + a3.astype(cd) * half).astype(cd),
            (b1.astype(cd) * half + b3.astype(cd)).astype(cd),
        )
        update_basis(0, product)

    def r6():
        encoded_a = (
            (a0.astype(cd) - a1.astype(cd) * half)
            + (a2.astype(cd) * half - a3.astype(cd) * quarter)
        ).astype(cd)
        encoded_b = (b0.astype(cd) - b1.astype(cd) * half).astype(cd)
        product = dot(encoded_a, encoded_b)
        update_basis(1, product)

    def r7():
        encoded_b = (
            (b0.astype(cd) * half + b2.astype(cd))
            - (b1.astype(cd) * quarter + b3.astype(cd) * half)
        ).astype(cd)
        product = dot(
            (a1.astype(cd) + a3.astype(cd) * half).astype(cd),
            encoded_b,
        )
        update_basis(1, product)
        update_basis(3, product)

    def q1():
        # Dual P1: (A0+A3)(B0+B3): C0 += Q1, C3 += Q1.
        product = dot(
            (a0.astype(cd) + a3.astype(cd)).astype(cd),
            (b0.astype(cd) + b3.astype(cd)).astype(cd),
        )
        update(0, product)
        update(3, product)

    def q2():
        # Dual P2: A0(B1+B3): C1 += Q2, C3 -= Q2.
        product = dot(
            a0.astype(cd), (b1.astype(cd) + b3.astype(cd)).astype(cd)
        )
        update(1, product)
        update(3, product, subtract=True)

    def q3():
        # Dual P3: (A2-A3)B0: C2 += Q3, C3 += Q3.
        product = dot(
            (a2.astype(cd) - a3.astype(cd)).astype(cd), b0.astype(cd)
        )
        update(2, product)
        update(3, product)

    def q4():
        # Dual P4: (A1-A0)B3: C0 += Q4, C1 += Q4.
        product = dot(
            (a1.astype(cd) - a0.astype(cd)).astype(cd), b3.astype(cd)
        )
        update(0, product)
        update(1, product)

    def q5():
        # Dual P5: A3(B0+B2): C0 -= Q5, C2 += Q5.
        product = dot(
            a3.astype(cd), (b0.astype(cd) + b2.astype(cd)).astype(cd)
        )
        update(0, product, subtract=True)
        update(2, product)

    def q6():
        # Dual P6: (A0+A2)(B1-B0): C3 += Q6.
        product = dot(
            (a0.astype(cd) + a2.astype(cd)).astype(cd),
            (b1.astype(cd) - b0.astype(cd)).astype(cd),
        )
        update(3, product)

    def q7():
        # Dual P7: (A1+A3)(B2-B3): C0 += Q7.
        product = dot(
            (a1.astype(cd) + a3.astype(cd)).astype(cd),
            (b2.astype(cd) - b3.astype(cd)).astype(cd),
        )
        update(0, product)

    def run_seven_products():
        if formula_variant == "powers" and interleave_products:
            # Contribution sets: 03, 2, 1, 0, 23, 13, 3.  Only the final two
            # transitions share an accumulator.
            r3()
            r2()
            r6()
            r5()
            r4()
            r7()
            r1()
        elif formula_variant == "powers":
            r1()
            r2()
            r3()
            r4()
            r5()
            r6()
            r7()
        elif formula_variant == "dual" and interleave_products:
            # The dual has the same contribution-set schedule as classical:
            # 02, 3, 01, 23, 0, 13, 03.
            q5()
            q6()
            q4()
            q3()
            q7()
            q2()
            q1()
        elif formula_variant == "dual":
            q1()
            q2()
            q3()
            q4()
            q5()
            q6()
            q7()
        elif interleave_products:
            # Consecutive contribution sets are 02, 3, 01, 23, 0, 13, 03.
            # Only the final transition shares an accumulator.
            p4()
            p6()
            p5()
            p2()
            p7()
            p3()
            p1()
        else:
            p1()
            p2()
            p3()
            p4()
            p5()
            p6()
            p7()

    def run_exact_panel():
        # Exact eight-product cubic update of the same quadrant accumulators,
        # ordered so consecutive updates target distinct quadrants. No input
        # pre-adds, so this panel contributes native-level rounding only.
        update(0, dot(a0.astype(cd), b0.astype(cd)))
        update(1, dot(a0.astype(cd), b1.astype(cd)))
        update(2, dot(a2.astype(cd), b0.astype(cd)))
        update(3, dot(a2.astype(cd), b1.astype(cd)))
        update(0, dot(a1.astype(cd), b2.astype(cd)))
        update(1, dot(a1.astype(cd), b3.astype(cd)))
        update(2, dot(a3.astype(cd), b2.astype(cd)))
        update(3, dot(a3.astype(cd), b3.astype(cd)))

    # Two ways to say "this K panel is exact". The static form bakes the panel
    # set into the traced predicate, so every distinct budget compiles its own
    # executable -- fine for experiments, poor for a per-layer deployment
    # policy. The prefetched form reads the decision from a scalar array, so a
    # single executable serves every budget.
    if product_aware_swiglu:
        def store_swiglu_pair(top):
            gate_index, up_index = ((0, 1) if top else (2, 3))
            destination = slice(0, hm) if top else slice(hm, m)
            o_ref[destination, :] = (
                jax.nn.silu(acc_ref[gate_index]) * acc_ref[up_index]
            ).astype(o_ref.dtype)

        @pl.when(step != nk - 1)
        def _ordinary_panel():
            run_seven_products()

        @pl.when(step == nk - 1)
        def _product_aware_final_panel():
            # Complete the top gate/up pair first, then expose its SwiGLU VPU
            # work while the two full-size P6/P2 MXU products complete the
            # bottom pair. This deliberately changes bottom-quadrant FP32
            # accumulation order, so callers must requalify numerical error.
            p4()
            p5()
            p7()
            p3()
            p1()
            store_swiglu_pair(True)
            p6()
            p2()
            store_swiglu_pair(False)
    elif finalize_quadrant is not None:
        # Keep the dependency-spaced product order, and preserve the relative
        # accumulation order within every output quadrant.  C21 is complete
        # after P2, leaving P7/P3/P1 as independent MXU work; C12 is complete
        # after P3, leaving P1.  The callback can expose their vector epilogues
        # to Mosaic without shrinking any MXU product.  C11/C22 finish at P1.
        @pl.when(step != nk - 1)
        def _ordinary_panel():
            run_seven_products()

        @pl.when(step == nk - 1)
        def _product_aware_final_panel():
            p4()
            p6()
            p5()
            p2()
            finalize_quadrant(2)
            p7()
            p3()
            finalize_quadrant(1)
            p1()
            finalize_quadrant(0)
            finalize_quadrant(3)
    elif panel_schedule_ref is not None:
        is_exact_panel = panel_schedule_ref[step] != 0

        @pl.when(is_exact_panel)
        def _exact_panel_scheduled():
            run_exact_panel()

        @pl.when(jnp.logical_not(is_exact_panel))
        def _strassen_panel_scheduled():
            run_seven_products()
    elif classical_panels:
        is_exact_panel = functools.reduce(
            jnp.logical_or,
            [step == panel for panel in classical_panels],
        )

        @pl.when(is_exact_panel)
        def _exact_panel():
            run_exact_panel()

        @pl.when(jnp.logical_not(is_exact_panel))
        def _strassen_panel():
            run_seven_products()
    else:
        run_seven_products()

    if not output_is_accumulator:
        @pl.when(step == nk - 1)
        def _store_output():
            if product_aware_swiglu:
                return
            if formula_variant == "powers":
                z0, z1, z2, z3 = tuple(acc_ref[index] for index in range(4))
                # Decode and store one quadrant at a time.  Keeping c0..c3
                # live together adds four full FP32 quarter tiles to the
                # compiler's peak VMEM demand without changing the result.
                store_transformed(3, z0)
                store_transformed(
                    2, z3 + z0 * jnp.asarray(0.5, dtype=z0.dtype)
                )
                store_transformed(0, (
                    (z1 + z2) * jnp.asarray(0.5, dtype=z0.dtype)
                    + z0 * jnp.asarray(0.25, dtype=z0.dtype)
                ))
                store_transformed(1, (
                    (z2 - z1)
                    + z3
                    + z0 * jnp.asarray(0.5, dtype=z0.dtype)
                ))
            elif epilogue == "swiglu":
                # The quadrant layout already separates the operands: with the
                # combined gate/up weight column-blocked so each N tile holds
                # gate channels then the matching up channels, quadrants 0/2
                # are gate and 1/3 are up. So SwiGLU is a pure register-level
                # epilogue over accumulators that are already in FP32 -- no
                # extra pass over HBM, and the activation is computed at
                # accumulator precision rather than after a BF16 round trip.
                gate_top, up_top = acc_ref[0], acc_ref[1]
                gate_bottom, up_bottom = acc_ref[2], acc_ref[3]
                o_ref[:hm, :] = (
                    jax.nn.silu(gate_top) * up_top
                ).astype(o_ref.dtype)
                o_ref[hm:, :] = (
                    jax.nn.silu(gate_bottom) * up_bottom
                ).astype(o_ref.dtype)
            elif epilogue == "residual_add":
                # Fold the residual add into the store so the output tile is
                # written once instead of being re-read by a separate kernel.
                o_ref[:hm, :hn] = (
                    acc_ref[0] + residual_ref[:hm, :hn].astype(jnp.float32)
                ).astype(o_ref.dtype)
                o_ref[:hm, hn:] = (
                    acc_ref[1] + residual_ref[:hm, hn:].astype(jnp.float32)
                ).astype(o_ref.dtype)
                o_ref[hm:, :hn] = (
                    acc_ref[2] + residual_ref[hm:, :hn].astype(jnp.float32)
                ).astype(o_ref.dtype)
                o_ref[hm:, hn:] = (
                    acc_ref[3] + residual_ref[hm:, hn:].astype(jnp.float32)
                ).astype(o_ref.dtype)
            elif epilogue == "bias_add":
                # Fold a Linear bias into the final store.  The bias arrives
                # as (1, bn) so Mosaic can use a legal 2-D tiled layout.
                bias = bias_ref[0].astype(jnp.float32)
                o_ref[:hm, :hn] = (acc_ref[0] + bias[:hn]).astype(o_ref.dtype)
                o_ref[:hm, hn:] = (acc_ref[1] + bias[hn:]).astype(o_ref.dtype)
                o_ref[hm:, :hn] = (acc_ref[2] + bias[:hn]).astype(o_ref.dtype)
                o_ref[hm:, hn:] = (acc_ref[3] + bias[hn:]).astype(o_ref.dtype)
            else:
                o_ref[:hm, :hn] = acc_ref[0].astype(o_ref.dtype)
                o_ref[:hm, hn:] = acc_ref[1].astype(o_ref.dtype)
                o_ref[hm:, :hn] = acc_ref[2].astype(o_ref.dtype)
                o_ref[hm:, hn:] = acc_ref[3].astype(o_ref.dtype)


def strassen_matmul(
    a,
    b,
    *,
    bm=TUNED_BM,
    bn=TUNED_BN,
    bk=TUNED_BK,
    dot_precision=jax.lax.Precision.DEFAULT,
    interleave_products=False,
    block_permutation=0,
    formula_variant="classical",
    classical_panels=(),
    panel_schedule=None,
    epilogue=None,
    residual=None,
    bias=None,
    output_dtype=None,
    allow_input_fusion=None,
    product_aware_swiglu=False,
    interpret=False,
    vmem_limit_bytes=None,
):
    """Compute ``a @ b`` with one tile-local classical Strassen level.

    Inputs must share a BF16 or FP32 dtype. BF16 output uses four local FP32
    quadrant accumulators; FP32 output uses the Pallas output tile directly as
    its accumulator. ``interleave_products`` selects a dependency-spaced
    execution order without changing the formula. ``block_permutation`` is a
    three-bit mask that swaps the M, K, and N block halves before applying the
    same formula and maps the quadrants back before returning.
    ``formula_variant`` exposes classical, dual, and lower-growth power-of-two
    research formulas; the tuned dispatcher selects classical only.
    ``allow_input_fusion`` is a tuple aligned with the user-visible operands
    ``(a, b)`` or ``(a, b, residual)``.  It is forwarded to Mosaic's TPU
    custom-call backend configuration so XLA may absorb eligible producers.
    This is a permission, not a guarantee that the backend will fuse them.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("strassen_matmul expects two rank-2 matrices")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions disagree: {k} vs {k2}")
    if a.dtype != b.dtype:
        raise ValueError(f"input dtypes disagree: {a.dtype} vs {b.dtype}")
    if a.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("inputs must be BF16 or FP32")
    if not isinstance(block_permutation, int) or not 0 <= block_permutation < 8:
        raise ValueError("block_permutation must be an integer from 0 through 7")
    if formula_variant not in ("classical", "dual", "powers"):
        raise ValueError(
            "formula_variant must be 'classical', 'dual', or 'powers'"
        )
    classical_panels = tuple(sorted({int(p) for p in classical_panels}))
    if epilogue is not None:
        if epilogue not in ("swiglu", "residual_add", "bias_add"):
            raise ValueError(
                f"unknown epilogue {epilogue!r}; choose 'swiglu' or "
                "'residual_add' or 'bias_add'"
            )
        if formula_variant == "powers":
            raise ValueError("fused epilogues require a quadrant-basis formula")
        if output_dtype is not None and jnp.dtype(output_dtype) != jnp.dtype(
                jnp.bfloat16):
            raise ValueError("fused epilogues write a BF16 output tile")
        if epilogue == "swiglu" and bn % 2:
            raise ValueError(f"swiglu epilogue needs an even bn, got {bn}")
        if epilogue == "residual_add" and residual is None:
            raise ValueError("residual_add epilogue requires residual=")
        if epilogue == "bias_add" and bias is None:
            raise ValueError("bias_add epilogue requires bias=")
        if epilogue == "bias_add" and formula_variant == "powers":
            raise ValueError("bias_add requires a quadrant-basis formula")
    if residual is not None and epilogue != "residual_add":
        raise ValueError("residual= is only used by the residual_add epilogue")
    if bias is not None and epilogue != "bias_add":
        raise ValueError("bias= is only used by the bias_add epilogue")
    if product_aware_swiglu and epilogue != "swiglu":
        raise ValueError(
            "product_aware_swiglu is only valid with epilogue='swiglu'")
    if epilogue == "bias_add" and bias.shape != (n,):
        raise ValueError(f"bias shape {bias.shape} != output width {(n,)}")
    operand_count = 3 if epilogue in ("residual_add", "bias_add") else 2
    if allow_input_fusion is not None:
        allow_input_fusion = tuple(allow_input_fusion)
        if len(allow_input_fusion) != operand_count:
            raise ValueError(
                "allow_input_fusion must contain one boolean per operand: "
                f"expected {operand_count}, got {len(allow_input_fusion)}"
            )
        if not all(isinstance(value, bool) for value in allow_input_fusion):
            raise TypeError("allow_input_fusion entries must be bool")
    if panel_schedule is not None and classical_panels:
        raise ValueError(
            "pass either classical_panels (static) or panel_schedule "
            "(prefetched), not both"
        )
    if classical_panels:
        if formula_variant == "powers":
            raise ValueError(
                "classical_panels requires a quadrant-basis formula"
            )
        if classical_panels[0] < 0 or classical_panels[-1] >= k // bk:
            raise ValueError(
                f"classical_panels {classical_panels} outside 0..{k // bk - 1}"
            )
    for name, dimension, block in (("M", m, bm), ("N", n, bn), ("K", k, bk)):
        if dimension % block:
            raise ValueError(f"{name}={dimension} is not divisible by {block}")
    check_tiling(bm, bn, bk)

    output_dtype = jnp.dtype(a.dtype if output_dtype is None else output_dtype)
    if output_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("output_dtype must be BF16 or FP32")
    # The powers formula accumulates a sparse output basis and must decode it
    # before returning even when the requested result itself is FP32.
    output_is_accumulator = (
        output_dtype == jnp.dtype(jnp.float32)
        and formula_variant != "powers"
    )
    nk = k // bk

    kernel = functools.partial(
        _classical_strassen_kernel,
        nk=nk,
        compute_dtype=a.dtype,
        dot_precision=dot_precision,
        interleave_products=interleave_products,
        block_permutation=block_permutation,
        formula_variant=formula_variant,
        output_is_accumulator=output_is_accumulator,
        classical_panels=classical_panels,
        epilogue=epilogue,
        product_aware_swiglu=product_aware_swiglu,
    )
    scratch_shape = (
        pltpu.VMEM((1,), jnp.float32)
        if output_is_accumulator
        else pltpu.VMEM((4, bm // 2, bn // 2), jnp.float32)
    )
    if panel_schedule is not None:
        if epilogue is not None:
            raise ValueError("panel_schedule does not support fused epilogues")
        # One executable serves every per-layer budget: the exact-panel
        # decision is read from a prefetched scalar array instead of being
        # baked into the traced predicate.
        schedule = jnp.asarray(panel_schedule, dtype=jnp.int32)
        if schedule.shape != (nk,):
            raise ValueError(
                f"panel_schedule must have shape ({nk},), got {schedule.shape}"
            )

        def scheduled_kernel(schedule_ref, a_ref, b_ref, o_ref, acc_ref):
            return _classical_strassen_kernel(
                a_ref, b_ref, o_ref, acc_ref,
                nk=nk,
                compute_dtype=a.dtype,
                dot_precision=dot_precision,
                interleave_products=interleave_products,
                block_permutation=block_permutation,
                formula_variant=formula_variant,
                output_is_accumulator=output_is_accumulator,
                panel_schedule_ref=schedule_ref,
            )

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(m // bm, n // bn, nk),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, step, _: (i, step)),
                pl.BlockSpec((bk, bn), lambda i, j, step, _: (step, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, step, _: (i, j)),
            scratch_shapes=[scratch_shape],
        )
        return pl.pallas_call(
            scheduled_kernel,
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct((m, n), output_dtype),
            compiler_params=TPU_COMPILER_PARAMS(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                allow_input_fusion=(
                    None
                    if allow_input_fusion is None
                    # The prefetched schedule is an implementation operand,
                    # not one of the user-visible matrix operands.
                    else (False,) + allow_input_fusion
                ),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=interpret,
        )(schedule, a, b)

    in_specs = [
        pl.BlockSpec((bm, bk), lambda i, j, step: (i, step)),
        pl.BlockSpec((bk, bn), lambda i, j, step: (step, j)),
    ]
    operands = [a, b]
    if epilogue == "swiglu":
        # SwiGLU halves the output width: each (bm, bn) tile of gate|up
        # produces a (bm, bn // 2) tile of activated intermediate.
        out_specs = pl.BlockSpec((bm, bn // 2), lambda i, j, step: (i, j))
        out_shape = jax.ShapeDtypeStruct((m, n // 2), output_dtype)
    else:
        out_specs = pl.BlockSpec((bm, bn), lambda i, j, step: (i, j))
        out_shape = jax.ShapeDtypeStruct((m, n), output_dtype)
    if epilogue == "residual_add":
        if residual.shape != (m, n):
            raise ValueError(
                f"residual shape {residual.shape} != output shape {(m, n)}"
            )
        in_specs.append(pl.BlockSpec((bm, bn), lambda i, j, step: (i, j)))
        operands.append(residual)
        # Pallas passes refs positionally, so the residual arrives between
        # b_ref and o_ref; adapt rather than reorder the kernel's signature.
        base_kernel = kernel

        def kernel(a_ref, b_ref, residual_ref, o_ref, acc_ref):
            return base_kernel(
                a_ref, b_ref, o_ref, acc_ref, residual_ref=residual_ref
            )
    elif epilogue == "bias_add":
        bias_matrix = bias.reshape(1, n)
        in_specs.append(pl.BlockSpec((1, bn), lambda i, j, step: (0, j)))
        operands.append(bias_matrix)
        # As above, adapt the positional input-ref order Pallas constructs.
        base_kernel = kernel

        def kernel(a_ref, b_ref, bias_ref, o_ref, acc_ref):
            return base_kernel(a_ref, b_ref, o_ref, acc_ref, bias_ref=bias_ref)

    return pl.pallas_call(
        kernel,
        grid=(m // bm, n // bn, nk),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[scratch_shape],
        compiler_params=TPU_COMPILER_PARAMS(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=allow_input_fusion,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(*operands)


def strassen_matmul_lhs_transposed(
    a,
    b,
    *,
    bm=TUNED_BM,
    bn=TUNED_BN,
    bk=TUNED_BK,
    dot_precision=jax.lax.Precision.DEFAULT,
    interleave_products=False,
    output_dtype=None,
    allow_input_fusion=None,
    interpret=False,
    vmem_limit_bytes=None,
):
    """Compute ``a.T @ b`` without materializing a global transpose.

    The first input remains in its producer layout ``(K, M)``. Each program
    loads a ``(bk, bm)`` HBM tile and transposes that tile in VMEM before the
    ordinary Strassen body sees it. This is intended for weight-gradient
    products where XLA's dot can express transpose semantics but an opaque
    Pallas call would otherwise receive a materialized ``swapaxes`` result.

    This deliberately exposes only the options used by the training policy;
    add other epilogues or formula variants only after they receive their own
    matched cubic control.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("strassen_matmul_lhs_transposed expects rank-2 inputs")
    k, m = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"contracting dimensions disagree: {k} vs {k2}")
    if a.dtype != b.dtype:
        raise ValueError(f"input dtypes disagree: {a.dtype} vs {b.dtype}")
    if a.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("inputs must be BF16 or FP32")
    for name, dimension, block in (("M", m, bm), ("N", n, bn), ("K", k, bk)):
        if dimension % block:
            raise ValueError(f"{name}={dimension} is not divisible by {block}")
    check_tiling(bm, bn, bk)
    if allow_input_fusion is not None:
        allow_input_fusion = tuple(allow_input_fusion)
        if len(allow_input_fusion) != 2:
            raise ValueError("allow_input_fusion must contain two booleans")

    output_dtype = jnp.dtype(a.dtype if output_dtype is None else output_dtype)
    output_is_accumulator = output_dtype == jnp.dtype(jnp.float32)
    nk = k // bk

    def kernel(a_physical_ref, b_ref, o_ref, acc_ref):
        # The block spec preserves the producer's (K, M) layout. Only the
        # resident tile is transposed; no KxM temporary exists in HBM.
        a_tile = jnp.swapaxes(a_physical_ref[...], 0, 1)
        return _classical_strassen_kernel(
            a_tile,
            b_ref,
            o_ref,
            acc_ref,
            nk=nk,
            compute_dtype=a.dtype,
            dot_precision=dot_precision,
            interleave_products=interleave_products,
            block_permutation=0,
            formula_variant="classical",
            output_is_accumulator=output_is_accumulator,
        )

    scratch_shape = (
        pltpu.VMEM((1,), jnp.float32)
        if output_is_accumulator
        else pltpu.VMEM((4, bm // 2, bn // 2), jnp.float32)
    )
    return pl.pallas_call(
        kernel,
        grid=(m // bm, n // bn, nk),
        in_specs=[
            pl.BlockSpec((bk, bm), lambda i, j, step: (step, i)),
            pl.BlockSpec((bk, bn), lambda i, j, step: (step, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, step: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), output_dtype),
        scratch_shapes=[scratch_shape],
        compiler_params=TPU_COMPILER_PARAMS(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=allow_input_fusion,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(a, b)


def swiglu_weight_layout(gate_up, bn):
    """Reorder a combined gate/up weight so SwiGLU can be fused in-kernel.

    A combined projection is stored as ``[gate(0..I-1) | up(0..I-1)]``, so a
    single (bm, bn) output tile holds either gate channels or up channels --
    never a matching pair -- and the activation cannot be computed inside the
    kernel. Blocking the columns as
    ``gate[0:h] up[0:h] gate[h:2h] up[h:2h] ...`` with ``h = bn // 2`` puts
    each gate channel in the same tile as its partner, at which point the
    kernel's own quadrant layout separates them exactly (quadrants 0/2 gate,
    1/3 up).

    This is a free offline relayout of the weights -- the same class of
    transformation as the hybrid's intermediate permutation -- and changes
    nothing the model computes.
    """
    two_i = gate_up.shape[1]
    if two_i % 2:
        raise ValueError(f"combined gate/up width {two_i} is not even")
    intermediate = two_i // 2
    half = bn // 2
    if intermediate % half:
        raise ValueError(
            f"intermediate {intermediate} is not divisible by bn//2={half}"
        )
    order = []
    for block in range(intermediate // half):
        start = block * half
        order.extend(range(start, start + half))
        order.extend(range(intermediate + start, intermediate + start + half))
    return gate_up[:, jnp.asarray(order)]


def native_matmul(a, b):
    """Matched native cubic baseline for the tuned BF16 dispatcher."""
    return jnp.matmul(
        a,
        b,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    ).astype(a.dtype)


def _on_tpu(interpret):
    return interpret or jax.default_backend() == "tpu"


def tuned_matmul(a, b, *, interpret=False):
    """Dispatch measured BF16 square and projection shapes to Strassen."""
    if a.ndim != 2 or b.ndim != 2:
        return native_matmul(a, b)
    m, k = a.shape
    k2, n = b.shape
    shape = (m, k, n)
    square_supported = m == k == n and m >= TUNED_SQUARE_CROSSOVER
    projection_supported = (
        VMEM_PROFILE in ("spatial34", "max48")
        and shape in TUNED_PROJECTION_SHAPES
    )
    use_strassen = (
        k == k2
        and (square_supported or projection_supported)
        and a.dtype == jnp.bfloat16
        and b.dtype == jnp.bfloat16
        and m % TUNED_BM == 0
        and n % TUNED_BN == 0
        and k % TUNED_BK == 0
        and _on_tpu(interpret)
    )
    if not use_strassen:
        return native_matmul(a, b)
    interleave_products = (
        VMEM_PROFILE == "max48" and shape in TUNED_INTERLEAVED_SHAPES
    )
    return strassen_matmul(
        a,
        b,
        interleave_products=interleave_products,
        interpret=interpret,
        vmem_limit_bytes=None if interpret else TUNED_VMEM_LIMIT_BYTES,
    )
