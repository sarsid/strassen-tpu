"""Cubic Pallas control: split the Strassen speedup into substrate and algorithm.

The promoted comparison is hand-tuned Pallas Strassen against native XLA
``jnp.matmul``. Two things differ at once: the algorithm (seven versus eight
half-size products) and the substrate (custom Pallas tiling under the measured
scoped-VMEM profile). This control runs two cubic Pallas kernels under the
identical substrate as the promoted kernel:

- ``cubic_full``: one full-tile MXU product per K panel with a single
  persistent FP32 accumulator. The natural cubic Pallas kernel.
- ``cubic_blocked``: the same 2x2 quadrant decomposition, half-size MXU
  product granularity, and four persistent FP32 quadrant accumulators as
  Strassen, but the classical eight products with no operand pre-adds.

Interleaving native, cubic_full, cubic_blocked, and Strassen in one same-run
protocol attributes the promoted speedup: native -> cubic isolates the
substrate; cubic_blocked -> Strassen isolates the 8 -> 7 product saving at
identical granularity.
"""

from __future__ import annotations

import functools
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import strassen_pallas as sp

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import benchmark_common as bench


OUTPUT_ROOT = Path(os.environ.get("STRASSEN_OUTPUT_DIR", "/content/runs"))
OUTPUT = OUTPUT_ROOT / "strassen_cubic_control.jsonl"
ACCURACY_SHAPE = (2048, 14336, 2048)
PERFORMANCE_CASES = (
    ("square_8192", (8192, 8192, 8192)),
    ("paper_rectangle", (8192, 4096, 8192)),
    ("qkv_projection", (8192, 4096, 12288)),
    ("mlp_up", (8192, 4096, 28672)),
    ("mlp_down", (8192, 14336, 4096)),
)
CANDIDATES = ("cubic_full", "cubic_blocked", "strassen")


def _cubic_full_kernel(a_ref, b_ref, o_ref, acc_ref, *, nk, dot_precision):
    step = pl.program_id(2)

    @pl.when(step == 0)
    def _zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

    acc_ref[...] += jnp.dot(
        a_ref[...],
        b_ref[...],
        preferred_element_type=jnp.float32,
        precision=dot_precision,
    )

    @pl.when(step == nk - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _cubic_blocked_kernel(a_ref, b_ref, o_ref, acc_ref, *, nk, dot_precision):
    step = pl.program_id(2)
    m, k = a_ref.shape
    _, n = b_ref.shape
    hm, hk, hn = m // 2, k // 2, n // 2

    a0, a1 = a_ref[:hm, :hk], a_ref[:hm, hk:]
    a2, a3 = a_ref[hm:, :hk], a_ref[hm:, hk:]
    b0, b1 = b_ref[:hk, :hn], b_ref[:hk, hn:]
    b2, b3 = b_ref[hk:, :hn], b_ref[hk:, hn:]

    dot = functools.partial(
        jnp.dot,
        preferred_element_type=jnp.float32,
        precision=dot_precision,
    )

    @pl.when(step == 0)
    def _zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

    # Eight products ordered so consecutive updates target distinct quadrant
    # accumulators, matching the dependency spacing available to Strassen.
    acc_ref[0] += dot(a0, b0)
    acc_ref[1] += dot(a0, b1)
    acc_ref[2] += dot(a2, b0)
    acc_ref[3] += dot(a2, b1)
    acc_ref[0] += dot(a1, b2)
    acc_ref[1] += dot(a1, b3)
    acc_ref[2] += dot(a3, b2)
    acc_ref[3] += dot(a3, b3)

    @pl.when(step == nk - 1)
    def _store():
        o_ref[:hm, :hn] = acc_ref[0].astype(o_ref.dtype)
        o_ref[:hm, hn:] = acc_ref[1].astype(o_ref.dtype)
        o_ref[hm:, :hn] = acc_ref[2].astype(o_ref.dtype)
        o_ref[hm:, hn:] = acc_ref[3].astype(o_ref.dtype)


def cubic_matmul(
    a,
    b,
    *,
    variant,
    bm=sp.TUNED_BM,
    bn=sp.TUNED_BN,
    bk=sp.TUNED_BK,
    dot_precision=jax.lax.Precision.DEFAULT,
    allow_input_fusion=None,
    interpret=False,
    vmem_limit_bytes=None,
):
    """Cubic Pallas GEMM matching ``strassen_matmul``'s call structure."""
    if variant not in ("full", "blocked"):
        raise ValueError(f"unknown cubic variant {variant!r}")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions disagree: {k} vs {k2}")
    if allow_input_fusion is not None:
        allow_input_fusion = tuple(allow_input_fusion)
        if len(allow_input_fusion) != 2:
            raise ValueError(
                "allow_input_fusion must contain one boolean per operand"
            )
        if not all(isinstance(value, bool) for value in allow_input_fusion):
            raise TypeError("allow_input_fusion entries must be bool")
    for name, dimension, block in (("M", m, bm), ("N", n, bn), ("K", k, bk)):
        if dimension % block:
            raise ValueError(f"{name}={dimension} is not divisible by {block}")
    sp.check_tiling(bm, bn, bk)
    nk = k // bk

    if variant == "full":
        kernel = functools.partial(
            _cubic_full_kernel, nk=nk, dot_precision=dot_precision
        )
        scratch_shape = pltpu.VMEM((bm, bn), jnp.float32)
    else:
        kernel = functools.partial(
            _cubic_blocked_kernel, nk=nk, dot_precision=dot_precision
        )
        scratch_shape = pltpu.VMEM((4, bm // 2, bn // 2), jnp.float32)

    return pl.pallas_call(
        kernel,
        grid=(m // bm, n // bn, nk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, step: (i, step)),
            pl.BlockSpec((bk, bn), lambda i, j, step: (step, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, step: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[scratch_shape],
        compiler_params=sp.TPU_COMPILER_PARAMS(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=allow_input_fusion,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(a, b)


def cubic_matmul_lhs_transposed(
    a,
    b,
    *,
    bm=sp.TUNED_BM,
    bn=sp.TUNED_BN,
    bk=sp.TUNED_BK,
    dot_precision=jax.lax.Precision.DEFAULT,
    allow_input_fusion=None,
    interpret=False,
    vmem_limit_bytes=None,
):
    """Matched blocked-cubic control for ``a.T @ b``.

    Like ``sp.strassen_matmul_lhs_transposed``, this preserves the source
    ``(K, M)`` layout in HBM and transposes only the loaded VMEM tile.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cubic_matmul_lhs_transposed expects rank-2 inputs")
    k, m = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"contracting dimensions disagree: {k} vs {k2}")
    if a.dtype != b.dtype:
        raise ValueError(f"input dtypes disagree: {a.dtype} vs {b.dtype}")
    for name, dimension, block in (("M", m, bm), ("N", n, bn), ("K", k, bk)):
        if dimension % block:
            raise ValueError(f"{name}={dimension} is not divisible by {block}")
    sp.check_tiling(bm, bn, bk)
    if allow_input_fusion is not None:
        allow_input_fusion = tuple(allow_input_fusion)
        if len(allow_input_fusion) != 2:
            raise ValueError("allow_input_fusion must contain two booleans")
    nk = k // bk

    def kernel(a_physical_ref, b_ref, o_ref, acc_ref):
        a_tile = jnp.swapaxes(a_physical_ref[...], 0, 1)
        return _cubic_blocked_kernel(
            a_tile, b_ref, o_ref, acc_ref,
            nk=nk, dot_precision=dot_precision)

    return pl.pallas_call(
        kernel,
        grid=(m // bm, n // bn, nk),
        in_specs=[
            pl.BlockSpec((bk, bm), lambda i, j, step: (step, i)),
            pl.BlockSpec((bk, bn), lambda i, j, step: (step, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, step: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[pltpu.VMEM((4, bm // 2, bn // 2), jnp.float32)],
        compiler_params=sp.TPU_COMPILER_PARAMS(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=allow_input_fusion,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(a, b)


# Both cubic controls report 47.89 MiB compiler demand at the max48 tile,
# above the 47 MiB promoted Strassen kernel cap (Strassen's own demand was
# 46.38 MiB), so they receive the full 48 MiB scoped ceiling. A candidate that
# still fails to compile is recorded as an OOM and skipped.
CUBIC_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def make_executable_fns(*, promoted_strassen):
    limit = sp.TUNED_VMEM_LIMIT_BYTES
    # Performance cases go through tuned_matmul so the Strassen arm carries the
    # exact promoted policy (including per-shape product-order selection); the
    # accuracy gate calls strassen_matmul directly because its long-K shape is
    # deliberately outside the dispatcher.
    strassen = (
        sp.tuned_matmul
        if promoted_strassen
        else functools.partial(sp.strassen_matmul, vmem_limit_bytes=limit)
    )
    return {
        "native": sp.native_matmul,
        "cubic_full": functools.partial(
            cubic_matmul,
            variant="full",
            vmem_limit_bytes=CUBIC_VMEM_LIMIT_BYTES,
        ),
        "cubic_blocked": functools.partial(
            cubic_matmul,
            variant="blocked",
            vmem_limit_bytes=CUBIC_VMEM_LIMIT_BYTES,
        ),
        "strassen": strassen,
    }


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CUBIC_CONTROL_JSON " + line, flush=True)


def compile_one(name, fn, a, b, phase, shape):
    started = time.perf_counter()
    executable = jax.jit(fn).lower(a, b).compile()
    emit({
        "kind": "compile",
        "name": name,
        "phase": phase,
        "shape": shape,
        "compile_s": time.perf_counter() - started,
    })
    return executable


def try_compile(name, fn, a, b, phase, shape):
    """Compile a candidate, recording rather than raising a VMEM OOM."""
    try:
        return compile_one(name, fn, a, b, phase, shape)
    except Exception as error:  # XlaRuntimeError has no stable import path.
        message = str(error)
        if "RESOURCE_EXHAUSTED" not in message:
            raise
        emit({
            "kind": "compile_oom",
            "name": name,
            "phase": phase,
            "shape": shape,
            "error": message.splitlines()[0][:500],
        })
        return None


def paired_delta_interval(baseline_samples, candidate_samples):
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_samples, candidate_samples)
    ]
    mean = statistics.fmean(deltas)
    half_width = 2.093 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, mean - half_width, mean + half_width


def accuracy_gate():
    fns = make_executable_fns(promoted_strassen=False)
    a, b = bench.device_uniform_inputs(ACCURACY_SHAPE, 20260833)
    reference = compile_one(
        "reference", bench.high_precision_reference,
        a, b, "accuracy", ACCURACY_SHAPE,
    )
    outputs = {"reference": reference(a, b)}
    for name, fn in fns.items():
        executable = try_compile(name, fn, a, b, "accuracy", ACCURACY_SHAPE)
        if executable is not None:
            outputs[name] = executable(a, b)
    jax.block_until_ready(tuple(outputs.values()))

    available = [name for name in fns if name in outputs]
    errors = {
        name: bench.device_error(outputs[name], outputs["reference"])
        for name in available
    }
    native_rmse = max(errors["native"]["rmse"], 1e-30)
    ratios = {
        name: errors[name]["rmse"] / native_rmse
        for name in CANDIDATES
        if name in errors
    }
    # The cubic controls compute the same classical GEMM as native and must
    # land near its rounding profile; Strassen keeps its known 4x bound.
    # cubic_blocked is the required substrate control; cubic_full may drop out
    # with a recorded OOM.
    cubic_bounds = all(
        ratios[name] <= 1.25
        for name in ("cubic_full", "cubic_blocked")
        if name in ratios
    )
    passes = (
        "native" in errors
        and "cubic_blocked" in errors
        and "strassen" in errors
        and all(
            math.isfinite(value)
            for error in errors.values()
            for value in error.values()
        )
        and cubic_bounds
        and ratios["strassen"] <= 4.0
        and all(
            errors[name]["maxnorm_relative"] <= 0.01
            for name in CANDIDATES
            if name in errors
        )
    )
    emit({
        "kind": "accuracy",
        "shape": ACCURACY_SHAPE,
        "available": available,
        "errors_vs_reference": errors,
        "rmse_ratio_vs_native": ratios,
        "passes": passes,
    })
    del outputs, reference, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def timed_shape_correctness():
    """Full-tensor correctness for the five TIMED shapes, on random inputs.

    The performance cases feed constant matrices and validate only
    `output[0, 0]`, which cannot expose a quadrant, layout, or index error
    because every output element is identical. The accuracy gate does use
    random inputs, but at one shape and -- for the Strassen arm -- through
    `strassen_matmul` directly, while the timed shapes go through
    `tuned_matmul` and its per-shape product-order selection. This closes that
    gap: every timed shape, through the promoted dispatcher, compared
    full-tensor against a high-precision reference.
    """
    fns = make_executable_fns(promoted_strassen=True)
    verdicts = {}
    for case_name, shape in PERFORMANCE_CASES:
        label = case_name
        a, b = bench.device_uniform_inputs(shape, 20260901)
        reference = compile_one(
            f"{label}_reference", bench.high_precision_reference,
            a, b, "timed_shape_correctness", shape,
        )
        outputs = {"reference": reference(a, b)}
        for name, fn in fns.items():
            executable = try_compile(
                f"{label}_{name}", fn, a, b, "timed_shape_correctness", shape
            )
            if executable is not None:
                outputs[name] = executable(a, b)
        jax.block_until_ready(tuple(outputs.values()))

        errors = {
            name: bench.device_error(outputs[name], outputs["reference"])
            for name in outputs if name != "reference"
        }
        native_rmse = max(errors["native"]["rmse"], 1e-30)
        ratios = {
            name: errors[name]["rmse"] / native_rmse
            for name in errors if name != "native"
        }
        # Cubic controls must reproduce native's rounding profile; a quadrant
        # or index bug shows up here as a ratio far from 1.0.
        cubic_ok = all(
            ratios[name] <= 1.25
            for name in ("cubic_full", "cubic_blocked") if name in ratios
        )
        verdicts[case_name] = (
            cubic_ok
            and ratios.get("strassen", 0.0) <= 4.0
            and all(
                math.isfinite(value)
                for error in errors.values() for value in error.values()
            )
        )
        emit({
            "kind": "timed_shape_correctness",
            "case": case_name,
            "shape": list(shape),
            "errors_vs_reference": errors,
            "rmse_ratio_vs_native": ratios,
            "passes": verdicts[case_name],
        })
        del outputs, reference, a, b
        gc.collect()
        jax.clear_caches()
    return verdicts


def performance_case(name, shape):
    fns = make_executable_fns(promoted_strassen=True)
    m, k, n = shape
    a = jnp.full((m, k), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((k, n), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    executables = {}
    for label, fn in fns.items():
        executable = try_compile(
            f"{name}_{label}", fn, a, b, "performance", shape
        )
        if executable is not None:
            executables[label] = executable
    timings, outputs = bench.interleaved_timings(executables, a, b, shape)

    speedups = {
        label: timings["native"]["mean_ms"] / timings[label]["mean_ms"]
        for label in CANDIDATES
        if label in timings
    }
    paired = {}
    for baseline, candidate in (
        ("native", "cubic_full"),
        ("native", "cubic_blocked"),
        ("native", "strassen"),
        ("cubic_full", "strassen"),
        ("cubic_blocked", "strassen"),
        ("cubic_full", "cubic_blocked"),
    ):
        if baseline not in timings or candidate not in timings:
            continue
        mean, low, high = paired_delta_interval(
            timings[baseline]["samples_ms"],
            timings[candidate]["samples_ms"],
        )
        paired[f"{candidate}_minus_{baseline}"] = {
            "mean_ms": mean,
            "ci95_ms": [low, high],
        }

    expected = -k / 8
    sentinels = {key: float(value[0, 0]) for key, value in outputs.items()}
    passes = (
        {"native", "cubic_blocked", "strassen"} <= set(executables)
        and all(value == expected for value in sentinels.values())
    )
    emit({
        "kind": "performance",
        "name": name,
        "shape": shape,
        "contract": "BF16 inputs, FP32 accumulation, BF16 output",
        "timings": timings,
        "speedup_vs_native": speedups,
        "paired_deltas": paired,
        "sentinels": sentinels,
        "expected_sentinel": expected,
        "passes": passes,
    })
    del outputs, executables, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "profile": sp.VMEM_PROFILE,
        "tile": [sp.TUNED_BM, sp.TUNED_BN, sp.TUNED_BK],
        "contract": "BF16 inputs, FP32 accumulation, BF16 output",
        "warmups": bench.WARMUPS,
        "runs": bench.RUNS,
        "candidates": list(CANDIDATES),
        "purpose": (
            "attribute promoted Strassen speedup between Pallas substrate "
            "and the 8->7 product saving"
        ),
    })
    accuracy = accuracy_gate()
    if not accuracy:
        emit({"kind": "final", "success": False, "accuracy": False})
        raise RuntimeError("cubic control accuracy gate failed")
    shape_correctness = timed_shape_correctness()
    if not all(shape_correctness.values()):
        emit({
            "kind": "final", "success": False, "accuracy": accuracy,
            "timed_shape_correctness": shape_correctness,
        })
        raise RuntimeError(f"timed-shape correctness failed: {shape_correctness}")
    performance = {
        name: performance_case(name, shape)
        for name, shape in PERFORMANCE_CASES
    }
    success = all(performance.values())
    emit({
        "kind": "final",
        "success": success,
        "accuracy": accuracy,
        "timed_shape_correctness": shape_correctness,
        "performance": performance,
    })
    if not success:
        raise RuntimeError(performance)


if __name__ == "__main__":
    main()
