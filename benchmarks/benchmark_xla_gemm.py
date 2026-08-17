"""Reproducible native XLA/JAX GEMM baselines for a single TPU device.

The primary comparison target for ``strassen_pallas.py`` is
``bf16_f32acc_bf16out``: BF16 inputs, an explicitly requested FP32 dot result,
and a final BF16 cast.  This matches the current Strassen kernel's public
input/output contract while leaving the cubic GEMM entirely to XLA.

This module intentionally contains no Pallas code and no Strassen code.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import jaxlib


HARNESS_VERSION = 1
PAPER_PROTOCOL_WARMUPS = 10
PAPER_PROTOCOL_RUNS = 20

# Figure 8 evaluates equal-dimension GEMMs at 512-element intervals.  The 8K
# sweep is the first pinned TPU set; the full paper sweep continues to 16K.
SUITES = {
    "smoke": tuple((n, n, n) for n in (512, 1024)),
    "anchor_square": tuple((n, n, n) for n in (512, 1024, 2048, 4096, 8192)),
    "paper_square_8k": tuple((n, n, n) for n in range(512, 8192 + 1, 512)),
    "paper_square_full": tuple((n, n, n) for n in range(512, 16384 + 1, 512)),
}


@dataclass(frozen=True)
class Mode:
    input_dtype: object
    precision: jax.lax.Precision
    preferred_element_type: object | None = None
    output_dtype: object | None = None
    description: str = ""


MODES = {
    "bf16_native": Mode(
        jnp.bfloat16,
        jax.lax.Precision.DEFAULT,
        description="Stock BF16 jnp.matmul contract.",
    ),
    "bf16_f32acc_bf16out": Mode(
        jnp.bfloat16,
        jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
        output_dtype=jnp.bfloat16,
        description="BF16 inputs, explicit FP32 dot result, BF16 output cast.",
    ),
    "fp32_default": Mode(
        jnp.float32,
        jax.lax.Precision.DEFAULT,
        description="FP32 inputs with XLA Precision.DEFAULT.",
    ),
    "fp32_high": Mode(
        jnp.float32,
        jax.lax.Precision.HIGH,
        description="FP32 inputs with XLA Precision.HIGH.",
    ),
    "fp32_highest": Mode(
        jnp.float32,
        jax.lax.Precision.HIGHEST,
        description="FP32 inputs with XLA Precision.HIGHEST.",
    ),
}


def _gemm(mode: Mode):
    def gemm(a, b):
        out = jnp.matmul(
            a,
            b,
            precision=mode.precision,
            preferred_element_type=mode.preferred_element_type,
        )
        if mode.output_dtype is not None:
            out = out.astype(mode.output_dtype)
        return out

    return gemm


def _metadata(suite: str, mode_names: Sequence[str], warmups: int, runs: int):
    devices = jax.devices()
    device = devices[0]
    return {
        "kind": "metadata",
        "schema_version": 1,
        "harness_version": HARNESS_VERSION,
        "operation": "dense_2d_gemm",
        "implementation": "jax.jit(jnp.matmul)",
        "suite": suite,
        "modes": list(mode_names),
        "warmups": warmups,
        "runs": runs,
        "primary_statistic": "mean_ms",
        "timed_region": "compiled executable call plus block_until_ready",
        "compile_excluded": True,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": device.device_kind,
        "device_platform": device.platform,
        "device_platform_version": getattr(device, "platform_version", ""),
    }


def benchmark_one(
    m: int,
    k: int,
    n: int,
    mode_name: str,
    *,
    warmups: int = PAPER_PROTOCOL_WARMUPS,
    runs: int = PAPER_PROTOCOL_RUNS,
):
    mode = MODES[mode_name]

    # Inputs are function arguments, so the compiler cannot constant-fold the
    # GEMM even though deterministic fill values make setup cheap.
    a = jnp.full((m, k), 0.5, dtype=mode.input_dtype)
    b = jnp.full((k, n), -0.25, dtype=mode.input_dtype)
    jax.block_until_ready((a, b))

    lowered = jax.jit(_gemm(mode)).lower(a, b)
    compile_start = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - compile_start

    for _ in range(warmups):
        jax.block_until_ready(executable(a, b))

    samples_ms = []
    out = None
    for _ in range(runs):
        start = time.perf_counter_ns()
        out = executable(a, b)
        jax.block_until_ready(out)
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    assert out is not None
    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    std_ms = statistics.pstdev(samples_ms)
    flop_count = 2 * m * n * k

    # This scalar transfer is outside the timed region and serves as a simple
    # execution sanity check.  Expected value is -K/8 for the fill values.
    sample_c00 = float(out[0, 0])

    result = {
        "kind": "result",
        "operation": "dense_2d_gemm",
        "implementation": "xla_jax_matmul",
        "mode": mode_name,
        "mode_description": mode.description,
        "m": m,
        "k": k,
        "n": n,
        "input_dtype": str(jnp.dtype(mode.input_dtype)),
        "output_dtype": str(out.dtype),
        "precision": mode.precision.name,
        "preferred_element_type": (
            None
            if mode.preferred_element_type is None
            else str(jnp.dtype(mode.preferred_element_type))
        ),
        "warmups": warmups,
        "runs": runs,
        "compile_s": compile_s,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "mean_tflops": flop_count / (mean_ms / 1e3) / 1e12,
        "peak_sample_tflops": flop_count / (min_ms / 1e3) / 1e12,
        "sample_c00": sample_c00,
        "expected_c00": -k / 8,
        "samples_ms": samples_ms,
    }

    del out, executable, lowered, a, b
    gc.collect()
    jax.clear_caches()
    return result


def run_suite(
    suite: str = "smoke",
    modes: Iterable[str] = ("bf16_native", "bf16_f32acc_bf16out"),
    *,
    warmups: int = PAPER_PROTOCOL_WARMUPS,
    runs: int = PAPER_PROTOCOL_RUNS,
    output_path: str | None = None,
    emit_json: bool = True,
):
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
    mode_names = tuple(modes)
    unknown_modes = sorted(set(mode_names) - set(MODES))
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}; choose from {sorted(MODES)}")
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"baseline must run on TPU, got {jax.default_backend()!r}")
    if len(jax.devices()) != 1:
        raise RuntimeError(f"single-device baseline expected, got {jax.devices()}")

    metadata = _metadata(suite, mode_names, warmups, runs)
    output = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open("w", encoding="utf-8")

    def emit(record):
        line = json.dumps(record, sort_keys=True)
        if emit_json:
            print("BASELINE_JSON " + line, flush=True)
        if output is not None:
            output.write(line + "\n")
            output.flush()

    emit(metadata)

    results = []
    try:
        for mode_name in mode_names:
            for m, k, n in SUITES[suite]:
                result = benchmark_one(
                    m, k, n, mode_name, warmups=warmups, runs=runs
                )
                results.append(result)
                emit(result)
                print(
                    f"{mode_name:<25} {m:>5}x{k:<5}x{n:<5} "
                    f"{result['mean_ms']:>9.3f} ms  "
                    f"{result['mean_tflops']:>8.2f} TFLOP/s",
                    flush=True,
                )
    finally:
        if output is not None:
            output.close()
    return metadata, results


if __name__ == "__main__":
    run_suite()
