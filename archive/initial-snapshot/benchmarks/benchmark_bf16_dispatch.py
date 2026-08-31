"""Requalify promoted BF16 dispatch directly against native BF16 GEMM."""

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

import benchmark_common as bench


OUTPUT = Path("/content/results/strassen_bf16_dispatch.jsonl")
ACCURACY_SHAPE = (2048, 14336, 2048)
PERFORMANCE_CASES = (
    ("square_8192", (8192, 8192, 8192)),
    ("paper_rectangle", (8192, 4096, 8192)),
    ("qkv_projection", (8192, 4096, 12288)),
    ("mlp_up", (8192, 4096, 28672)),
    ("mlp_down", (8192, 14336, 4096)),
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("BF16_DISPATCH_JSON " + line, flush=True)


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


def accuracy_gate():
    a, b = bench.device_uniform_inputs(ACCURACY_SHAPE, 20260833)
    native = compile_one(
        "native", bench.native_bf16_gemm, a, b, "accuracy", ACCURACY_SHAPE
    )
    reference = compile_one(
        "reference", bench.high_precision_reference,
        a, b, "accuracy", ACCURACY_SHAPE,
    )
    candidate = compile_one(
        "strassen",
        functools.partial(
            sp.strassen_matmul,
            bm=sp.TUNED_BM,
            bn=sp.TUNED_BN,
            bk=sp.TUNED_BK,
            vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
        ),
        a,
        b,
        "accuracy",
        ACCURACY_SHAPE,
    )
    outputs = {
        "native": native(a, b),
        "reference": reference(a, b),
        "strassen": candidate(a, b),
    }
    jax.block_until_ready(tuple(outputs.values()))
    native_error = bench.device_error(outputs["native"], outputs["reference"])
    candidate_error = bench.device_error(
        outputs["strassen"], outputs["reference"]
    )
    ratio = candidate_error["rmse"] / max(native_error["rmse"], 1e-30)
    passes = (
        all(math.isfinite(value) for value in candidate_error.values())
        and ratio <= 4.0
        and candidate_error["maxnorm_relative"] <= 0.01
    )
    emit({
        "kind": "accuracy",
        "shape": ACCURACY_SHAPE,
        "native_vs_reference": native_error,
        "strassen_vs_reference": candidate_error,
        "strassen_native_rmse_ratio": ratio,
        "passes": passes,
    })
    del outputs, candidate, reference, native, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def paired_delta_interval(native_samples, strassen_samples):
    deltas = [
        strassen - native
        for native, strassen in zip(native_samples, strassen_samples)
    ]
    mean = statistics.fmean(deltas)
    half_width = 2.093 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, mean - half_width, mean + half_width


def performance_case(name, shape):
    m, k, n = shape
    a = jnp.full((m, k), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((k, n), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    executables = {
        "native": compile_one(
            name + "_native", sp.native_matmul,
            a, b, "performance", shape,
        ),
        "strassen": compile_one(
            name + "_strassen", sp.tuned_matmul,
            a, b, "performance", shape,
        ),
    }
    timings, outputs = bench.interleaved_timings(executables, a, b, shape)
    speedup = timings["native"]["mean_ms"] / timings["strassen"]["mean_ms"]
    delta_mean, delta_low, delta_high = paired_delta_interval(
        timings["native"]["samples_ms"],
        timings["strassen"]["samples_ms"],
    )
    expected = -k / 8
    sentinels = {key: float(value[0, 0]) for key, value in outputs.items()}
    passes = (
        speedup > 1.0
        and delta_high < 0.0
        and all(value == expected for value in sentinels.values())
    )
    emit({
        "kind": "performance",
        "name": name,
        "shape": shape,
        "contract": "BF16 inputs, FP32 accumulation, BF16 output",
        "timings": timings,
        "speedup": speedup,
        "strassen_minus_native_mean_ms": delta_mean,
        "strassen_minus_native_95ci_ms": [delta_low, delta_high],
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
    })
    accuracy = accuracy_gate()
    if not accuracy:
        emit({"kind": "final", "success": False, "accuracy": False})
        raise RuntimeError("BF16 accuracy gate failed")
    performance = {
        name: performance_case(name, shape)
        for name, shape in PERFORMANCE_CASES
    }
    success = all(performance.values())
    emit({
        "kind": "final",
        "success": success,
        "accuracy": accuracy,
        "performance": performance,
    })
    if not success:
        raise RuntimeError(performance)


if __name__ == "__main__":
    main()
