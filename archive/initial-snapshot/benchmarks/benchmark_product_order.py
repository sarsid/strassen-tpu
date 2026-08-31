"""Test a dependency-spaced Strassen product order against BF16 controls."""

from __future__ import annotations

import argparse
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


OUTPUT = Path("/content/results/strassen_product_order.jsonl")
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
    print("PRODUCT_ORDER_JSON " + line, flush=True)


def compile_one(name, fn, a, b, phase, shape):
    started = time.perf_counter()
    try:
        executable = jax.jit(fn).lower(a, b).compile()
    except Exception as error:
        emit({
            "kind": "compile",
            "name": name,
            "phase": phase,
            "shape": shape,
            "success": False,
            "compile_s": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        return None
    emit({
        "kind": "compile",
        "name": name,
        "phase": phase,
        "shape": shape,
        "success": True,
        "compile_s": time.perf_counter() - started,
    })
    return executable


def strassen_function(*, interleave):
    return functools.partial(
        sp.strassen_matmul,
        bm=sp.TUNED_BM,
        bn=sp.TUNED_BN,
        bk=sp.TUNED_BK,
        interleave_products=interleave,
        vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
    )


def paired_interval(left_samples, right_samples):
    deltas = [
        right - left for left, right in zip(left_samples, right_samples)
    ]
    mean = statistics.fmean(deltas)
    half_width = 2.093 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, mean - half_width, mean + half_width


def accuracy_gate():
    a, b = bench.device_uniform_inputs(ACCURACY_SHAPE, 20260836)
    functions = {
        "native": bench.native_bf16_gemm,
        "reference": bench.high_precision_reference,
        "control": strassen_function(interleave=False),
        "interleaved": strassen_function(interleave=True),
    }
    executables = {
        name: compile_one(name, fn, a, b, "accuracy", ACCURACY_SHAPE)
        for name, fn in functions.items()
    }
    if any(executable is None for executable in executables.values()):
        emit({"kind": "accuracy", "passes": False, "reason": "compile"})
        return False
    outputs = {name: fn(a, b) for name, fn in executables.items()}
    jax.block_until_ready(tuple(outputs.values()))
    errors = {
        name: bench.device_error(output, outputs["reference"])
        for name, output in outputs.items() if name != "reference"
    }
    candidate = errors["interleaved"]
    native_ratio = candidate["rmse"] / max(errors["native"]["rmse"], 1e-30)
    control_ratio = candidate["rmse"] / max(errors["control"]["rmse"], 1e-30)
    passes = (
        all(math.isfinite(value) for value in candidate.values())
        and native_ratio <= 4.0
        and control_ratio <= 1.05
        and candidate["maxnorm_relative"] <= 0.01
    )
    emit({
        "kind": "accuracy",
        "shape": ACCURACY_SHAPE,
        "errors_vs_reference": errors,
        "candidate_native_rmse_ratio": native_ratio,
        "candidate_control_rmse_ratio": control_ratio,
        "passes": passes,
    })
    del outputs, executables, functions, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def performance_case(name, shape, *, warmups, runs):
    m, k, n = shape
    a = jnp.full((m, k), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((k, n), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    functions = {
        "native": sp.native_matmul,
        "control": strassen_function(interleave=False),
        "interleaved": strassen_function(interleave=True),
    }
    executables = {
        variant: compile_one(
            name + "_" + variant, fn, a, b, "performance", shape
        )
        for variant, fn in functions.items()
    }
    if any(executable is None for executable in executables.values()):
        emit({
            "kind": "performance", "name": name, "shape": shape,
            "passes_native": False, "improves_control": False,
            "correct": False, "reason": "compile",
        })
        return False, False, False
    timings, outputs = bench.interleaved_timings(
        executables, a, b, shape, warmups=warmups, runs=runs
    )
    candidate = timings["interleaved"]
    native_speedup = timings["native"]["mean_ms"] / candidate["mean_ms"]
    control_speedup = timings["control"]["mean_ms"] / candidate["mean_ms"]
    native_delta = paired_interval(
        timings["native"]["samples_ms"], candidate["samples_ms"]
    )
    control_delta = paired_interval(
        timings["control"]["samples_ms"], candidate["samples_ms"]
    )
    expected = -k / 8
    sentinels = {key: float(value[0, 0]) for key, value in outputs.items()}
    passes_native = native_speedup > 1.0 and native_delta[2] < 0.0
    improves_control = control_speedup > 1.0 and control_delta[2] < 0.0
    correct = all(value == expected for value in sentinels.values())
    emit({
        "kind": "performance",
        "name": name,
        "shape": shape,
        "contract": "BF16 inputs, FP32 accumulation, BF16 output",
        "timings": timings,
        "candidate_speedup_vs_native": native_speedup,
        "candidate_speedup_vs_control": control_speedup,
        "candidate_minus_native_95ci_ms": native_delta[1:],
        "candidate_minus_control_95ci_ms": control_delta[1:],
        "sentinels": sentinels,
        "expected_sentinel": expected,
        "passes_native": passes_native,
        "improves_control": improves_control,
        "correct": correct,
    })
    del outputs, executables, functions, a, b
    gc.collect()
    jax.clear_caches()
    return passes_native, improves_control, correct


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all")
    parser.add_argument("--warmups", type=int, default=bench.WARMUPS)
    parser.add_argument("--runs", type=int, default=bench.RUNS)
    parser.add_argument("--output", default=str(OUTPUT))
    args, _ = parser.parse_known_args()
    selected_names = (
        {name for name, _ in PERFORMANCE_CASES}
        if args.cases == "all"
        else set(args.cases.split(","))
    )
    unknown = selected_names - {name for name, _ in PERFORMANCE_CASES}
    if unknown:
        raise ValueError(f"unknown performance cases: {sorted(unknown)}")
    selected_cases = tuple(
        case for case in PERFORMANCE_CASES if case[0] in selected_names
    )
    OUTPUT = Path(args.output)

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
        "control_order": [1, 2, 3, 4, 5, 6, 7],
        "candidate_order": [4, 6, 5, 2, 7, 3, 1],
        "warmups": args.warmups,
        "runs": args.runs,
    })
    accuracy = accuracy_gate()
    if not accuracy:
        emit({"kind": "final", "success": False, "accuracy": False})
        raise RuntimeError("interleaved product-order accuracy/compile gate failed")
    performance = {
        name: performance_case(
            name, shape, warmups=args.warmups, runs=args.runs
        )
        for name, shape in selected_cases
    }
    emit({
        "kind": "final",
        "success": all(all(gates) for gates in performance.values()),
        "accuracy": accuracy,
        "performance": {
            name: {
                "passes_native": gates[0],
                "improves_control": gates[1],
                "correct": gates[2],
            }
            for name, gates in performance.items()
        },
    })


if __name__ == "__main__":
    main()
