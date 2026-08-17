"""Gated datatype screen for the one-level classical Strassen TPU kernel.

This benchmark treats a GEMM datatype as a complete numerical contract:
input, Strassen-sum, dot precision, accumulator, and output dtype.  Every
candidate is compared with an XLA GEMM having the same public input/output and
dot-precision contract.  A separate FP32 Precision.HIGHEST GEMM is retained as
the accuracy reference.

Run in a fresh one-device v5e kernel. The benchmark defaults to the measured
48 MiB profile and tests only the retained datatype candidates.
"""

from __future__ import annotations

import functools
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("STRASSEN_VMEM_PROFILE", "max48")
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as bench


MIB = 1024 * 1024
ACCURACY_SHAPE = (2048, 2048, 2048)
PERFORMANCE_SIZES = (4096, 8192)
OUTPUT = Path("/content/results/strassen_datatype_screen.jsonl")


MODES = {
    "bf16_bf16": {
        "input_dtype": jnp.bfloat16,
        "output_dtype": jnp.bfloat16,
        "precision": jax.lax.Precision.DEFAULT,
        "preferred_element_type": jnp.float32,
        "variants": {
            "square2048": {
                "tile": (2048, 2048, 512),
                "limit_mib": 47,
            },
        },
    },
    "bf16_f32": {
        "input_dtype": jnp.bfloat16,
        "output_dtype": jnp.float32,
        "precision": jax.lax.Precision.DEFAULT,
        "preferred_element_type": jnp.float32,
        "variants": {
            "compact": {
                "tile": (1024, 1024, 512),
                "limit_mib": 32,
            },
            "spatial_n": {
                "tile": (1024, 2048, 512),
                "limit_mib": 47,
            },
        },
    },
    "fp32_default": {
        "input_dtype": jnp.float32,
        "output_dtype": jnp.float32,
        "precision": jax.lax.Precision.DEFAULT,
        "preferred_element_type": None,
        "variants": {
            "compact": {
                "tile": (1024, 1024, 512),
                "limit_mib": 40,
            },
        },
    },
    "fp32_highest": {
        "input_dtype": jnp.float32,
        "output_dtype": jnp.float32,
        "precision": jax.lax.Precision.HIGHEST,
        "preferred_element_type": None,
        "variants": {
            "compact": {
                "tile": (1024, 1024, 512),
                "limit_mib": 40,
            },
        },
    },
}


def jsonable_mode(mode):
    return {
        "input_dtype": str(jnp.dtype(mode["input_dtype"])),
        "output_dtype": str(jnp.dtype(mode["output_dtype"])),
        "precision": mode["precision"].name,
        "preferred_element_type": (
            None
            if mode["preferred_element_type"] is None
            else str(jnp.dtype(mode["preferred_element_type"]))
        ),
        "variants": mode["variants"],
    }


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("DTYPE_SCREEN_JSON " + line, flush=True)


def native_function(mode):
    def gemm(a, b):
        out = jnp.matmul(
            a,
            b,
            precision=mode["precision"],
            preferred_element_type=mode["preferred_element_type"],
        )
        return out.astype(mode["output_dtype"])

    return gemm


def reference_function(a, b):
    return jnp.matmul(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )


def candidate_function(mode, variant):
    bm, bn, bk = variant["tile"]
    return functools.partial(
        sp.strassen_matmul,
        bm=bm,
        bn=bn,
        bk=bk,
        dot_precision=mode["precision"],
        output_dtype=mode["output_dtype"],
        interpret=False,
        vmem_limit_bytes=variant["limit_mib"] * MIB,
    )


def compile_one(fn, a, b, *, phase, mode_name, name, shape, config=None):
    started = time.perf_counter()
    try:
        executable = jax.jit(fn).lower(a, b).compile()
    except Exception as error:
        emit({
            "kind": "compile",
            "phase": phase,
            "mode": mode_name,
            "name": name,
            "shape": shape,
            "config": config,
            "success": False,
            "compile_s": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        return None
    emit({
        "kind": "compile",
        "phase": phase,
        "mode": mode_name,
        "name": name,
        "shape": shape,
        "config": config,
        "success": True,
        "compile_s": time.perf_counter() - started,
    })
    return executable


def inputs(shape, dtype, *, random, seed=20260815):
    m, k, n = shape
    if random:
        rng = np.random.default_rng(seed)
        a_host = rng.uniform(-1.0, 1.0, (m, k)).astype(np.float32)
        b_host = rng.uniform(-1.0, 1.0, (k, n)).astype(np.float32)
        a = jnp.asarray(a_host, dtype=dtype)
        b = jnp.asarray(b_host, dtype=dtype)
        del a_host, b_host
    else:
        a = jnp.full((m, k), 0.5, dtype=dtype)
        b = jnp.full((k, n), -0.25, dtype=dtype)
    jax.block_until_ready((a, b))
    return a, b


def compile_mode(mode_name, mode, a, b, phase, shape, reference):
    executables = {}
    native = compile_one(
        native_function(mode), a, b,
        phase=phase, mode_name=mode_name, name="native", shape=shape,
    )
    if native is not None:
        executables["native"] = native
    if reference:
        reference_executable = compile_one(
            reference_function, a, b,
            phase=phase, mode_name=mode_name, name="reference", shape=shape,
        )
        if reference_executable is not None:
            executables["reference"] = reference_executable
    for variant_name, variant in mode["variants"].items():
        executable = compile_one(
            candidate_function(mode, variant), a, b,
            phase=phase,
            mode_name=mode_name,
            name=variant_name,
            shape=shape,
            config=variant,
        )
        if executable is not None:
            executables[variant_name] = executable
    return executables


def run_accuracy_mode(mode_name, mode):
    a, b = inputs(ACCURACY_SHAPE, mode["input_dtype"], random=True)
    executables = compile_mode(
        mode_name, mode, a, b, "accuracy", ACCURACY_SHAPE, reference=True
    )
    required = {"native", "reference"}
    if not required.issubset(executables):
        emit({
            "kind": "accuracy_gates",
            "mode": mode_name,
            "passes": {name: False for name in mode["variants"]},
            "reason": "native/reference compile failure",
        })
        del executables, a, b
        gc.collect()
        jax.clear_caches()
        return {name: False for name in mode["variants"]}

    outputs = {name: executable(a, b) for name, executable in executables.items()}
    jax.block_until_ready(tuple(outputs.values()))
    native_error = bench.device_error(outputs["native"], outputs["reference"])
    passes = {}
    for variant_name in mode["variants"]:
        if variant_name not in outputs:
            passes[variant_name] = False
            continue
        reference_error = bench.device_error(
            outputs[variant_name], outputs["reference"]
        )
        direct_error = bench.device_error(
            outputs[variant_name], outputs["native"]
        )
        native_rmse = native_error["rmse"]
        ratio = (
            reference_error["rmse"] / native_rmse
            if native_rmse > 1e-30 else None
        )
        finite = all(
            math.isfinite(value)
            for error in (reference_error, direct_error)
            for value in error.values()
        )
        if ratio is None:
            passes[variant_name] = (
                finite and direct_error["maxnorm_relative"] <= 0.01
            )
        else:
            passes[variant_name] = (
                finite
                and ratio <= 4.0
                and reference_error["maxnorm_relative"] <= 0.01
            )
        emit({
            "kind": "accuracy",
            "mode": mode_name,
            "name": variant_name,
            "shape": ACCURACY_SHAPE,
            "distribution": "uniform",
            "seed": 20260815,
            "finite": finite,
            "native_vs_reference": native_error,
            "candidate_vs_reference": reference_error,
            "candidate_vs_native": direct_error,
            "candidate_native_rmse_ratio": ratio,
            "passes": passes[variant_name],
        })
    emit({"kind": "accuracy_gates", "mode": mode_name, "passes": passes})
    del outputs, executables, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def run_performance_mode(mode_name, mode, size, allowed):
    shape = (size, size, size)
    a, b = inputs(shape, mode["input_dtype"], random=False)
    executables = compile_mode(
        mode_name, mode, a, b, "performance", shape, reference=False
    )
    executables = {
        name: executable
        for name, executable in executables.items()
        if name == "native" or allowed.get(name, False)
    }
    if "native" not in executables or len(executables) == 1:
        emit({
            "kind": "performance",
            "mode": mode_name,
            "shape": shape,
            "success": False,
            "reason": "no compiling accuracy-qualified candidate",
        })
        del executables, a, b
        gc.collect()
        jax.clear_caches()
        return
    timings, outputs = bench.interleaved_timings(
        executables, a, b, (size, size, size)
    )
    speedups = {
        name: timings["native"]["mean_ms"] / timing["mean_ms"]
        for name, timing in timings.items()
        if name != "native"
    }
    emit({
        "kind": "performance",
        "mode": mode_name,
        "shape": shape,
        "success": True,
        "timings": timings,
        "speedups_vs_native": speedups,
        "sentinels": {name: float(value[0, 0]) for name, value in outputs.items()},
    })
    del outputs, executables, a, b
    gc.collect()
    jax.clear_caches()


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "experiment": "datatype_screen",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "accuracy_shape": ACCURACY_SHAPE,
        "performance_sizes": PERFORMANCE_SIZES,
        "warmups": bench.WARMUPS,
        "runs": bench.RUNS,
        "modes": {name: jsonable_mode(mode) for name, mode in MODES.items()},
    })
    gates = {
        mode_name: run_accuracy_mode(mode_name, mode)
        for mode_name, mode in MODES.items()
    }
    for size in PERFORMANCE_SIZES:
        for mode_name, mode in MODES.items():
            run_performance_mode(mode_name, mode, size, gates[mode_name])
    emit({"kind": "final", "success": True, "accuracy_gates": gates})


if __name__ == "__main__":
    main()
