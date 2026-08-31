"""Paper-aligned and common-workload rectangular GEMM screen on one TPU v5e.

The suite advances only the two meaningful datatype contracts from
benchmark_datatype_screen.py:

* BF16 inputs, FP32 accumulation, BF16 output.
* FP32 inputs/output with Precision.HIGHEST.

It covers a paper-style rectangular GEMM, Phi-4-like QKV and MLP projections,
and small-token decode boundaries.  Large shapes compare both spatial tile
orientations because the square-optimal tile is not assumed to transfer.
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

import benchmark_common as bench


MIB = 1024 * 1024
OUTPUT = Path("/content/results/strassen_rectangular_workloads.jsonl")


MODES = {
    "bf16_bf16": {
        "input_dtype": jnp.bfloat16,
        "output_dtype": jnp.bfloat16,
        "precision": jax.lax.Precision.DEFAULT,
        "preferred_element_type": jnp.float32,
    },
    "fp32_highest": {
        "input_dtype": jnp.float32,
        "output_dtype": jnp.float32,
        "precision": jax.lax.Precision.HIGHEST,
        "preferred_element_type": None,
    },
}


BF16_LARGE_VARIANTS = {
    "compact": {"tile": (1024, 1024, 512), "limit_mib": 16},
    "spatial_n": {"tile": (1024, 2048, 512), "limit_mib": 32},
    "spatial_m": {"tile": (2048, 1024, 512), "limit_mib": 32},
    "square2048": {"tile": (2048, 2048, 512), "limit_mib": 47},
}

FP32_LARGE_VARIANTS = {
    "compact": {"tile": (1024, 1024, 512), "limit_mib": 40},
}

LARGE_WORKLOADS = (
    ("paper_rect_k4096", (8192, 4096, 8192)),
    ("phi4_qkv_b8_s1024", (8192, 4096, 12288)),
    ("phi4_mlp_up_b8_s1024", (8192, 4096, 28672)),
    ("phi4_mlp_down_b8_s1024", (8192, 14336, 4096)),
)

DECODE_WORKLOADS = (
    (
        "phi4_decode_32_tokens",
        (32, 4096, 14336),
        {"decode": {"tile": (32, 1024, 512), "limit_mib": 16}},
    ),
    (
        "phi4_decode_256_tokens",
        (256, 4096, 14336),
        {"decode": {"tile": (256, 1024, 512), "limit_mib": 16}},
    ),
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("RECTANGULAR_JSON " + line, flush=True)


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


def host_inputs(shape, dtype, seed):
    m, k, n = shape
    a_host, b_host = bench.host_uniform_inputs(shape, seed)
    a = jnp.asarray(a_host, dtype=dtype)
    b = jnp.asarray(b_host, dtype=dtype)
    del a_host, b_host
    jax.block_until_ready((a, b))
    return a, b


def constant_inputs(shape, dtype):
    m, k, n = shape
    a = jnp.full((m, k), 0.5, dtype=dtype)
    b = jnp.full((k, n), -0.25, dtype=dtype)
    jax.block_until_ready((a, b))
    return a, b


def compile_one(fn, a, b, *, phase, workload, mode_name, name, shape,
                config=None):
    started = time.perf_counter()
    try:
        executable = jax.jit(fn).lower(a, b).compile()
    except Exception as error:
        emit({
            "kind": "compile",
            "phase": phase,
            "workload": workload,
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
        "workload": workload,
        "mode": mode_name,
        "name": name,
        "shape": shape,
        "config": config,
        "success": True,
        "compile_s": time.perf_counter() - started,
    })
    return executable


def compile_set(a, b, *, phase, workload, shape, mode_name, mode, variants,
                reference):
    executables = {}
    functions = {"native": native_function(mode)}
    if reference:
        functions["reference"] = reference_function
    for name, fn in functions.items():
        executable = compile_one(
            fn, a, b, phase=phase, workload=workload, mode_name=mode_name,
            name=name, shape=shape,
        )
        if executable is not None:
            executables[name] = executable
    for name, variant in variants.items():
        executable = compile_one(
            candidate_function(mode, variant), a, b,
            phase=phase, workload=workload, mode_name=mode_name,
            name=name, shape=shape, config=variant,
        )
        if executable is not None:
            executables[name] = executable
    return executables


def accuracy_case(workload, shape, mode_name, variants, seed):
    mode = MODES[mode_name]
    a, b = host_inputs(shape, mode["input_dtype"], seed)
    executables = compile_set(
        a, b, phase="accuracy", workload=workload, shape=shape,
        mode_name=mode_name, mode=mode, variants=variants, reference=True,
    )
    passes = {name: False for name in variants}
    if {"native", "reference"}.issubset(executables):
        outputs = {
            name: executable(a, b) for name, executable in executables.items()
        }
        jax.block_until_ready(tuple(outputs.values()))
        native_error = bench.device_error(outputs["native"], outputs["reference"])
        for name in variants:
            if name not in outputs:
                continue
            reference_error = bench.device_error(outputs[name], outputs["reference"])
            direct_error = bench.device_error(outputs[name], outputs["native"])
            finite = all(
                math.isfinite(value)
                for error in (reference_error, direct_error)
                for value in error.values()
            )
            if native_error["rmse"] > 1e-30:
                ratio = reference_error["rmse"] / native_error["rmse"]
                passes[name] = (
                    finite and ratio <= 4.0
                    and reference_error["maxnorm_relative"] <= 0.01
                )
            else:
                ratio = None
                passes[name] = (
                    finite and direct_error["maxnorm_relative"] <= 0.01
                )
            emit({
                "kind": "accuracy",
                "workload": workload,
                "mode": mode_name,
                "name": name,
                "shape": shape,
                "seed": seed,
                "native_vs_reference": native_error,
                "candidate_vs_reference": reference_error,
                "candidate_vs_native": direct_error,
                "candidate_native_rmse_ratio": ratio,
                "finite": finite,
                "passes": passes[name],
            })
        del outputs
    emit({
        "kind": "accuracy_gates",
        "workload": workload,
        "mode": mode_name,
        "shape": shape,
        "passes": passes,
    })
    del executables, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def performance_case(workload, shape, mode_name, variants, allowed):
    mode = MODES[mode_name]
    a, b = constant_inputs(shape, mode["input_dtype"])
    executables = compile_set(
        a, b, phase="performance", workload=workload, shape=shape,
        mode_name=mode_name, mode=mode, variants=variants, reference=False,
    )
    executables = {
        name: executable for name, executable in executables.items()
        if name == "native" or allowed.get(name, False)
    }
    if "native" not in executables or len(executables) == 1:
        emit({
            "kind": "performance",
            "workload": workload,
            "mode": mode_name,
            "shape": shape,
            "success": False,
            "reason": "no compiling accuracy-qualified candidate",
        })
    else:
        timings, outputs = bench.interleaved_timings(executables, a, b, shape)
        speedups = {
            name: timings["native"]["mean_ms"] / timing["mean_ms"]
            for name, timing in timings.items() if name != "native"
        }
        emit({
            "kind": "performance",
            "workload": workload,
            "mode": mode_name,
            "shape": shape,
            "success": True,
            "timings": timings,
            "speedups_vs_native": speedups,
            "sentinels": {
                name: float(value[0, 0]) for name, value in outputs.items()
            },
        })
        del outputs
    del executables, a, b
    gc.collect()
    jax.clear_caches()


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "experiment": "rectangular_and_common_workloads",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "warmups": bench.WARMUPS,
        "runs": bench.RUNS,
        "large_workloads": LARGE_WORKLOADS,
        "decode_workloads": [
            (name, shape, variants) for name, shape, variants in DECODE_WORKLOADS
        ],
    })

    bf16_gates = accuracy_case(
        "long_k_accuracy", (2048, 14336, 2048),
        "bf16_bf16", BF16_LARGE_VARIANTS, 20260833,
    )
    fp32_gates = accuracy_case(
        "long_k_accuracy", (2048, 14336, 2048),
        "fp32_highest", FP32_LARGE_VARIANTS, 20260833,
    )
    decode_gates = {}
    for index, (name, shape, variants) in enumerate(DECODE_WORKLOADS):
        decode_gates[name] = accuracy_case(
            name, shape, "bf16_bf16", variants, 20260835 + index,
        )

    for workload, shape in LARGE_WORKLOADS:
        performance_case(
            workload, shape, "bf16_bf16", BF16_LARGE_VARIANTS, bf16_gates
        )
        performance_case(
            workload, shape, "fp32_highest", FP32_LARGE_VARIANTS, fp32_gates
        )
    for name, shape, variants in DECODE_WORKLOADS:
        performance_case(
            name, shape, "bf16_bf16", variants, decode_gates[name]
        )
    emit({
        "kind": "final",
        "success": True,
        "accuracy_gates": {
            "bf16_large": bf16_gates,
            "fp32_highest_large": fp32_gates,
            "decode": decode_gates,
        },
    })


if __name__ == "__main__":
    main()
