"""Common runner for isolated scoped-VMEM deployment profiles.

Each tiny wrapper sets STRASSEN_VMEM_PROFILE and LIBTPU_INIT_ARGS before this
module imports JAX. Run wrappers in fresh TPU kernels because the global scoped
VMEM ceiling is consumed when libtpu initializes.
"""

from __future__ import annotations

import functools
import gc
import json
import math
import os
import time
from pathlib import Path

import strassen_pallas as sp

import jax
import jax.numpy as jnp

import benchmark_common as bench


MIB = 1024 * 1024
ACCURACY_SHAPE = (2048, 2048, 2048)
PERFORMANCE_SIZES = (8192, 12288, 16384)

PROFILE_NAME = sp.VMEM_PROFILE
PROFILE = {
    "tile": (sp.TUNED_BM, sp.TUNED_BN, sp.TUNED_BK),
    "global_mib": sp.TUNED_SCOPED_VMEM_KIB // 1024,
    "limit_mib": (
        None
        if sp.TUNED_VMEM_LIMIT_BYTES is None
        else sp.TUNED_VMEM_LIMIT_BYTES // MIB
    ),
}
OUTPUT = Path(f"/content/results/strassen_vmem_profile_{PROFILE_NAME}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("VMEM_PROFILE_JSON " + line, flush=True)


def compile_variant(a, b):
    bm, bn, bk = PROFILE["tile"]
    fn = functools.partial(
        sp.strassen_matmul,
        bm=bm,
        bn=bn,
        bk=bk,
        interpret=False,
        vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
    )
    return jax.jit(fn).lower(a, b).compile()


def compile_all(a, b, phase, shape, reference=False):
    executables = {}
    functions = {"native": bench.native_bf16_gemm}
    if reference:
        functions["reference"] = bench.high_precision_reference
    for name, fn in functions.items():
        started = time.perf_counter()
        executables[name] = jax.jit(fn).lower(a, b).compile()
        emit({
            "kind": "compile",
            "phase": phase,
            "shape": shape,
            "name": name,
            "success": True,
            "compile_s": time.perf_counter() - started,
        })
    started = time.perf_counter()
    try:
        executables["strassen"] = compile_variant(a, b)
        emit({
            "kind": "compile", "phase": phase, "shape": shape,
            "name": "strassen", "config": PROFILE, "success": True,
            "compile_s": time.perf_counter() - started,
        })
    except Exception as error:
        emit({
            "kind": "compile", "phase": phase, "shape": shape,
            "name": "strassen", "config": PROFILE, "success": False,
            "compile_s": time.perf_counter() - started,
            "error_type": type(error).__name__, "error": str(error),
        })
    return executables


def run_accuracy():
    distribution, seed = "uniform", 20260815
    a, b = bench.device_uniform_inputs(ACCURACY_SHAPE, seed)
    executables = compile_all(a, b, "accuracy", ACCURACY_SHAPE, reference=True)
    outputs = {name: executable(a, b) for name, executable in executables.items()}
    jax.block_until_ready(tuple(outputs.values()))
    reference = outputs["reference"]
    native_error = bench.device_error(outputs["native"], reference)
    passes = {"strassen": False}
    if "strassen" in outputs:
        error = bench.device_error(outputs["strassen"], reference)
        native_ratio = error["rmse"] / max(native_error["rmse"], 1e-30)
        finite = all(math.isfinite(value) for value in error.values())
        passes["strassen"] = (
            finite and native_ratio <= 4.0 and error["maxnorm_relative"] <= 0.01
        )
        emit({
            "kind": "accuracy",
            "profile": PROFILE_NAME,
            "name": "strassen",
            "shape": ACCURACY_SHAPE,
            "distribution": distribution,
            "seed": seed,
            "finite": finite,
            "native_vs_reference": native_error,
            "candidate_vs_reference": error,
            "candidate_native_rmse_ratio": native_ratio,
        })
    emit({"kind": "accuracy_gates", "passes": passes})
    del outputs, executables, a, b
    gc.collect()
    jax.clear_caches()
    return passes


def run_performance(size, allowed):
    a = jnp.full((size, size), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((size, size), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    executables = compile_all(a, b, "performance", size)
    executables = {
        name: executable for name, executable in executables.items()
        if name == "native" or allowed.get(name, False)
    }
    timings, outputs = bench.interleaved_timings(
        executables, a, b, (size, size, size)
    )
    speedups = {
        name: timings["native"]["mean_ms"] / timing["mean_ms"]
        for name, timing in timings.items() if name != "native"
    }
    emit({
        "kind": "performance",
        "profile": PROFILE_NAME,
        "size": size,
        "timings": timings,
        "speedups_vs_native": speedups,
        "selected": "strassen",
        "selected_speedup_vs_native": speedups.get("strassen"),
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
        "profile": PROFILE_NAME,
        "profile_config": PROFILE,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "warmups": bench.WARMUPS,
        "runs": bench.RUNS,
    })
    passes = run_accuracy()
    if not passes.get("strassen", False):
        emit({"kind": "final", "success": False, "reason": "accuracy/compile"})
        return
    for size in PERFORMANCE_SIZES:
        run_performance(size, passes)
    emit({"kind": "final", "success": True})


if __name__ == "__main__":
    main()
