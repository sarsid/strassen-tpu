"""Product-aware SwiGLU inference screen at Qwen3-32B geometry."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_use_enhanced_launch_barrier=true "
    "--xla_tpu_scoped_vmem_limit_kib=49152")

import jax
import jax.numpy as jnp

import benchmark_common as bench
import benchmark_cubic_control as cubic
import benchmark_qwen3_32b_layer as layer
import strassen_pallas as sp


TOKENS, MODEL_DIM, INTERMEDIATE = (
    layer.TOKENS, layer.MODEL_DIM, layer.INTERMEDIATE_DIM)
BM, BN, BK = (int(v) for v in os.environ.get(
    "QWEN3_PRODUCT_TILE", "2048,2048,512").split(","))
CUBIC_BM, CUBIC_BN, CUBIC_BK = (int(v) for v in os.environ.get(
    "QWEN3_CUBIC_TILE", f"{BM},{BN},{BK}").split(","))
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
WARMUPS, RUNS = 3, 20
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(
    f"{RESULTS_DIR}/strassen_qwen3_{layer.MODEL_NAME}"
    f"_product_aware_inference{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_PRODUCT_INFERENCE_JSON " + line, flush=True)


def activate(full):
    gate, up = jnp.split(full, 2, axis=-1)
    return (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata", "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "model": layer.MODEL_NAME, "repository": layer.REPOSITORY,
        "shape": [TOKENS, MODEL_DIM, 2 * INTERMEDIATE],
        "tile": [BM, BN, BK], "cubic_tile": [CUBIC_BM, CUBIC_BN, CUBIC_BK],
        "arms": ["regular_xla", "cubic_pallas", "strassen_fused",
                 "product_strassen_fused"],
        "changed_field": "final-panel product order and early top-half SwiGLU",
        "gate": (
            "product Strassen CI upper < standard fused Strassen and XLA; "
            "relative L2 <= standard Strassen + 1e-3"),
    })
    x, weight = bench.device_uniform_inputs(
        (TOKENS, MODEL_DIM, 2 * INTERMEDIATE), 20260904)
    laid = sp.swiglu_weight_layout(weight, BN)
    jax.block_until_ready((x, weight, laid))

    functions = {
        "regular_xla": lambda a, w, wl: activate(sp.native_matmul(a, w)),
        "cubic_pallas": lambda a, w, wl: activate(cubic.cubic_matmul(
            a, w, variant="blocked", bm=CUBIC_BM, bn=CUBIC_BN, bk=CUBIC_BK,
            vmem_limit_bytes=48 * 1024 * 1024)),
        "strassen_fused": lambda a, w, wl: sp.strassen_matmul(
            a, wl, bm=BM, bn=BN, bk=BK, epilogue="swiglu",
            interleave_products=True, vmem_limit_bytes=47 * 1024 * 1024),
        "product_strassen_fused": lambda a, w, wl: sp.strassen_matmul(
            a, wl, bm=BM, bn=BN, bk=BK, epilogue="swiglu",
            interleave_products=True, product_aware_swiglu=True,
            vmem_limit_bytes=47 * 1024 * 1024),
    }
    args = (x, weight, laid)
    executables, samples, errors = {}, {}, {}
    reference = None
    for name, function in functions.items():
        lowered = jax.jit(function).lower(*args)
        started = time.perf_counter()
        executable = lowered.compile()
        emit({"kind": "compile", "arm": name,
              "seconds": time.perf_counter() - started,
              "stablehlo_custom_calls": lowered.as_text().count(
                  "stablehlo.custom_call"),
              "stablehlo_dots": lowered.as_text().count(
                  "stablehlo.dot_general")})
        output = executable(*args)
        jax.block_until_ready(output)
        if name == "regular_xla":
            reference = output
        else:
            errors[name] = {
                key: float(value)
                for key, value in bench.device_error(output, reference).items()}
            emit({"kind": "accuracy", "arm": name,
                  "error_vs_xla": errors[name]})
        executables[name] = executable

    names = tuple(executables)
    for warmup in range(WARMUPS):
        offset = warmup % len(names)
        for name in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[name](*args))
    samples = {name: [] for name in names}
    for run in range(RUNS):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
        if run % 2:
            order = tuple(reversed(order))
        for name in order:
            started_ns = time.perf_counter_ns()
            jax.block_until_ready(executables[name](*args))
            samples[name].append((time.perf_counter_ns() - started_ns) / 1e6)
    for name, values in samples.items():
        emit({"kind": "performance", "arm": name,
              "mean_ms": statistics.fmean(values), "samples_ms": values})

    def independent(base, candidate):
        mean = statistics.fmean(candidate) - statistics.fmean(base)
        half = 2.093 * math.sqrt(
            statistics.variance(base) / len(base)
            + statistics.variance(candidate) / len(candidate))
        return {"mean_ms": mean, "ci95_ms": [mean - half, mean + half]}

    product_vs_standard = independent(
        samples["strassen_fused"], samples["product_strassen_fused"])
    product_vs_xla = independent(
        samples["regular_xla"], samples["product_strassen_fused"])
    product_error = errors["product_strassen_fused"]["l2_relative"]
    standard_error = errors["strassen_fused"]["l2_relative"]
    schedule_pass = (
        product_vs_standard["ci95_ms"][1] < 0
        and product_error <= standard_error + 1e-3)
    deployment_pass = product_vs_xla["ci95_ms"][1] < 0
    emit({
        "kind": "comparison",
        "mean_ms": {name: statistics.fmean(value)
                    for name, value in samples.items()},
        "product_minus_standard_strassen": product_vs_standard,
        "product_minus_xla": product_vs_xla,
        "speedup_vs_xla": (
            statistics.fmean(samples["regular_xla"])
            / statistics.fmean(samples["product_strassen_fused"])),
        "schedule_pass": schedule_pass,
        "deployment_pass": deployment_pass,
    })
    emit({"kind": "verdict", "passes": schedule_pass and deployment_pass,
          "schedule_pass": schedule_pass,
          "deployment_pass": deployment_pass,
          "scope": "Qwen3 gate/up plus SwiGLU inference microtask"})


if __name__ == "__main__":
    main()
