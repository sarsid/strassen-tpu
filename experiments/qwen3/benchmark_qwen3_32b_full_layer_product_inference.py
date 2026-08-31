"""Real-weight Qwen3-32B layer with product-aware fused SwiGLU."""

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

import benchmark_common as common
import benchmark_cubic_control as cubic
import benchmark_qwen3_32b_layer as layer
import mosaic_compat
import strassen_pallas as sp


BM, BN, BK = (int(v) for v in os.environ.get(
    "QWEN3_PRODUCT_TILE", "2048,2048,512").split(","))
CUBIC_BM, CUBIC_BN, CUBIC_BK = (int(v) for v in os.environ.get(
    "QWEN3_CUBIC_TILE", f"{BM},{BN},{BK}").split(","))
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
WARMUPS, RUNS = 3, 20
ARMS = (
    "regular_xla", "cubic_pallas",
    "strassen_fused", "product_strassen_fused",
)
OUTPUT_ROOT = Path(os.environ.get("STRASSEN_OUTPUT_DIR", "/content/runs"))
OUTPUT = OUTPUT_ROOT / (
    f"strassen_qwen3_{layer.MODEL_NAME}"
    f"_full_layer_product_inference{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_FULL_PRODUCT_INFERENCE_JSON " + line, flush=True)


def activate(full):
    gate, up = jnp.split(full, 2, axis=-1)
    return (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def make_layer(arm):
    def run(x, params, laid_gate_up):
        residual, normalized = layer.attention_prefix(x, params)
        if arm == "regular_xla":
            activated = activate(sp.native_matmul(normalized, params.gate_up))
        elif arm == "cubic_pallas":
            activated = activate(cubic.cubic_matmul(
                normalized, params.gate_up, variant="blocked",
                bm=CUBIC_BM, bn=CUBIC_BN, bk=CUBIC_BK,
                vmem_limit_bytes=48 * 1024 * 1024))
        else:
            activated = sp.strassen_matmul(
                normalized, laid_gate_up,
                bm=BM, bn=BN, bk=BK,
                epilogue="swiglu", interleave_products=True,
                product_aware_swiglu=(arm == "product_strassen_fused"),
                vmem_limit_bytes=47 * 1024 * 1024)
        projected = sp.native_matmul(activated, params.mlp_down)
        return (
            residual.astype(jnp.float32) + projected.astype(jnp.float32)
        ).astype(jnp.bfloat16)
    return run


def independent(reference, candidate):
    mean = statistics.fmean(candidate) - statistics.fmean(reference)
    half = 2.093 * math.sqrt(
        statistics.variance(reference) / len(reference)
        + statistics.variance(candidate) / len(candidate))
    return {"mean_ms": mean, "ci95_ms": [mean - half, mean + half]}


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata", "repository": layer.REPOSITORY,
        "revision": layer.REVISION, "device": jax.devices()[0].device_kind,
        "jax": jax.__version__, "arms": list(ARMS),
        "model": layer.MODEL_NAME,
        "tile": [BM, BN, BK], "cubic_tile": [CUBIC_BM, CUBIC_BN, CUBIC_BK],
        "policy": "gate/up+SwiGLU only; down and attention ordinary XLA",
        "scope": f"complete real-weight Qwen3-{layer.MODEL_NAME} layer-0 inference",
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
    })
    layer.emit = emit
    x = layer.deterministic_hidden()
    params = layer.load_params()
    laid = sp.swiglu_weight_layout(params.gate_up, BN)
    jax.block_until_ready((x, params, laid))
    args = (x, params, laid)

    executables, outputs, errors = {}, {}, {}
    for arm in ARMS:
        lowered = jax.jit(make_layer(arm)).lower(*args)
        started = time.perf_counter()
        executable = lowered.compile()
        emit({
            "kind": "compile", "arm": arm,
            "seconds": time.perf_counter() - started,
            "stablehlo_custom_calls": lowered.as_text().count(
                "stablehlo.custom_call"),
            "stablehlo_dots": lowered.as_text().count("stablehlo.dot_general"),
        })
        executables[arm] = executable
        outputs[arm] = executable(*args)
        jax.block_until_ready(outputs[arm])
    reference = outputs["regular_xla"]
    for arm in ARMS[1:]:
        errors[arm] = common.device_error(outputs[arm], reference)
    emit({"kind": "accuracy", "errors_vs_xla": errors})

    names = tuple(executables)
    for warmup in range(WARMUPS):
        offset = warmup % len(names)
        for arm in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[arm](*args))
    samples = {arm: [] for arm in names}
    for run in range(RUNS):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
        if run % 2:
            order = tuple(reversed(order))
        for arm in order:
            started_ns = time.perf_counter_ns()
            jax.block_until_ready(executables[arm](*args))
            samples[arm].append((time.perf_counter_ns() - started_ns) / 1e6)
    means = {arm: statistics.fmean(value) for arm, value in samples.items()}
    product_vs_standard = independent(
        samples["strassen_fused"], samples["product_strassen_fused"])
    product_vs_xla = independent(
        samples["regular_xla"], samples["product_strassen_fused"])
    product_vs_cubic = independent(
        samples["cubic_pallas"], samples["product_strassen_fused"])
    schedule_pass = (
        product_vs_standard["ci95_ms"][1] < 0
        and errors["product_strassen_fused"]["l2_relative"]
        <= errors["strassen_fused"]["l2_relative"] + 1e-3)
    deployment_pass = product_vs_xla["ci95_ms"][1] < 0
    algorithmic_pass = product_vs_cubic["ci95_ms"][1] < 0
    emit({
        "kind": "performance", "mean_ms": means,
        "samples_ms": samples,
        "product_minus_standard_strassen": product_vs_standard,
        "product_minus_xla": product_vs_xla,
        "product_minus_cubic": product_vs_cubic,
        "speedup_vs_xla": means["regular_xla"] / means["product_strassen_fused"],
        "speedup_vs_cubic": means["cubic_pallas"] / means["product_strassen_fused"],
        "schedule_pass": schedule_pass,
        "deployment_pass": deployment_pass,
        "algorithmic_pass": algorithmic_pass,
    })
    emit({
        "kind": "verdict",
        "passes": schedule_pass and deployment_pass and algorithmic_pass,
        "schedule_pass": schedule_pass,
        "deployment_pass": deployment_pass,
        "algorithmic_pass": algorithmic_pass,
        "scope": "complete real-weight single-layer inference",
    })


if __name__ == "__main__":
    main()
