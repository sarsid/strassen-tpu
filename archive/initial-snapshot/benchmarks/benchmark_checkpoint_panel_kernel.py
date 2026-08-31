"""In-kernel exact-panel hybrid on the real layer under a raised VMEM ceiling.

The split-K realization confirmed the accuracy gain on TPU but lost its speed
to slice copies and an unfused partial add. This benchmark tests the fused
realization: ``classical_panels=(0,)`` inside the down-GEMM kernel, which
demands 83.79 MiB and therefore requires the process scoped-VMEM ceiling
raised to 96 MiB (run via ``run_panel_kernel_96m.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_outlier_panels as split_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_panel_kernel.jsonl")
WARMUPS = 10
RUNS = 20
DOWN_PANELS = (0,)
PANEL_VMEM_LIMIT_BYTES = 90 * 1024 * 1024


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("PANEL_KERNEL_JSON " + line, flush=True)


split_bench.emit = emit
layer_bench.emit = emit


def gemm(lhs, rhs, *, strassen=False, panels=()):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    if not strassen:
        output = sp.native_matmul(lhs, rhs)
    else:
        shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
        output = sp.strassen_matmul(
            lhs,
            rhs,
            interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
            classical_panels=panels,
            vmem_limit_bytes=(
                PANEL_VMEM_LIMIT_BYTES if panels else sp.TUNED_VMEM_LIMIT_BYTES
            ),
        )
    return jax.lax.optimization_barrier(output)


def make_layer(*, strassen, down_panels=()):
    def layer(x, params):
        residual, normalized = split_bench.layer_parts.attention_prefix(
            x, params
        )
        gate_up = gemm(normalized, params.gate_up, strassen=strassen)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(
            activated, params.mlp_down, strassen=strassen, panels=down_panels
        )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "down_panels": list(DOWN_PANELS),
        "panel_vmem_limit_bytes": PANEL_VMEM_LIMIT_BYTES,
    })

    x, params = layer_bench.load_inputs()
    iperm, _ = split_bench.calibrate_permutations(x, params)
    params_iperm = split_bench.permute_intermediate(params, iperm)
    jax.block_until_ready(params_iperm.gate_up)

    cases = {
        "matched_native": (make_layer(strassen=False), (x, params)),
        "strassen_control": (make_layer(strassen=True), (x, params)),
        "down_panel_kernel": (
            make_layer(strassen=True, down_panels=DOWN_PANELS),
            (x, params_iperm),
        ),
    }
    executables = {
        name: split_bench.compile_one(name, function, args)
        for name, (function, args) in cases.items()
    }
    outputs = {
        name: executables[name](*cases[name][1]) for name in cases
    }
    jax.block_until_ready(outputs)

    errors = {}
    for name, output in outputs.items():
        error = bench.device_error(output, outputs["matched_native"])
        errors[name] = error
        emit({"kind": "accuracy", "name": name, "vs_native": error})

    names = tuple(cases)
    for warmup in range(WARMUPS):
        order = names[warmup % len(names):] + names[:warmup % len(names)]
        for name in order:
            jax.block_until_ready(executables[name](*cases[name][1]))
    samples = {name: [] for name in names}
    for run in range(RUNS):
        order = names[run % len(names):] + names[:run % len(names)]
        if run % 2:
            order = tuple(reversed(order))
        for name in order:
            started = time.perf_counter_ns()
            jax.block_until_ready(executables[name](*cases[name][1]))
            samples[name].append((time.perf_counter_ns() - started) / 1e6)

    native_mean = statistics.fmean(samples["matched_native"])
    performance = {}
    for index, name in enumerate(names):
        mean_ms = statistics.fmean(samples[name])
        delta, low, high = split_bench.bootstrap_delta_interval(
            samples["matched_native"], samples[name], 20260817 + index
        )
        record = {
            "kind": "performance",
            "name": name,
            "mean_ms": mean_ms,
            "samples_ms": samples[name],
            "speedup_vs_native": native_mean / mean_ms,
            "candidate_minus_native_mean_ms": delta,
            "candidate_minus_native_bootstrap_95ci_ms": [low, high],
            "statistically_faster": name != "matched_native" and high < 0,
        }
        performance[name] = record
        emit(record)

    candidate = "down_panel_kernel"
    verdict = {
        "improves_l2": errors[candidate]["l2_relative"]
        < errors["strassen_control"]["l2_relative"],
        "improves_maxnorm": errors[candidate]["maxnorm_relative"]
        < errors["strassen_control"]["maxnorm_relative"],
        "retains_speed": performance[candidate]["statistically_faster"],
    }
    success = all(verdict.values())
    emit({"kind": "final", "success": success, "verdict": verdict})
    if not success:
        raise RuntimeError(f"panel kernel did not pass: {verdict}")


if __name__ == "__main__":
    main()
