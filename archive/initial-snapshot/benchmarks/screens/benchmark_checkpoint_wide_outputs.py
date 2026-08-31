"""Test retaining Strassen FP32 outputs through adjacent MLP operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_permutations as permutation_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_wide_outputs.jsonl")
WARMUPS = 10
RUNS = 20
MODES = {
    "matched_native": (False, False),
    "bf16_control": (False, False),
    "fp32_up": (True, False),
    "fp32_down": (False, True),
    "fp32_both": (True, True),
}


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_WIDE_JSON " + line, flush=True)


def strassen(lhs, rhs, *, wide):
    shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    output = sp.strassen_matmul(
        lhs,
        rhs,
        interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
        output_dtype=jnp.float32 if wide else jnp.bfloat16,
        vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
    )
    return jax.lax.optimization_barrier(output)


def make_layer(name):
    wide_up, wide_down = MODES[name]
    native = name == "matched_native"

    def layer(x, params):
        residual, normalized = permutation_bench.attention_prefix(x, params)
        gate_up = (
            permutation_bench.gemm(normalized, params.gate_up)
            if native
            else strassen(normalized, params.gate_up, wide=wide_up)
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = (
            permutation_bench.gemm(activated, params.mlp_down)
            if native
            else strassen(activated, params.mlp_down, wide=wide_down)
        )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def bootstrap_delta_interval(reference, candidate, seed):
    deltas = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(deltas, size=(20000, len(deltas)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(deltas)), float(low), float(high)


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "repository": layer_bench.REPOSITORY,
        "revision": layer_bench.REVISION,
        "device": jax.devices()[0].device_kind,
        "modes": MODES,
        "contract": "same checkpoint/workload; widen Strassen output only",
    })
    x, params = layer_bench.load_inputs()
    executables = {}
    for name in MODES:
        started = time.perf_counter()
        lowered = jax.jit(make_layer(name)).lower(x, params)
        executables[name] = lowered.compile()
        emit({
            "kind": "compile",
            "name": name,
            "compile_s": time.perf_counter() - started,
            "lowered_audit": layer_bench.lowered_audit(lowered),
        })

    outputs = {name: executable(x, params) for name, executable in executables.items()}
    jax.block_until_ready(outputs)
    errors = {}
    for name, output in outputs.items():
        error = bench.device_error(output, outputs["matched_native"])
        errors[name] = error
        emit({"kind": "accuracy", "name": name, "vs_native": error})

    names = tuple(executables)
    for warmup in range(WARMUPS):
        order = names[warmup % len(names):] + names[:warmup % len(names)]
        for name in order:
            jax.block_until_ready(executables[name](x, params))
    samples = {name: [] for name in names}
    for run in range(RUNS):
        order = names[run % len(names):] + names[:run % len(names)]
        if run % 2:
            order = tuple(reversed(order))
        for name in order:
            started = time.perf_counter_ns()
            jax.block_until_ready(executables[name](x, params))
            samples[name].append((time.perf_counter_ns() - started) / 1e6)

    native_mean = statistics.fmean(samples["matched_native"])
    performance = {}
    for index, name in enumerate(names):
        mean_ms = statistics.fmean(samples[name])
        delta, low, high = bootstrap_delta_interval(
            samples["matched_native"], samples[name], 20260820 + index
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

    viable = [
        name for name in ("fp32_up", "fp32_down", "fp32_both")
        if errors[name]["l2_relative"] < errors["bf16_control"]["l2_relative"]
        and performance[name]["statistically_faster"]
    ]
    emit({"kind": "final", "success": bool(viable), "viable": viable})
    if not viable:
        raise RuntimeError("no wide-output mode reduced error while retaining speed")


if __name__ == "__main__":
    main()
