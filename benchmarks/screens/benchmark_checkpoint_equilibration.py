"""Test precomputed power-of-two inner equilibration on a real Mistral layer.

For A @ B, a diagonal power-of-two matrix D gives the exact identity
(A D) @ (D^-1 B).  The weight transformation is performed once outside the
timed layer.  The timed candidate only scales activation columns before each
of the same two seven-product Strassen calls.  Exponents are calibrated from
the existing layer-0 checkpoint workload and then held fixed during timing.
"""

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

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_equilibration.jsonl")
BEST_UP_PERMUTATION = 4
BEST_DOWN_PERMUTATION = 7
WARMUPS = 10
RUNS = 20


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_EQUILIBRATION_JSON " + line, flush=True)


def max_exponent(max_abs):
    value = max_abs.astype(jnp.float32)
    safe = jnp.where(value > 0, value, jnp.ones_like(value))
    exponent = jnp.floor(jnp.log2(safe)).astype(jnp.int32)
    return jnp.where(value > 0, exponent, jnp.zeros_like(exponent))


@jax.jit
def balance_exponents(a, b):
    a_exponent = max_exponent(jnp.max(jnp.abs(a), axis=0))
    b_exponent = max_exponent(jnp.max(jnp.abs(b), axis=1))
    exponent = jnp.rint(
        (b_exponent - a_exponent).astype(jnp.float32) * 0.5
    ).astype(jnp.int32)
    return jnp.clip(exponent, -30, 30)


def power_of_two(exponent, dtype):
    return jnp.exp2(exponent.astype(jnp.float32)).astype(dtype)


def native_activation(normalized, gate_up):
    projected = permutation_bench.gemm(normalized, gate_up)
    gate, up = jnp.split(projected, 2, axis=-1)
    return (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def prepare_equilibrated_params(params, normalized):
    up_exponent = balance_exponents(normalized, params.gate_up)
    activation = jax.jit(native_activation)(normalized, params.gate_up)
    down_exponent = balance_exponents(activation, params.mlp_down)
    up_scale = power_of_two(up_exponent, jnp.bfloat16)
    down_scale = power_of_two(down_exponent, jnp.bfloat16)
    scaled_params = params._replace(
        gate_up=(
            params.gate_up.astype(jnp.float32)
            * power_of_two(-up_exponent, jnp.float32)[:, None]
        ).astype(jnp.bfloat16),
        mlp_down=(
            params.mlp_down.astype(jnp.float32)
            * power_of_two(-down_exponent, jnp.float32)[:, None]
        ).astype(jnp.bfloat16),
    )
    jax.block_until_ready((scaled_params, up_scale, down_scale))
    emit({
        "kind": "calibration",
        "up_exponent_min": int(jnp.min(up_exponent)),
        "up_exponent_max": int(jnp.max(up_exponent)),
        "up_exponent_nonzero": int(jnp.count_nonzero(up_exponent)),
        "down_exponent_min": int(jnp.min(down_exponent)),
        "down_exponent_max": int(jnp.max(down_exponent)),
        "down_exponent_nonzero": int(jnp.count_nonzero(down_exponent)),
    })
    return scaled_params, up_scale, down_scale


def make_layer(*, native, up_permutation=0, down_permutation=0, equilibrated=False):
    def layer(x, params, up_scale, down_scale):
        residual, normalized = permutation_bench.attention_prefix(x, params)
        if equilibrated:
            normalized = (normalized * up_scale[None, :]).astype(jnp.bfloat16)
        if native:
            gate_up = permutation_bench.gemm(normalized, params.gate_up)
        else:
            gate_up = permutation_bench.gemm(
                normalized, params.gate_up, up_permutation
            )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        if equilibrated:
            activated = (activated * down_scale[None, :]).astype(jnp.bfloat16)
        if native:
            mlp_output = permutation_bench.gemm(activated, params.mlp_down)
        else:
            mlp_output = permutation_bench.gemm(
                activated, params.mlp_down, down_permutation
            )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def compile_one(name, function, args):
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*args)
    audit = layer_bench.lowered_audit(lowered)
    executable = lowered.compile()
    emit({
        "kind": "compile",
        "name": name,
        "compile_s": time.perf_counter() - started,
        "lowered_audit": audit,
    })
    return executable


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
        "jax": jax.__version__,
        "contract": "precomputed exact power-of-two inner equilibration",
        "best_permutations": [BEST_UP_PERMUTATION, BEST_DOWN_PERMUTATION],
    })

    x, params = layer_bench.load_inputs()
    prefix = jax.jit(permutation_bench.attention_prefix).lower(x, params).compile()
    _, normalized = prefix(x, params)
    scaled_params, up_scale, down_scale = prepare_equilibrated_params(
        params, normalized
    )
    ones_up = jnp.ones_like(up_scale)
    ones_down = jnp.ones_like(down_scale)

    cases = {
        "matched_native": (
            make_layer(native=True),
            (x, params, ones_up, ones_down),
        ),
        "unpermuted": (
            make_layer(native=False),
            (x, params, ones_up, ones_down),
        ),
        "permuted": (
            make_layer(
                native=False,
                up_permutation=BEST_UP_PERMUTATION,
                down_permutation=BEST_DOWN_PERMUTATION,
            ),
            (x, params, ones_up, ones_down),
        ),
        "equilibrated": (
            make_layer(native=False, equilibrated=True),
            (x, scaled_params, up_scale, down_scale),
        ),
        "equilibrated_permuted": (
            make_layer(
                native=False,
                up_permutation=BEST_UP_PERMUTATION,
                down_permutation=BEST_DOWN_PERMUTATION,
                equilibrated=True,
            ),
            (x, scaled_params, up_scale, down_scale),
        ),
    }
    executables = {
        name: compile_one(name, function, args)
        for name, (function, args) in cases.items()
    }
    outputs = {
        name: executables[name](*cases[name][1])
        for name in cases
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
        delta, low, high = bootstrap_delta_interval(
            samples["matched_native"], samples[name], 20260819 + index
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

    candidate = "equilibrated_permuted"
    improves_l2 = (
        errors[candidate]["l2_relative"]
        < errors["unpermuted"]["l2_relative"]
    )
    retains_speed = performance[candidate]["statistically_faster"]
    emit({
        "kind": "final",
        "success": improves_l2 and retains_speed,
        "improves_l2": improves_l2,
        "retains_speed": retains_speed,
    })
    if not (improves_l2 and retains_speed):
        raise RuntimeError("equilibration did not reduce error while retaining speed")


if __name__ == "__main__":
    main()
