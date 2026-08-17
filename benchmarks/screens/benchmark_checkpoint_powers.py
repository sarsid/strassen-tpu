"""Real-layer gate for a lower-growth power-of-two rank-7 formula.

The candidate is the Dumas--Pernet--Sedoglavic formula with growth factor
12.2034, implemented with seven MXU products, exact binary input coefficients,
and a ten-update sparse output basis.  Native attention and every non-MLP
operation remain fixed.  Up-only, down-only, and both-site policies determine
whether either checkpoint weight distribution benefits before an all-layer run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_permutations as layer_parts
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_powers.jsonl")
WARMUPS = 10
RUNS = 20
MODES = {
    "matched_native": (None, None),
    "classical_control": ("classical", "classical"),
    "powers_up": ("powers", "classical"),
    "powers_down": ("classical", "powers"),
    "powers_both": ("powers", "powers"),
}


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_POWERS_JSON " + line, flush=True)


def gemm(lhs, rhs, formula=None):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    if formula is None:
        output = sp.native_matmul(lhs, rhs)
    else:
        shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
        output = sp.strassen_matmul(
            lhs,
            rhs,
            formula_variant=formula,
            interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
            vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
        )
    return jax.lax.optimization_barrier(output)


def make_layer(mode):
    up_formula, down_formula = MODES[mode]

    def layer(x, params):
        residual, normalized = layer_parts.attention_prefix(x, params)
        gate_up = gemm(normalized, params.gate_up, up_formula)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(activated, params.mlp_down, down_formula)
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
        "jax": jax.__version__,
        "profile": sp.VMEM_PROFILE,
        "tile": [sp.TUNED_BM, sp.TUNED_BN, sp.TUNED_BK],
        "modes": MODES,
        "candidate": {
            "formula": "Dumas--Pernet--Sedoglavic power-of-two rank-7",
            "growth_factor": 12.2034,
            "mxu_products": 7,
            "persistent_output_updates_per_panel": 10,
            "output_basis_decode": "once after final K panel",
        },
        "warmups": WARMUPS,
        "runs": RUNS,
    })

    x, params = layer_bench.load_inputs()
    executables = {}
    for name in MODES:
        started = time.perf_counter()
        lowered = jax.jit(make_layer(name)).lower(x, params)
        try:
            executables[name] = lowered.compile()
        except Exception as error:
            emit({
                "kind": "compile_failure",
                "name": name,
                "compile_s": time.perf_counter() - started,
                "lowered_audit": layer_bench.lowered_audit(lowered),
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise
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
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(output))))
        errors[name] = error
        emit({
            "kind": "accuracy",
            "name": name,
            "finite": finite,
            "vs_native": error,
        })

    names = tuple(executables)
    for warmup in range(WARMUPS):
        offset = warmup % len(names)
        for name in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[name](x, params))

    samples = {name: [] for name in names}
    for run in range(RUNS):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
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
            samples["matched_native"], samples[name], 20260821 + index
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

    control_l2 = errors["classical_control"]["l2_relative"]
    viable = [
        name
        for name in ("powers_up", "powers_down", "powers_both")
        if errors[name]["l2_relative"] <= 0.8 * control_l2
        and performance[name]["statistically_faster"]
    ]
    emit({
        "kind": "final",
        "success": bool(viable),
        "viable": viable,
        "advance_rule": "at least 20% lower layer L2 and significantly faster",
    })
    if not viable:
        raise RuntimeError("lower-growth powers formula did not clear the layer gate")


if __name__ == "__main__":
    main()
