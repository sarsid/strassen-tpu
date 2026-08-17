"""Real-layer gate for outlier-concentrated exact-panel Strassen hybrids.

Two candidates against the classical Strassen control on the pinned Mistral
layer, both built from free offline transformations plus the kernel's new
``classical_panels`` feature (selected K panels use the exact eight-product
update):

Both candidates reorder the intermediate dimension by calibrated activation
magnitude (a pure weight-column/row permutation of gate, up, and down),
concentrating outlier channels into the first 512 K channels of the down
GEMM, which are computed by a small exact native GEMM while Strassen covers
the remaining K:

- ``down_split_bf16``: the Strassen partial is stored BF16 (cheapest; one
  extra partial-sum rounding).
- ``down_split_fp32``: the Strassen partial stays FP32 through the kernel's
  output-as-accumulator path (no extra rounding; extra FP32 output traffic).

The in-kernel ``classical_panels`` branch was measured VMEM-infeasible at the
max48 tile (83.79 MiB compiler demand: both branch bodies allocate), and an
up-site split is excluded because its extra partial rounding or FP32 traffic
costs more than it saves. Local kernel-faithful emulation measured layer
error 0.2559%/0.1142% (bf16 split) versus the control's 0.3077%/0.2283%;
the fp32 split should recover the panel-variant 0.2414%/0.0571%. This
benchmark validates error and resident speed on TPU.
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


OUTPUT = Path("/content/results/strassen_checkpoint_outlier_panels.jsonl")
WARMUPS = 10
RUNS = 20
EXACT_K = 512


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("OUTLIER_PANELS_JSON " + line, flush=True)


def gemm(lhs, rhs, *, strassen=False, split=None):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    if not strassen:
        output = sp.native_matmul(lhs, rhs)
    elif split is None:
        shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
        output = sp.strassen_matmul(
            lhs,
            rhs,
            interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
            vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
        )
    else:
        # Exact native GEMM over the concentrated first EXACT_K channels in
        # FP32, Strassen over the rest, one fused FP32 add and BF16 store.
        # ``split`` selects the Strassen partial dtype.
        exact = jnp.matmul(
            lhs[:, :EXACT_K],
            rhs[:EXACT_K, :],
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        rest_shape = (lhs.shape[0], lhs.shape[1] - EXACT_K, rhs.shape[1])
        fast = sp.strassen_matmul(
            lhs[:, EXACT_K:],
            rhs[EXACT_K:, :],
            interleave_products=rest_shape in sp.TUNED_INTERLEAVED_SHAPES,
            output_dtype=split,
            vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
        )
        output = (exact + fast.astype(jnp.float32)).astype(jnp.bfloat16)
    return jax.lax.optimization_barrier(output)


def make_layer(*, strassen, down_split=None):
    def layer(x, params):
        residual, normalized = layer_parts.attention_prefix(x, params)
        gate_up = gemm(normalized, params.gate_up, strassen=strassen)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(
            activated, params.mlp_down, strassen=strassen, split=down_split
        )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def calibrate_permutations(x, params):
    """Channel-magnitude permutations from the native path on this workload."""
    prefix = jax.jit(layer_parts.attention_prefix).lower(x, params).compile()
    _, normalized = prefix(x, params)
    projected = jax.jit(sp.native_matmul).lower(
        normalized, params.gate_up
    ).compile()(normalized, params.gate_up)
    gate, up = jnp.split(projected, 2, axis=-1)
    activated = (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    inter_mags = np.max(
        np.abs(np.asarray(activated, dtype=np.float32)), axis=0
    )
    hidden_mags = np.max(
        np.abs(np.asarray(normalized, dtype=np.float32)), axis=0
    )
    iperm = np.argsort(-inter_mags)
    hperm = np.argsort(-hidden_mags)
    emit({
        "kind": "calibration",
        "inter_top8": inter_mags[iperm[:8]].tolist(),
        "inter_median": float(np.median(inter_mags)),
        "hidden_top8": hidden_mags[hperm[:8]].tolist(),
        "hidden_median": float(np.median(hidden_mags)),
    })
    return iperm, hperm


def permute_intermediate(params, iperm):
    gate, up = jnp.split(params.gate_up, 2, axis=-1)
    return params._replace(
        gate_up=jnp.concatenate((gate[:, iperm], up[:, iperm]), axis=1),
        mlp_down=params.mlp_down[iperm, :],
    )


def permute_hidden(x, params, hperm):
    return x[:, hperm], params._replace(
        attention_norm=params.attention_norm[hperm],
        query=params.query[hperm, :],
        key=params.key[hperm, :],
        value=params.value[hperm, :],
        attention_output=params.attention_output[:, hperm],
        mlp_norm=params.mlp_norm[hperm],
        gate_up=params.gate_up[hperm, :],
        mlp_down=params.mlp_down[:, hperm],
    )


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
        "contract": (
            "free intermediate-dim outlier permutation + split-K exact slice"
        ),
        "exact_k": EXACT_K,
    })

    x, params = layer_bench.load_inputs()
    iperm, hperm = calibrate_permutations(x, params)

    params_iperm = permute_intermediate(params, iperm)
    jax.block_until_ready(params_iperm.gate_up)

    cases = {
        "matched_native": (
            make_layer(strassen=False), (x, params)
        ),
        "strassen_control": (
            make_layer(strassen=True), (x, params)
        ),
        "down_split_bf16": (
            make_layer(strassen=True, down_split=jnp.bfloat16),
            (x, params_iperm),
        ),
        "down_split_fp32": (
            make_layer(strassen=True, down_split=jnp.float32),
            (x, params_iperm),
        ),
    }

    executables = {
        name: compile_one(name, function, args)
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
        delta, low, high = bootstrap_delta_interval(
            samples["matched_native"], samples[name], 20260816 + index
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

    control = errors["strassen_control"]
    verdicts = {}
    for name in ("down_split_bf16", "down_split_fp32"):
        verdicts[name] = {
            "improves_l2": errors[name]["l2_relative"]
            < control["l2_relative"],
            "improves_maxnorm": errors[name]["maxnorm_relative"]
            < control["maxnorm_relative"],
            "retains_speed": performance[name]["statistically_faster"],
        }
    success = any(
        all(v.values()) for v in verdicts.values()
    )
    emit({"kind": "final", "success": success, "verdicts": verdicts})
    if not success:
        raise RuntimeError(f"no candidate passed: {verdicts}")


if __name__ == "__main__":
    main()
