"""Screen algebraically equivalent Strassen block permutations on layer 0.

The eight variants swap the two local M, K, and N block halves before the
same classical seven-product formula, then restore the output-quadrant order.
They add no products, buffers, or tensor materializations.  Real checkpoint
weights and the existing 8x1024 layer workload select one permutation for the
combined gate/up projection and one for the down projection.  Only the chosen
pair and the unpermuted control advance to full-layer timing.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import statistics
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import benchmark_checkpoint_layer as layer_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_permutations.jsonl")
FORMULA_VARIANT = "classical"
PERMUTATIONS = tuple(range(8))
WARMUPS = 10
RUNS = 20


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_PERMUTATION_JSON " + line, flush=True)


def candidate_matmul(lhs, rhs, permutation):
    shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
    return sp.strassen_matmul(
        lhs,
        rhs,
        interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
        block_permutation=permutation,
        formula_variant=FORMULA_VARIANT,
        vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
    )


def gemm(lhs, rhs, permutation=None):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    output = (
        sp.native_matmul(lhs, rhs)
        if permutation is None
        else candidate_matmul(lhs, rhs, permutation)
    )
    return jax.lax.optimization_barrier(output)


def attention_prefix(x, params):
    normalized = layer_bench.rms_norm(x, params.attention_norm)
    query = gemm(normalized, params.query)
    key = gemm(normalized, params.key)
    value = gemm(normalized, params.value)
    query = query.reshape(
        layer_bench.BATCH,
        layer_bench.SEQUENCE,
        layer_bench.HEADS,
        layer_bench.HEAD_DIM,
    )
    key = key.reshape(
        layer_bench.BATCH,
        layer_bench.SEQUENCE,
        layer_bench.KV_HEADS,
        layer_bench.HEAD_DIM,
    )
    value = value.reshape(
        layer_bench.BATCH,
        layer_bench.SEQUENCE,
        layer_bench.KV_HEADS,
        layer_bench.HEAD_DIM,
    )
    query = layer_bench.apply_rope(query, params.rope_cos, params.rope_sin)
    key = layer_bench.apply_rope(key, params.rope_cos, params.rope_sin)
    attended = jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="xla",
    )
    attended = attended.reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    ).astype(jnp.bfloat16)
    attention_output = gemm(attended, params.attention_output)
    residual = (
        x.astype(jnp.float32) + attention_output.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    normalized = layer_bench.rms_norm(residual, params.mlp_norm)
    return residual, normalized


def mlp_from_prefix(residual, normalized, params, up_permutation, down_permutation):
    gate_up = gemm(normalized, params.gate_up, up_permutation)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    mlp_output = gemm(activated, params.mlp_down, down_permutation)
    return (
        residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def make_layer(up_permutation, down_permutation):
    def layer(x, params):
        residual, normalized = attention_prefix(x, params)
        return mlp_from_prefix(
            residual,
            normalized,
            params,
            up_permutation,
            down_permutation,
        )

    return layer


def compile_one(name, function, *args):
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


def screen_site(site, residual, normalized, params, reference):
    records = []
    for permutation in PERMUTATIONS:
        if site == "up":
            function = lambda r, n, p, q=permutation: mlp_from_prefix(
                r, n, p, q, None
            )
        else:
            function = lambda r, n, p, q=permutation: mlp_from_prefix(
                r, n, p, None, q
            )
        executable = compile_one(
            f"{site}_permutation_{permutation}",
            function,
            residual,
            normalized,
            params,
        )
        output = executable(residual, normalized, params)
        jax.block_until_ready(output)
        error = bench.device_error(output, reference)
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(output))))
        record = {
            "kind": "site_accuracy",
            "site": site,
            "permutation": permutation,
            "finite": finite,
            "layer_output_vs_native": error,
        }
        emit(record)
        records.append(record)
        del output, executable
        gc.collect()
        jax.clear_caches()
    return min(
        records,
        key=lambda record: (
            not record["finite"],
            record["layer_output_vs_native"]["l2_relative"],
            record["layer_output_vs_native"]["maxnorm_relative"],
        ),
    )["permutation"]


def bootstrap_delta_interval(reference, candidate, seed):
    deltas = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(deltas, size=(20000, len(deltas)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(deltas)), float(low), float(high)


def time_finalists(executables, x, params):
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

    reference_mean = statistics.fmean(samples["matched_native"])
    records = {}
    for index, name in enumerate(names):
        values = samples[name]
        mean_ms = statistics.fmean(values)
        delta, low, high = bootstrap_delta_interval(
            samples["matched_native"], values, 20260818 + index
        )
        record = {
            "kind": "performance",
            "name": name,
            "mean_ms": mean_ms,
            "samples_ms": values,
            "speedup_vs_matched_native": reference_mean / mean_ms,
            "candidate_minus_matched_mean_ms": delta,
            "candidate_minus_matched_bootstrap_95ci_ms": [low, high],
            "statistically_faster": name != "matched_native" and high < 0.0,
        }
        emit(record)
        records[name] = record
    return records


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
        "permutations": list(PERMUTATIONS),
        "formula_variant": FORMULA_VARIANT,
        "contract": "same seven products and existing layer workload/gate",
    })

    x, params = layer_bench.load_inputs()
    prefix_executable = compile_one("native_attention_prefix", attention_prefix, x, params)
    residual, normalized = prefix_executable(x, params)
    reference_executable = compile_one(
        "native_mlp", mlp_from_prefix, residual, normalized, params, None, None
    )
    reference = reference_executable(residual, normalized, params, None, None)
    jax.block_until_ready(reference)

    best_up = screen_site("up", residual, normalized, params, reference)
    best_down = screen_site("down", residual, normalized, params, reference)
    emit({"kind": "selection", "best_up": best_up, "best_down": best_down})

    del prefix_executable, reference_executable
    jax.clear_caches()
    finalists = {
        "matched_native": compile_one(
            "matched_native", make_layer(None, None), x, params
        ),
        "unpermuted": compile_one(
            "unpermuted", make_layer(0, 0), x, params
        ),
        "selected": compile_one(
            "selected", make_layer(best_up, best_down), x, params
        ),
    }
    outputs = {name: executable(x, params) for name, executable in finalists.items()}
    jax.block_until_ready(outputs)
    errors = {}
    for name, output in outputs.items():
        error = bench.device_error(output, outputs["matched_native"])
        errors[name] = error
        emit({"kind": "finalist_accuracy", "name": name, "vs_native": error})
    performance = time_finalists(finalists, x, params)

    improves_l2 = (
        errors["selected"]["l2_relative"]
        < errors["unpermuted"]["l2_relative"]
    )
    retains_speed = performance["selected"]["statistically_faster"]
    emit({
        "kind": "final",
        "success": improves_l2 and retains_speed,
        "best_up": best_up,
        "best_down": best_down,
        "improves_l2": improves_l2,
        "retains_speed": retains_speed,
    })
    if not (improves_l2 and retains_speed):
        raise RuntimeError("no block permutation retained speed while reducing error")


if __name__ == "__main__":
    main()
