"""Controlled end-to-end Transformer-block evidence for tuned TPU GEMMs.

The benchmark holds a Phi-like prefill block fixed and changes only which of
three qualified projection GEMMs use the Strassen dispatcher.  All variants
run in one process under the max48 VMEM profile and use identical resident
BF16 inputs and weights.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import NamedTuple


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as bench


OUTPUT = Path("/content/results/strassen_model_block.jsonl")
BATCH = 8
SEQUENCE = 1024
TOKENS = BATCH * SEQUENCE
MODEL_DIM = 4096
HEADS = 32
HEAD_DIM = MODEL_DIM // HEADS
INTERMEDIATE_DIM = 14336
QKV_DIM = 3 * MODEL_DIM
GATE_UP_DIM = 2 * INTERMEDIATE_DIM
WARMUPS = 10
RUNS = 20

# Each non-native policy changes exactly the named qualified projection(s).
POLICIES = {
    "native": frozenset(),
    "qkv": frozenset({"qkv"}),
    "mlp_up": frozenset({"mlp_up"}),
    "mlp_down": frozenset({"mlp_down"}),
    "all": frozenset({"qkv", "mlp_up", "mlp_down"}),
}


class Parameters(NamedTuple):
    attention_norm: jax.Array
    qkv: jax.Array
    attention_output: jax.Array
    mlp_norm: jax.Array
    gate_up: jax.Array
    mlp_down: jax.Array
    rope_cos: jax.Array
    rope_sin: jax.Array


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("MODEL_BLOCK_JSON " + line, flush=True)


def _random_bf16(key, shape, fan_in):
    bound = math.sqrt(3.0 / fan_in)
    return jax.random.uniform(
        key, shape, dtype=jnp.bfloat16, minval=-bound, maxval=bound
    )


def make_inputs():
    keys = jax.random.split(jax.random.key(20260816), 5)
    x = jax.random.normal(keys[0], (TOKENS, MODEL_DIM), dtype=jnp.bfloat16)
    positions = jnp.arange(SEQUENCE, dtype=jnp.float32)[:, None]
    frequencies = 1.0 / (
        10000.0
        ** (jnp.arange(HEAD_DIM // 2, dtype=jnp.float32) * 2 / HEAD_DIM)
    )
    angles = positions * frequencies[None, :]
    params = Parameters(
        attention_norm=jnp.ones((MODEL_DIM,), dtype=jnp.bfloat16),
        qkv=_random_bf16(keys[1], (MODEL_DIM, QKV_DIM), MODEL_DIM),
        attention_output=_random_bf16(
            keys[2], (MODEL_DIM, MODEL_DIM), MODEL_DIM
        ),
        mlp_norm=jnp.ones((MODEL_DIM,), dtype=jnp.bfloat16),
        gate_up=_random_bf16(
            keys[3], (MODEL_DIM, GATE_UP_DIM), MODEL_DIM
        ),
        mlp_down=_random_bf16(
            keys[4], (INTERMEDIATE_DIM, MODEL_DIM), INTERMEDIATE_DIM
        ),
        rope_cos=jnp.cos(angles).astype(jnp.bfloat16),
        rope_sin=jnp.sin(angles).astype(jnp.bfloat16),
    )
    jax.block_until_ready((x, params))
    return x, params


def rms_norm(x, scale):
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + 1e-5) * scale).astype(jnp.bfloat16)


def apply_rope(x, cos, sin):
    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    cos = cos[None, :, None, :].astype(jnp.float32)
    sin = sin[None, :, None, :].astype(jnp.float32)
    return jnp.concatenate(
        (first * cos - second * sin, first * sin + second * cos), axis=-1
    ).astype(jnp.bfloat16)


def gemm(lhs, rhs, use_strassen):
    # Symmetric barriers isolate every GEMM boundary in every policy.
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    out = sp.tuned_matmul(lhs, rhs) if use_strassen else sp.native_matmul(lhs, rhs)
    return jax.lax.optimization_barrier(out)


def make_block(policy_name):
    selected = POLICIES[policy_name]

    def block(x, params):
        normalized = rms_norm(x, params.attention_norm)
        qkv = gemm(normalized, params.qkv, "qkv" in selected)
        qkv = qkv.reshape(BATCH, SEQUENCE, 3, HEADS, HEAD_DIM)
        query = apply_rope(qkv[:, :, 0], params.rope_cos, params.rope_sin)
        key = apply_rope(qkv[:, :, 1], params.rope_cos, params.rope_sin)
        value = qkv[:, :, 2]
        attended = jax.nn.dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            implementation="xla",
        )
        attended = attended.reshape(TOKENS, MODEL_DIM).astype(jnp.bfloat16)
        attention_output = gemm(attended, params.attention_output, False)
        residual = (
            x.astype(jnp.float32) + attention_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

        normalized = rms_norm(residual, params.mlp_norm)
        gate_up = gemm(normalized, params.gate_up, "mlp_up" in selected)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(activated, params.mlp_down, "mlp_down" in selected)
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return block


def lowered_audit(lowered):
    text = lowered.as_text()
    terms = (
        "stablehlo.dot_general",
        "stablehlo.optimization_barrier",
        "stablehlo.custom_call",
        "mosaic_tpu",
        "pallas_call",
    )
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "term_counts": {term: text.count(term) for term in terms},
    }


def compile_policies(x, params):
    executables = {}
    for name in POLICIES:
        started = time.perf_counter()
        lowered = jax.jit(make_block(name)).lower(x, params)
        audit = lowered_audit(lowered)
        executables[name] = lowered.compile()
        emit({
            "kind": "compile",
            "policy": name,
            "selected_gemms": sorted(POLICIES[name]),
            "compile_s": time.perf_counter() - started,
            "lowered_audit": audit,
        })
    return executables


def accuracy(executables, x, params):
    outputs = {
        name: executable(x, params)
        for name, executable in executables.items()
    }
    jax.block_until_ready(outputs)
    records = {}
    reference = outputs["native"]
    for name, output in outputs.items():
        error = bench.device_error(output, reference)
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(output))))
        passes = (
            finite
            and error["maxnorm_relative"] <= 0.01
            and error["l2_relative"] <= 0.01
        )
        records[name] = {"finite": finite, "vs_native": error, "passes": passes}
        emit({"kind": "accuracy", "policy": name, **records[name]})
    return records, outputs


def bootstrap_delta_interval(native_samples, candidate_samples, seed):
    deltas = np.asarray(candidate_samples) - np.asarray(native_samples)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(deltas, size=(20000, len(deltas)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(deltas)), float(low), float(high)


def timing(executables, x, params):
    names = tuple(executables)
    for warmup in range(WARMUPS):
        order = names[warmup % len(names):] + names[:warmup % len(names)]
        for name in order:
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

    native_mean = statistics.fmean(samples["native"])
    records = {}
    for index, name in enumerate(names):
        values = samples[name]
        mean_ms = statistics.fmean(values)
        delta, low, high = bootstrap_delta_interval(
            samples["native"], values, 20260816 + index
        )
        records[name] = {
            "mean_ms": mean_ms,
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "std_ms": statistics.pstdev(values),
            "samples_ms": values,
            "speedup_vs_native": native_mean / mean_ms,
            "candidate_minus_native_mean_ms": delta,
            "candidate_minus_native_bootstrap_95ci_ms": [low, high],
            "statistically_faster": name != "native" and high < 0.0,
        }
        emit({"kind": "performance", "policy": name, **records[name]})
    return records


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "profile": sp.VMEM_PROFILE,
        "tile": [sp.TUNED_BM, sp.TUNED_BN, sp.TUNED_BK],
        "contract": "BF16 inputs, FP32 GEMM accumulation, BF16 GEMM output",
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "tokens": TOKENS,
            "model_dim": MODEL_DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
        },
        "fixed_components": [
            "RMSNorm", "RoPE", "causal XLA attention", "attention output GEMM",
            "SwiGLU", "residuals", "inputs", "weights",
        ],
        "policies": {name: sorted(sites) for name, sites in POLICIES.items()},
        "warmups": WARMUPS,
        "runs": RUNS,
    })

    x, params = make_inputs()
    executables = compile_policies(x, params)
    accuracy_records, outputs = accuracy(executables, x, params)
    del outputs
    gc.collect()
    performance_records = timing(executables, x, params)
    accuracy_passes = all(record["passes"] for record in accuracy_records.values())
    primary_performance_passes = performance_records["all"]["statistically_faster"]
    success = accuracy_passes and primary_performance_passes
    emit({
        "kind": "final",
        "success": success,
        "accuracy_passes": accuracy_passes,
        "primary_performance_passes": primary_performance_passes,
    })
    if not success:
        raise RuntimeError("model-block gate failed; inspect emitted records")


if __name__ == "__main__":
    main()
