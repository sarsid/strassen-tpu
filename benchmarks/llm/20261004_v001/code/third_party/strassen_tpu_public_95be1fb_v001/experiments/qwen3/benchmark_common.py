"""Shared TPU benchmark inputs, accuracy metrics, and timing utilities."""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


WARMUPS = 10
RUNS = 20


def native_bf16_gemm(a, b):
    """Matched cubic baseline for BF16 inputs and output."""
    return jnp.matmul(
        a,
        b,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)


def high_precision_reference(a, b):
    """FP32 HIGHEST reference used by the numerical gates."""
    return jnp.matmul(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )


@jax.jit
def _device_error_values(got, reference):
    got = got.astype(jnp.float32)
    reference = reference.astype(jnp.float32)
    delta = got - reference
    tiny = jnp.finfo(jnp.float32).tiny
    return (
        jnp.max(jnp.abs(delta)),
        jnp.sqrt(jnp.mean(delta * delta)),
        jnp.max(jnp.abs(delta)) / jnp.maximum(jnp.max(jnp.abs(reference)), tiny),
        jnp.linalg.norm(delta) / jnp.maximum(jnp.linalg.norm(reference), tiny),
    )


def device_error(got, reference):
    """Return scalar error metrics without transferring full matrices."""
    max_abs, rmse, maxnorm_relative, l2_relative = jax.device_get(
        _device_error_values(got, reference)
    )
    return {
        "max_abs": float(max_abs),
        "rmse": float(rmse),
        "maxnorm_relative": float(maxnorm_relative),
        "l2_relative": float(l2_relative),
    }


def host_uniform_inputs(shape, seed):
    """Generate deterministic FP32 uniform [-1, 1] host inputs."""
    m, k, n = shape
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, (m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (k, n)).astype(np.float32)
    return a, b


def device_uniform_inputs(shape, seed, dtype=jnp.bfloat16):
    a_host, b_host = host_uniform_inputs(shape, seed)
    a = jnp.asarray(a_host, dtype=dtype)
    b = jnp.asarray(b_host, dtype=dtype)
    jax.block_until_ready((a, b))
    return a, b


def summarize_samples(samples_ms, shape):
    mean_ms = statistics.fmean(samples_ms)
    m, k, n = shape
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
        "mean_tflops": 2 * m * k * n / (mean_ms / 1e3) / 1e12,
        "samples_ms": samples_ms,
    }


def interleaved_timings(
    executables, a, b, shape, *, warmups=WARMUPS, runs=RUNS
):
    """Rotate execution order to limit thermal and runtime-order bias."""
    names = tuple(executables)
    for warmup in range(warmups):
        offset = warmup % len(names)
        for name in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[name](a, b))

    samples = {name: [] for name in names}
    outputs = {}
    for run in range(runs):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
        if run % 2:
            order = tuple(reversed(order))
        for name in order:
            started = time.perf_counter_ns()
            outputs[name] = executables[name](a, b)
            jax.block_until_ready(outputs[name])
            samples[name].append((time.perf_counter_ns() - started) / 1e6)

    timings = {
        name: summarize_samples(samples[name], shape) for name in names
    }
    return timings, outputs
