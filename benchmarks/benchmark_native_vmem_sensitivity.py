"""Does the scoped-VMEM flag itself perturb the native XLA baseline?

Every promoted comparison runs native XLA inside a process whose
``--xla_tpu_scoped_vmem_limit_kib`` flag was chosen for the Strassen kernel.
If that flag changes native's own compiled performance, the promoted speedups
are measured against a Strassen-configured baseline rather than native's
default configuration.

This script times native BF16 GEMM only, on the five promoted shapes. Run it
twice in separate processes:

    CONTROL_SCOPED_VMEM_KIB=       -> results/native_vmem_default.jsonl
    CONTROL_SCOPED_VMEM_KIB=49152  -> results/native_vmem_scoped_48m.jsonl

It deliberately does not import ``strassen_pallas`` so the flag state is
exactly what the environment variable requests.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


SCOPED_KIB = os.environ.get("CONTROL_SCOPED_VMEM_KIB", "").strip()
MODE = f"scoped_{int(SCOPED_KIB) // 1024}m" if SCOPED_KIB else "default"
if SCOPED_KIB:
    existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
    flag = f"--xla_tpu_scoped_vmem_limit_kib={SCOPED_KIB}"
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import jax
import jax.numpy as jnp

import benchmark_common as bench


OUTPUT = Path(f"/content/results/native_vmem_{MODE}.jsonl")
PERFORMANCE_CASES = (
    ("square_8192", (8192, 8192, 8192)),
    ("paper_rectangle", (8192, 4096, 8192)),
    ("qkv_projection", (8192, 4096, 12288)),
    ("mlp_up", (8192, 4096, 28672)),
    ("mlp_down", (8192, 14336, 4096)),
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("NATIVE_VMEM_JSON " + line, flush=True)


def native_matmul(a, b):
    return jnp.matmul(
        a,
        b,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    ).astype(a.dtype)


def performance_case(name, shape):
    m, k, n = shape
    a = jnp.full((m, k), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((k, n), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    started = time.perf_counter()
    executable = jax.jit(native_matmul).lower(a, b).compile()
    compile_s = time.perf_counter() - started
    timings, outputs = bench.interleaved_timings(
        {"native": executable}, a, b, shape
    )
    expected = -k / 8
    sentinel = float(outputs["native"][0, 0])
    emit({
        "kind": "performance",
        "mode": MODE,
        "name": name,
        "shape": shape,
        "compile_s": compile_s,
        "timings": timings,
        "sentinel": sentinel,
        "expected_sentinel": expected,
        "passes": sentinel == expected,
    })
    return sentinel == expected


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "mode": MODE,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "warmups": bench.WARMUPS,
        "runs": bench.RUNS,
    })
    results = {
        name: performance_case(name, shape)
        for name, shape in PERFORMANCE_CASES
    }
    success = all(results.values())
    emit({"kind": "final", "mode": MODE, "success": success})
    if not success:
        raise RuntimeError(results)


if __name__ == "__main__":
    main()
