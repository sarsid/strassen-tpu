"""Native-only real-layer timing under a scoped-VMEM ceiling set by env.

Measures what the scoped-VMEM flag alone costs the *native* layer, to give
the hybrid's ceiling requirement a drift-controlled estimate: run this in
A/B/A/B alternation across kernel restarts within one session
(``run_layer_vmem_48m.py`` / ``run_layer_vmem_96m.py``), then compare the
interleaved pairs. Layer tensors are cached to runtime disk on first use so
later restarts skip the download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import benchmark_checkpoint_layer as layer_bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


MODE = os.environ.get("LAYER_VMEM_MODE", "unset")
_RESULTS_DIR = Path("/content/results")
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPEAT = str(len(list(_RESULTS_DIR.glob(f"native_layer_vmem_{MODE}_r*.jsonl"))))
OUTPUT = _RESULTS_DIR / f"native_layer_vmem_{MODE}_r{REPEAT}.jsonl"
TENSOR_CACHE = Path("/content/tensor_cache")
WARMUPS = 10
RUNS = 20


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("LAYER_VMEM_JSON " + line, flush=True)


layer_bench.emit = emit


def cached_load_inputs():
    """layer_bench.load_inputs with a runtime-disk tensor cache."""
    TENSOR_CACHE.mkdir(parents=True, exist_ok=True)
    original_tensor = layer_bench.RemoteSafetensors.tensor

    def caching_tensor(self, name):
        path = TENSOR_CACHE / (name.replace("/", "_") + ".npy")
        if path.exists():
            raw = np.load(path)
            return raw.view(ml_dtypes.bfloat16)
        array = original_tensor(self, name)
        np.save(path, array.view(np.uint16))
        return array

    layer_bench.RemoteSafetensors.tensor = caching_tensor
    try:
        return layer_bench.load_inputs()
    finally:
        layer_bench.RemoteSafetensors.tensor = original_tensor


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "mode": MODE,
        "repeat": REPEAT,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "warmups": WARMUPS,
        "runs": RUNS,
    })

    x, params = cached_load_inputs()
    layer = layer_bench.make_layer("matched_native")
    started = time.perf_counter()
    executable = jax.jit(layer).lower(x, params).compile()
    emit({"kind": "compile", "compile_s": time.perf_counter() - started})

    for _ in range(WARMUPS):
        jax.block_until_ready(executable(x, params))
    samples = []
    for _ in range(RUNS):
        begin = time.perf_counter_ns()
        jax.block_until_ready(executable(x, params))
        samples.append((time.perf_counter_ns() - begin) / 1e6)

    emit({
        "kind": "performance",
        "mode": MODE,
        "repeat": REPEAT,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "std_ms": statistics.pstdev(samples),
        "samples_ms": samples,
    })
    emit({"kind": "final", "success": True})


if __name__ == "__main__":
    main()
