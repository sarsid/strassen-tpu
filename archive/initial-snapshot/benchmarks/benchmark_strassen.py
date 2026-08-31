"""Accuracy and performance gate for Strassen Pallas versus native XLA GEMM.

Both timed implementations receive identical BF16 inputs and return BF16.
Native XLA and Strassen both request FP32 dot accumulation. Compilation, input
creation, accuracy checks, and host transfers are outside the timed region.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
from pathlib import Path

# Import the kernel first so its tested libtpu scoped-VMEM option is installed
# before JAX initializes the TPU backend.
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as bench

# Colab CLI executions share a persistent Jupyter kernel. Always test the
# just-uploaded source instead of a module cached by an earlier iteration.
sp = importlib.reload(sp)


def compile_pair(a, b, *, bm, bn, bk):
    native = jax.jit(bench.native_bf16_gemm).lower(a, b).compile()
    strassen = jax.jit(functools.partial(
        sp.strassen_matmul,
        bm=bm,
        bn=bn,
        bk=bk,
        interpret=False,
        vmem_limit_bytes=sp.TUNED_VMEM_LIMIT_BYTES,
    )).lower(a, b).compile()
    return native, strassen


def accuracy_gate(size, *, bm, bn, bk):
    rng = np.random.default_rng(20260815)
    a = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(size, size)), dtype=jnp.bfloat16
    )
    b = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(size, size)), dtype=jnp.bfloat16
    )
    native, strassen = compile_pair(a, b, bm=bm, bn=bn, bk=bk)
    reference = jax.jit(bench.high_precision_reference)(a, b)
    native_out = native(a, b)
    strassen_out = strassen(a, b)
    jax.block_until_ready((reference, native_out, strassen_out))

    native_error = bench.device_error(native_out, reference)
    strassen_error = bench.device_error(strassen_out, reference)
    native_error["finite"] = all(math.isfinite(v) for v in native_error.values())
    strassen_error["finite"] = all(
        math.isfinite(v) for v in strassen_error.values()
    )
    max_ratio = strassen_error["max_abs"] / max(
        native_error["max_abs"], np.finfo(np.float32).tiny
    )
    rmse_ratio = strassen_error["rmse"] / max(
        native_error["rmse"], np.finfo(np.float32).tiny
    )
    passed = bool(
        strassen_error["finite"]
        and strassen_error["maxnorm_relative"] <= 0.01
        and max_ratio <= 4.0
        and rmse_ratio <= 4.0
    )
    return {
        "kind": "accuracy",
        "size": size,
        "native": native_error,
        "strassen": strassen_error,
        "strassen_over_native_max_abs": max_ratio,
        "strassen_over_native_rmse": rmse_ratio,
        "thresholds": {
            "strassen_maxnorm_relative": 0.01,
            "strassen_over_native_max_abs": 4.0,
            "strassen_over_native_rmse": 4.0,
        },
        "passed": passed,
    }


def benchmark_size(size, *, bm, bn, bk, warmups, runs):
    a = jnp.full((size, size), 0.5, dtype=jnp.bfloat16)
    b = jnp.full((size, size), -0.25, dtype=jnp.bfloat16)
    jax.block_until_ready((a, b))
    native, strassen = compile_pair(a, b, bm=bm, bn=bn, bk=bk)
    timings, outputs = bench.interleaved_timings(
        {"native": native, "strassen": strassen},
        a,
        b,
        (size, size, size),
        warmups=warmups,
        runs=runs,
    )
    native_stats = timings["native"]
    strassen_stats = timings["strassen"]
    speedup = native_stats["mean_ms"] / strassen_stats["mean_ms"]
    return {
        "kind": "performance",
        "size": size,
        "native": native_stats,
        "strassen": strassen_stats,
        "speedup": speedup,
        "matched_or_better": speedup >= 1.0,
        "native_c00": float(outputs["native"][0, 0]),
        "strassen_c00": float(outputs["strassen"][0, 0]),
        "expected_c00": -size / 8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8192,12288,16384")
    parser.add_argument("--accuracy-size", type=int, default=2048)
    parser.add_argument("--bm", type=int, default=sp.TUNED_BM)
    parser.add_argument("--bn", type=int, default=sp.TUNED_BN)
    parser.add_argument("--bk", type=int, default=sp.TUNED_BK)
    parser.add_argument("--warmups", type=int, default=bench.WARMUPS)
    parser.add_argument("--runs", type=int, default=bench.RUNS)
    parser.add_argument(
        "--output",
        default="/content/results/strassen_v5e_tuned.jsonl",
    )
    args, _ = parser.parse_known_args()

    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    sizes = tuple(int(value) for value in args.sizes.split(","))
    records = [{
        "kind": "metadata",
        "jax": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "vmem_profile": sp.VMEM_PROFILE,
        "mosaic_ir_v7_compat": sp.MOSAIC_IR_V7_COMPAT,
        "tile": [args.bm, args.bn, args.bk],
        "algorithm": "classical",
        "warmups": args.warmups,
        "runs": args.runs,
    }]

    accuracy = accuracy_gate(
        args.accuracy_size,
        bm=args.bm,
        bn=args.bn,
        bk=args.bk,
    )
    records.append(accuracy)
    print("STRASSEN_JSON " + json.dumps(accuracy, sort_keys=True), flush=True)
    if not accuracy["passed"]:
        raise RuntimeError("accuracy gate failed; performance run aborted")

    for size in sizes:
        result = benchmark_size(
            size,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            warmups=args.warmups,
            runs=args.runs,
        )
        records.append(result)
        print("STRASSEN_JSON " + json.dumps(result, sort_keys=True), flush=True)
        print(
            f"N={size}: native {result['native']['mean_ms']:.4f} ms, "
            f"Strassen {result['strassen']['mean_ms']:.4f} ms, "
            f"speedup {result['speedup']:.3f}x",
            flush=True,
        )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
