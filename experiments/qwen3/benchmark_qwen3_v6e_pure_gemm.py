"""Pure-GEMM v6e qualification at the Qwen3-32B gate/up geometry.

This benchmark deliberately excludes SwiGLU, product-aware finalization,
residuals, and every other epilogue.  All arms implement exactly

    BF16[M,K] @ BF16[K,N] -> BF16[M,N], with FP32 accumulation.

A small, equal-budget tile screen chooses one Strassen tile and one blocked
cubic Pallas tile.  The fixed winners are then recompiled and compared in an
interleaved confirmation against native XLA.  A second cubic arm uses the
Strassen winner's exact tile, separating algorithm from tile selection.
"""

from __future__ import annotations

import functools
import gc
import json
import math
import os
from pathlib import Path
import statistics


# The large v6e candidates require a raised Mosaic *kernel* limit, which is
# independent of the XLA scoped-vmem flag: the kernel can take 104 MiB while
# XLA keeps the 48 MiB it prefers.  Raising the flag as well is measurably
# expensive -- a paired same-chip comparison at one tile put the 128 MiB flag
# at 37.1% slower for native XLA and 40.8% for our own layer -- and this
# benchmark exists to attribute the rank-7 saving cleanly, so a handicapped
# XLA baseline would inflate exactly the number it is trying to measure.
# Override QWEN3_SCOPED_VMEM_KIB to run the pair and confirm the effect here.
_SCOPED_VMEM_KIB = os.environ.get("QWEN3_SCOPED_VMEM_KIB", "49152")
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_use_enhanced_launch_barrier=true "
    f"--xla_tpu_scoped_vmem_limit_kib={_SCOPED_VMEM_KIB}",
)
os.environ.setdefault("STRASSEN_VMEM_PROFILE", "compact16")

import jax

import benchmark_common as common
import benchmark_cubic_control as cubic
import mosaic_compat
import strassen_pallas as sp


SHAPE = (8192, 5120, 51200)
SEED = 20260902
SCREEN_WARMUPS = 2
SCREEN_RUNS = 5
CONFIRM_WARMUPS = 5
CONFIRM_RUNS = 30
KERNEL_LIMIT_BYTES = int(
    os.environ.get("QWEN3_STRASSEN_LIMIT_MIB", "104")) * 1024 * 1024
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
OUTPUT = Path(
    f"/content/results/strassen_qwen3_32b_v6e_pure_gemm{SUFFIX}.jsonl"
)

# All candidates divide the real geometry and keep every half-tile aligned to
# v6e's 256x256 MXU.  The list was declared before the run and is identical
# for Strassen and cubic Pallas.
CANDIDATES = (
    (1024, 1024, 2560),
    (1024, 1024, 5120),
    (1024, 2560, 2560),
    (1024, 2560, 5120),
    (2048, 1024, 2560),
    (2048, 1024, 5120),
    (2048, 2560, 2560),
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_V6E_PURE_GEMM_JSON " + line, flush=True)


def make_fn(arm, tile):
    bm, bn, bk = tile
    if arm == "strassen":
        return functools.partial(
            sp.strassen_matmul,
            bm=bm,
            bn=bn,
            bk=bk,
            interleave_products=True,
            vmem_limit_bytes=KERNEL_LIMIT_BYTES,
        )
    if arm == "cubic":
        return functools.partial(
            cubic.cubic_matmul,
            variant="blocked",
            bm=bm,
            bn=bn,
            bk=bk,
            vmem_limit_bytes=KERNEL_LIMIT_BYTES,
        )
    raise ValueError(arm)


def compile_fn(fn, a, b):
    return jax.jit(fn).lower(a, b).compile()


def screen(a, b):
    rows = {"strassen": [], "cubic": []}
    for tile in CANDIDATES:
        for arm in ("strassen", "cubic"):
            try:
                executable = compile_fn(make_fn(arm, tile), a, b)
                timing, _ = common.interleaved_timings(
                    {arm: executable}, a, b, SHAPE,
                    warmups=SCREEN_WARMUPS, runs=SCREEN_RUNS,
                )
            except Exception as error:
                emit({
                    "kind": "screen_failure",
                    "arm": arm,
                    "tile": list(tile),
                    "error_type": type(error).__name__,
                    "error": str(error).splitlines()[-1][:1000],
                })
            else:
                mean = timing[arm]["mean_ms"]
                rows[arm].append((tile, mean))
                emit({
                    "kind": "screen",
                    "arm": arm,
                    "tile": list(tile),
                    "timing": timing[arm],
                })
                del executable
            gc.collect()
            jax.clear_caches()
    winners = {
        arm: min(values, key=lambda item: item[1])[0]
        for arm, values in rows.items() if values
    }
    emit({
        "kind": "screen_winners",
        "tiles": {arm: list(tile) for arm, tile in winners.items()},
        "scope": "selection only; confirmation follows with fixed tiles",
    })
    if set(winners) != {"strassen", "cubic"}:
        raise RuntimeError("tile screen did not produce both winners")
    return winners


def paired_interval(reference, candidate):
    deltas = [c - r for r, c in zip(reference, candidate)]
    mean = statistics.fmean(deltas)
    half = 2.045 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {"mean_ms": mean, "ci95_ms": [mean - half, mean + half]}


def confirm(a, b, winners):
    strassen_tile = winners["strassen"]
    functions = {
        "regular_xla": common.native_bf16_gemm,
        "cubic_best": make_fn("cubic", winners["cubic"]),
        "cubic_matched": make_fn("cubic", strassen_tile),
        "strassen": make_fn("strassen", strassen_tile),
    }
    executables = {
        name: compile_fn(fn, a, b) for name, fn in functions.items()
    }
    timings, outputs = common.interleaved_timings(
        executables, a, b, SHAPE,
        warmups=CONFIRM_WARMUPS, runs=CONFIRM_RUNS,
    )
    reference = outputs["regular_xla"]
    errors = {
        name: common.device_error(value, reference)
        for name, value in outputs.items() if name != "regular_xla"
    }
    comparisons = {
        f"strassen_minus_{baseline}": paired_interval(
            timings[baseline]["samples_ms"],
            timings["strassen"]["samples_ms"],
        )
        for baseline in ("regular_xla", "cubic_best", "cubic_matched")
    }
    speedups = {
        baseline: timings[baseline]["mean_ms"] / timings["strassen"]["mean_ms"]
        for baseline in ("regular_xla", "cubic_best", "cubic_matched")
    }
    all_finite = all(
        math.isfinite(metric)
        for arm in errors.values() for metric in arm.values()
    )
    passes = bool(
        all_finite
        and errors["strassen"]["maxnorm_relative"] <= 0.01
        and comparisons["strassen_minus_regular_xla"]["ci95_ms"][1] < 0
        and comparisons["strassen_minus_cubic_best"]["ci95_ms"][1] < 0
        and comparisons["strassen_minus_cubic_matched"]["ci95_ms"][1] < 0
    )
    emit({
        "kind": "confirmation",
        "shape": list(SHAPE),
        "contract": "BF16 inputs, FP32 accumulation, BF16 output; no epilogue",
        "tiles": {
            "strassen": list(strassen_tile),
            "cubic_best": list(winners["cubic"]),
            "cubic_matched": list(strassen_tile),
        },
        "timings": timings,
        "errors_vs_xla": errors,
        "speedup_strassen_vs": speedups,
        "paired_comparisons": comparisons,
        "passes": passes,
    })
    emit({
        "kind": "verdict",
        "passes": passes,
        "scope": ("pure Qwen3-32B gate/up GEMM on one "
                  f"{jax.devices()[0].device_kind} chip"),
    })


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "shape": list(SHAPE),
        "seed": SEED,
        "candidates": [list(tile) for tile in CANDIDATES],
        "screen_budget": {
            "warmups": SCREEN_WARMUPS, "runs": SCREEN_RUNS,
            "same_candidates_per_pallas_arm": True,
        },
        "confirmation_budget": {
            "warmups": CONFIRM_WARMUPS, "runs": CONFIRM_RUNS,
            "interleaved": True,
        },
        "kernel_limit_mib": KERNEL_LIMIT_BYTES // (1024 * 1024),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    })
    a, b = common.device_uniform_inputs(SHAPE, SEED)
    jax.block_until_ready((a, b))
    winners = screen(a, b)
    confirm(a, b, winners)


if __name__ == "__main__":
    main()
