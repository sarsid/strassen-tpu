"""Fixed-budget product-aware tile tuning for the dense Qwen3 scaling study.

One enumeration rule and one measurement budget apply identically to every
dense Qwen3 model (8B, 14B, 32B), so no study point receives more tuning
effort than another.  This is a screening run, not a registered comparison:
it selects one product-aware Strassen tile and one blocked-cubic tile at the
model's synthetic gate/up+SwiGLU geometry, and those choices are then pinned
by wrapper scripts into the registered synthetic, real-layer, and streamed
harnesses.

Rules fixed before any run:

* Candidates are every ``(bm, bn, bk)`` in ``{1024, 2048} x {1024, 2048,
  2560, 3072} x {512, 1024}`` that divides the model geometry, satisfies v5e
  tile alignment, admits the paired gate/up column layout
  (``intermediate % (bn // 2) == 0``), and fits an analytic VMEM estimate in
  the fixed 47 MiB Strassen kernel budget (48 MiB for cubic, whose output
  tile is twice as wide).
* Every surviving candidate gets exactly 2 warmups and 5 timed samples per
  arm.  Lowering or compile failures are recorded in-band and skip only that
  candidate/arm.
* Winner per arm is the lowest 5-sample mean; ties break toward the earlier
  candidate in enumeration order.  No re-timing after seeing results.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

SCOPED_VMEM_KIB = int(os.environ.get("QWEN3_SCOPED_VMEM_KIB", "49152"))
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_use_enhanced_launch_barrier=true "
    f"--xla_tpu_scoped_vmem_limit_kib={SCOPED_VMEM_KIB}")

import jax
import jax.numpy as jnp

import benchmark_common as common
import benchmark_cubic_control as cubic
import benchmark_qwen3_32b_layer as layer
import mosaic_compat
import strassen_pallas as sp


TOKENS = layer.TOKENS
MODEL_DIM = layer.MODEL_DIM
INTERMEDIATE = layer.INTERMEDIATE_DIM
WIDTH = 2 * INTERMEDIATE
def _tile_axis(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(token) for token in raw.replace(",", " ").split())


STRASSEN_LIMIT = int(os.environ.get("QWEN3_STRASSEN_LIMIT_MIB", "47")) * 1024 * 1024
CUBIC_LIMIT = int(os.environ.get("QWEN3_CUBIC_LIMIT_MIB", "48")) * 1024 * 1024
TUNE_WARMUPS, TUNE_RUNS = 2, 5
BM_CANDIDATES = _tile_axis("QWEN3_TILE_BM", (1024, 2048))
BN_CANDIDATES = _tile_axis("QWEN3_TILE_BN", (1024, 2048, 2560, 3072))
BK_CANDIDATES = _tile_axis("QWEN3_TILE_BK", (512, 1024))
SEED = 20260907
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
OUTPUT = Path(
    f"/content/results/strassen_qwen3_{layer.MODEL_NAME}"
    f"_scaling_tiles{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_SCALING_TILES_JSON " + line, flush=True)


def strassen_vmem_estimate(bm, bn, bk):
    """Bytes: four FP32 quadrant accumulators, double-buffered BF16
    operands, and the double-buffered half-width BF16 SwiGLU output tile."""
    accumulators = 4 * bm * bn
    output = 2 * bm * (bn // 2) * 2
    operands = 2 * (bm * bk + bk * bn) * 2
    return accumulators + output + operands


def cubic_vmem_estimate(bm, bn, bk):
    """As above but with quadrant scratch and a full-width output tile."""
    accumulators = 4 * bm * bn
    output = 2 * bm * bn * 2
    operands = 2 * (bm * bk + bk * bn) * 2
    return accumulators + output + operands


def enumerate_candidates():
    kept, filtered = [], []
    for bm in BM_CANDIDATES:
        for bn in BN_CANDIDATES:
            for bk in BK_CANDIDATES:
                reasons = []
                if TOKENS % bm:
                    reasons.append(f"tokens {TOKENS} % bm")
                if WIDTH % bn:
                    reasons.append(f"width {WIDTH} % bn")
                elif INTERMEDIATE % (bn // 2):
                    reasons.append(f"intermediate {INTERMEDIATE} % (bn//2)")
                if MODEL_DIM % bk:
                    reasons.append(f"model_dim {MODEL_DIM} % bk")
                if bm % 16 or bn % 256 or bk % 256:
                    reasons.append("v5e tile alignment")
                estimate = strassen_vmem_estimate(bm, bn, bk)
                if not reasons and estimate > STRASSEN_LIMIT:
                    reasons.append(
                        f"strassen vmem estimate {estimate} > {STRASSEN_LIMIT}")
                entry = {
                    "tile": [bm, bn, bk],
                    "strassen_vmem_estimate": estimate,
                    "cubic_vmem_estimate": cubic_vmem_estimate(bm, bn, bk),
                }
                if reasons:
                    entry["filtered_because"] = reasons
                    filtered.append(entry)
                else:
                    kept.append(entry)
    return kept, filtered


def activate(full):
    gate, up = jnp.split(full, 2, axis=-1)
    return (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def time_function(function, args):
    lowered = jax.jit(function).lower(*args)
    executable = lowered.compile()
    for _ in range(TUNE_WARMUPS):
        jax.block_until_ready(executable(*args))
    values = []
    for _ in range(TUNE_RUNS):
        started_ns = time.perf_counter_ns()
        jax.block_until_ready(executable(*args))
        values.append((time.perf_counter_ns() - started_ns) / 1e6)
    del executable, lowered
    return values


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    kept, filtered = enumerate_candidates()
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "shape": [TOKENS, MODEL_DIM, WIDTH],
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "candidates": kept, "filtered": filtered,
        "budget": {"warmups": TUNE_WARMUPS, "runs": TUNE_RUNS,
                   "arms": ["product_strassen", "cubic"]},
        "seed": SEED,
        "rule": (
            "identical enumeration and budget for every dense model; winner "
            "per arm is the lowest 5-sample mean; failures recorded in-band "
            "skip only that candidate/arm; winners are pinned into the "
            "registered harnesses and requalified at full protocol there"),
        "scope": "tile screening only; not a registered comparison",
    })
    if not kept:
        emit({"kind": "verdict", "passes": False,
              "reason": "no feasible tile candidate"})
        return

    x, weight = common.device_uniform_inputs((TOKENS, MODEL_DIM, WIDTH), SEED)
    laid_cache = {}
    for entry in kept:
        bn = entry["tile"][1]
        if bn not in laid_cache:
            laid_cache[bn] = sp.swiglu_weight_layout(weight, bn)
    jax.block_until_ready((x, weight, tuple(laid_cache.values())))

    xla_values = time_function(
        lambda a, w: activate(sp.native_matmul(a, w)), (x, weight))
    emit({"kind": "tune", "arm": "regular_xla", "tile": None,
          "mean_ms": statistics.fmean(xla_values), "samples_ms": xla_values,
          "note": "context only; not part of the tile decision"})

    results = {"product_strassen": [], "cubic": []}
    for entry in kept:
        bm, bn, bk = entry["tile"]
        laid = laid_cache[bn]

        def product_fn(a, wl, bm=bm, bn=bn, bk=bk):
            return sp.strassen_matmul(
                a, wl, bm=bm, bn=bn, bk=bk,
                epilogue="swiglu", interleave_products=True,
                product_aware_swiglu=True,
                vmem_limit_bytes=STRASSEN_LIMIT)

        def cubic_fn(a, w, bm=bm, bn=bn, bk=bk):
            return activate(cubic.cubic_matmul(
                a, w, variant="blocked", bm=bm, bn=bn, bk=bk,
                vmem_limit_bytes=CUBIC_LIMIT))

        plans = [("product_strassen", product_fn, (x, laid))]
        if entry["cubic_vmem_estimate"] <= CUBIC_LIMIT:
            plans.append(("cubic", cubic_fn, (x, weight)))
        else:
            emit({"kind": "tune_skip", "arm": "cubic", "tile": [bm, bn, bk],
                  "reason": f"cubic vmem estimate exceeds {CUBIC_LIMIT // (1024 * 1024)} MiB"})
        for arm, function, args in plans:
            try:
                values = time_function(function, args)
            except Exception as error:
                emit({
                    "kind": "tune_failure", "arm": arm, "tile": [bm, bn, bk],
                    "error_type": type(error).__name__,
                    "error": str(error).splitlines()[-1][:2000],
                })
            else:
                mean = statistics.fmean(values)
                results[arm].append(([bm, bn, bk], mean))
                emit({"kind": "tune", "arm": arm, "tile": [bm, bn, bk],
                      "mean_ms": mean, "samples_ms": values})
            gc.collect()
            jax.clear_caches()

    chosen = {}
    for arm, rows in results.items():
        if rows:
            tile, mean = min(rows, key=lambda row: row[1])
            chosen[arm] = {"tile": tile, "mean_ms": mean}
    passes = "product_strassen" in chosen and "cubic" in chosen
    emit({
        "kind": "chosen", "model": layer.MODEL_NAME, "winners": chosen,
        "xla_context_mean_ms": statistics.fmean(xla_values),
    })
    emit({"kind": "verdict", "passes": passes,
          "scope": "tile screening; winners feed the registered harnesses"})


if __name__ == "__main__":
    main()
