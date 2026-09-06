"""Per-site tile screen for the five GEMM sites the gate/up tune never covers.

The scaling tuner already screens the fused SwiGLU kernel at the gate/up
geometry, and its Trillium winner moved to the full 5120 contraction depth.
The other five routed sites -- q, k, v, o, down -- never had a tile screen
at all: both the site screen and the full-layer harness pick their tile from
a hardcoded v5e-era heuristic that fixes bk=512 everywhere.  Those five
sites carry about 46% of the layer's GEMM work, so on a chip whose tune says
"make the contraction panel as deep as it fits", leaving them at bk=512 is
the largest untested lever.

One arm per candidate, same budget for every site, failures recorded in
band.  Winners are pinned into the full-layer harness through
QWEN3_SITE_TILES and requalified there at full protocol.
"""

from __future__ import annotations

import json
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


TOKENS, MODEL_DIM = layer.TOKENS, layer.MODEL_DIM
ATTN_DIM = layer.HEADS * layer.HEAD_DIM
KV_DIM = layer.KV_HEADS * layer.HEAD_DIM
INTERMEDIATE = layer.INTERMEDIATE_DIM

STRASSEN_LIMIT = int(
    os.environ.get("QWEN3_STRASSEN_LIMIT_MIB", "47")) * 1024 * 1024
TUNE_WARMUPS, TUNE_RUNS = 2, 5
SEED = 20260902
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(
    f"{RESULTS_DIR}/strassen_qwen3_{layer.MODEL_NAME}"
    f"_site_tiles{SUFFIX}.jsonl")

# site -> (contraction depth, output width, fused residual?)
SITES = {
    "q": (MODEL_DIM, ATTN_DIM, False),
    "k": (MODEL_DIM, KV_DIM, False),
    "v": (MODEL_DIM, KV_DIM, False),
    "o": (ATTN_DIM, MODEL_DIM, True),
    "down": (INTERMEDIATE, MODEL_DIM, True),
}
BM_CANDIDATES = (1024, 2048, 4096)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_SITE_TILES_JSON " + line, flush=True)


def vmem_estimate(bm, bn, bk):
    """Four FP32 quadrant accumulators, double-buffered BF16 operands, and
    a double-buffered BF16 output tile.  The Strassen arm's real footprint
    runs above this -- the v5e deep-bk retune OOMed tiles this model put at
    46 MiB against a 47 MiB cap -- so it prunes only, and every survivor is
    still confirmed by an actual compile."""
    return 4 * bm * bn + 2 * bm * bn * 2 + 2 * (bm * bk + bk * bn) * 2


def divisors_at_least_256(extent):
    return tuple(
        d for d in (512, 1024, 2048, 2560, 4096, 5120, 8192, 12800)
        if extent % d == 0)


def enumerate_site(depth, width):
    kept, filtered = [], []
    for bm in BM_CANDIDATES:
        for bn in divisors_at_least_256(width):
            for bk in divisors_at_least_256(depth):
                reasons = []
                if TOKENS % bm:
                    reasons.append(f"tokens {TOKENS} % bm")
                estimate = vmem_estimate(bm, bn, bk)
                if not reasons and estimate > STRASSEN_LIMIT:
                    reasons.append(f"vmem estimate {estimate} > {STRASSEN_LIMIT}")
                entry = {"tile": [bm, bn, bk], "vmem_estimate": estimate}
                if reasons:
                    entry["filtered_because"] = reasons
                    filtered.append(entry)
                else:
                    kept.append(entry)
    return kept, filtered


def time_function(function, args):
    compiled = jax.jit(function).lower(*args).compile()
    for _ in range(TUNE_WARMUPS):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(TUNE_RUNS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main():
    emit({
        "kind": "metadata",
        "model": layer.MODEL_NAME,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "strassen_limit_bytes": STRASSEN_LIMIT,
        "sites": {name: {"depth": d, "width": w, "fused_residual": r}
                  for name, (d, w, r) in SITES.items()},
        "budget": {"warmups": TUNE_WARMUPS, "runs": TUNE_RUNS},
        "seed": SEED,
        "rule": (
            "one arm per candidate at a fixed budget; winner per site is the "
            "lowest 5-sample mean; compile or run failures are recorded in "
            "band and skip only that candidate; winners are pinned into the "
            "full-layer harness and requalified there at full protocol"),
        "scope": "per-site tile screening only; not a registered comparison",
    })

    winners = {}
    for site, (depth, width, fused) in SITES.items():
        kept, filtered = enumerate_site(depth, width)
        emit({"kind": "site_candidates", "site": site, "kept": kept,
              "filtered": filtered})
        if not kept:
            emit({"kind": "site_skip", "site": site,
                  "reason": "no feasible tile candidate"})
            continue

        x, weight = common.device_uniform_inputs(
            (TOKENS, depth, width), SEED)
        residual = common.device_uniform_inputs(
            (TOKENS, width, 512), SEED + 1)[0] if fused else None
        jax.block_until_ready((x, weight)
                              if residual is None else (x, weight, residual))

        native = time_function(
            lambda a, w: sp.native_matmul(a, w) if residual is None
            else sp.native_matmul(a, w) + residual, (x, weight))
        emit({"kind": "site_native", "site": site,
              "mean_ms": statistics.fmean(native), "samples_ms": native,
              "note": "context only; not part of the tile decision"})

        best = None
        for entry in kept:
            bm, bn, bk = entry["tile"]

            def strassen_fn(a, w, bm=bm, bn=bn, bk=bk):
                return sp.strassen_matmul(
                    a, w, bm=bm, bn=bn, bk=bk, interleave_products=True,
                    residual=residual,
                    epilogue="residual_add" if residual is not None else None,
                    vmem_limit_bytes=STRASSEN_LIMIT)

            try:
                samples = time_function(strassen_fn, (x, weight))
            except Exception as error:
                emit({"kind": "site_failure", "site": site,
                      "tile": [bm, bn, bk], "error": str(error)[:400]})
                continue
            mean = statistics.fmean(samples)
            emit({"kind": "site_tune", "site": site, "tile": [bm, bn, bk],
                  "mean_ms": mean, "samples_ms": samples,
                  "vmem_estimate": entry["vmem_estimate"]})
            if best is None or mean < best[0]:
                best = (mean, [bm, bn, bk])

        if best is not None:
            winners[site] = {"tile": best[1], "mean_ms": best[0],
                             "native_mean_ms": statistics.fmean(native),
                             "speedup_vs_native": statistics.fmean(native) / best[0]}
            emit({"kind": "site_winner", "site": site, **winners[site]})
        del x, weight, residual

    emit({"kind": "chosen", "model": layer.MODEL_NAME, "winners": winners,
          "site_tiles_env": ";".join(
              f"{s}:{','.join(str(v) for v in w['tile'])}"
              for s, w in winners.items())})
    emit({"kind": "verdict", "passes": bool(winners),
          "scope": "per-site tile screening; winners feed the full-layer "
                   "harness"})


if __name__ == "__main__":
    main()
