"""Streamed full-checkpoint quality gate for the outlier-panel hybrid.

Identical protocol to benchmark_checkpoint_model.py — matched-native and
candidate hidden states traverse all 32 real layers side by side, then the
real final norm and LM head — but the candidate applies, per layer:

- the intermediate-dimension outlier permutation (gate/up columns and down
  rows reordered by that layer's calibrated activation magnitudes), and
- the exact eight-product kernel path on the first K panel of the down GEMM
  (``classical_panels=(0,)``), which requires the 96 MiB scoped-VMEM ceiling
  installed by ``run_model_hybrid_96m.py``.

Calibration mode (default) computes each layer's permutation from the
native path's activations on this run's text and saves all 32 permutations.
Holdout mode (``HYBRID_IPERM_FILE`` set, plus holdout texts patched in by
``benchmark_checkpoint_model_hybrid_holdout.py``) loads them frozen.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_model as model_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_model_hybrid.jsonl")
IPERM_SAVE = Path("/content/results/hybrid_iperms.npy")
BUDGET_SAVE = Path("/content/results/hybrid_budgets.npy")
STATS_SAVE = Path("/content/results/hybrid_hidden_stats.npy")
IPERM_FILE = os.environ.get("HYBRID_IPERM_FILE", "")
BUDGET_FILE = os.environ.get("HYBRID_BUDGET_FILE", "")
# Optional global hidden-dimension permutation (free whole-model reordering).
# When set, the hybrid stream runs in the permuted hidden basis and the up
# GEMM computes its first K panel (the concentrated hidden outliers) exactly.
HPERM_FILE = os.environ.get("HYBRID_HPERM_FILE", "")
UP_PANELS_WITH_HPERM = (0,)
PANEL_VMEM_LIMIT_BYTES = 90 * 1024 * 1024
NUM_LAYERS = model_bench.NUM_LAYERS
# Predeclared adaptive budget rule: the exact panels must cover every
# intermediate channel whose calibrated magnitude exceeds OUTLIER_TAU times
# the layer median, subject to MIN..MAX panels. Declared before running.
OUTLIER_TAU = 16.0
MIN_PANELS = 1
MAX_PANELS = 4
PANEL = 512


def budget_from_magnitudes(sorted_mags):
    median = float(np.median(sorted_mags))
    above = int(np.sum(sorted_mags > OUTLIER_TAU * max(median, 1e-30)))
    panels = int(np.ceil(above / PANEL)) if above else 0
    return max(MIN_PANELS, min(MAX_PANELS, panels))


def emit(record):
    import json

    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("MODEL_HYBRID_JSON " + line, flush=True)


model_bench.emit = emit
layer_bench.emit = emit


def gemm(lhs, rhs, *, strassen=False, panels=()):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    if not strassen:
        output = sp.native_matmul(lhs, rhs)
    else:
        shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
        output = sp.strassen_matmul(
            lhs,
            rhs,
            interleave_products=shape in sp.TUNED_INTERLEAVED_SHAPES,
            classical_panels=panels,
            vmem_limit_bytes=(
                PANEL_VMEM_LIMIT_BYTES if panels else sp.TUNED_VMEM_LIMIT_BYTES
            ),
        )
    return jax.lax.optimization_barrier(output)


def make_layer(*, strassen, down_panels=(), up_panels=()):
    def layer(x, params):
        residual, normalized = attention_prefix(x, params)
        gate_up = gemm(
            normalized, params.gate_up, strassen=strassen, panels=up_panels
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(
            activated, params.mlp_down, strassen=strassen, panels=down_panels
        )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def attention_prefix(x, params):
    normalized = layer_bench.rms_norm(x, params.attention_norm)
    query = gemm(normalized, params.query)
    key = gemm(normalized, params.key)
    value = gemm(normalized, params.value)
    query = query.reshape(
        layer_bench.BATCH, layer_bench.SEQUENCE,
        layer_bench.HEADS, layer_bench.HEAD_DIM,
    )
    key = key.reshape(
        layer_bench.BATCH, layer_bench.SEQUENCE,
        layer_bench.KV_HEADS, layer_bench.HEAD_DIM,
    )
    value = value.reshape(
        layer_bench.BATCH, layer_bench.SEQUENCE,
        layer_bench.KV_HEADS, layer_bench.HEAD_DIM,
    )
    query = layer_bench.apply_rope(query, params.rope_cos, params.rope_sin)
    key = layer_bench.apply_rope(key, params.rope_cos, params.rope_sin)
    attended = jax.nn.dot_product_attention(
        query, key, value, is_causal=True, implementation="xla"
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


@jax.jit
def native_activated(hidden, params):
    """Per-channel calibration stats from the native path: intermediate
    activation magnitudes and the up-GEMM input (normalized) magnitudes."""
    _, normalized = attention_prefix(hidden, params)
    projected = sp.native_matmul(normalized, params.gate_up)
    gate, up = jnp.split(projected, 2, axis=-1)
    activated = (
        jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    )
    inter_mags = jnp.max(jnp.abs(activated), axis=0)
    hidden_mags = jnp.max(jnp.abs(normalized.astype(jnp.float32)), axis=0)
    return inter_mags, hidden_mags


def permute_intermediate(params, iperm):
    gate, up = jnp.split(params.gate_up, 2, axis=-1)
    return params._replace(
        gate_up=jnp.concatenate((gate[:, iperm], up[:, iperm]), axis=1),
        mlp_down=params.mlp_down[iperm, :],
    )


def permute_hidden_params(params, hperm):
    """Reorder every hidden-dimension axis of one layer's weights."""
    return params._replace(
        attention_norm=params.attention_norm[hperm],
        query=params.query[hperm, :],
        key=params.key[hperm, :],
        value=params.value[hperm, :],
        attention_output=params.attention_output[:, hperm],
        mlp_norm=params.mlp_norm[hperm],
        gate_up=params.gate_up[hperm, :],
        mlp_down=params.mlp_down[:, hperm],
    )


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    scoped = os.environ.get("LIBTPU_INIT_ARGS", "")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    frozen = bool(IPERM_FILE)
    # Load the optional global hidden permutation BEFORE any use: the initial
    # hybrid hidden state and every layer's weights depend on it.
    global_hperm = np.load(HPERM_FILE) if HPERM_FILE else None
    up_panels = UP_PANELS_WITH_HPERM if global_hperm is not None else ()
    hperm_dev = inv_hperm_dev = None
    if global_hperm is not None:
        hperm_dev = jnp.asarray(global_hperm.astype(np.int32))
        inv_hperm_dev = jnp.asarray(
            np.argsort(global_hperm).astype(np.int32)
        )
    emit({
        "kind": "metadata",
        "repository": layer_bench.REPOSITORY,
        "revision": layer_bench.REVISION,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "libtpu_init_args": scoped,
        "layers": NUM_LAYERS,
        "quality_only": True,
        "candidate": (
            "per-layer intermediate outlier permutation + depth-graded "
            "exact down panels"
        ),
        "budget_rule": {
            "outlier_tau": OUTLIER_TAU,
            "min_panels": MIN_PANELS,
            "max_panels": MAX_PANELS,
            "panel": PANEL,
        },
        "calibration": "frozen from file" if frozen else "self, this text",
        "iperm_file": IPERM_FILE,
        "budget_file": BUDGET_FILE,
    })

    token_ids = model_bench.make_tokens()
    paths = model_bench.download_checkpoint()
    checkpoint = model_bench.Checkpoint(paths)
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    )
    native_hidden = jnp.asarray(initial)
    hybrid_hidden = (
        jnp.asarray(initial[:, global_hperm])
        if global_hperm is not None
        else jnp.asarray(initial)
    )
    rope_cos, rope_sin = layer_bench.rope_values()

    frozen_iperms = np.load(IPERM_FILE) if frozen else None
    frozen_budgets = np.load(BUDGET_FILE) if BUDGET_FILE else None
    saved_iperms = np.zeros(
        (NUM_LAYERS, layer_bench.INTERMEDIATE_DIM), dtype=np.int32
    )
    saved_budgets = np.zeros(NUM_LAYERS, dtype=np.int32)
    saved_hidden_stats = np.zeros(
        (NUM_LAYERS, layer_bench.MODEL_DIM), dtype=np.float32
    )

    native_layer_fn = make_layer(strassen=False)
    native_executable = None
    hybrid_executables = {}

    def hybrid_executable_for(budget, hidden, params_hybrid):
        if budget not in hybrid_executables:
            started = time.perf_counter()
            fn = make_layer(
                strassen=True,
                down_panels=tuple(range(budget)),
                up_panels=up_panels,
            )
            hybrid_executables[budget] = jax.jit(fn).lower(
                hidden, params_hybrid
            ).compile()
            emit({
                "kind": "compile",
                "budget": budget,
                "compile_s": time.perf_counter() - started,
            })
        return hybrid_executables[budget]

    all_finite = True
    final_hidden_error = None
    for index in range(NUM_LAYERS):
        params = model_bench.load_layer(checkpoint, index, rope_cos, rope_sin)
        if frozen:
            iperm = np.asarray(frozen_iperms[index], dtype=np.int64)
            budget = (
                int(frozen_budgets[index])
                if frozen_budgets is not None
                else MIN_PANELS
            )
        else:
            inter_mags, hidden_mags = native_activated(native_hidden, params)
            mags = np.asarray(inter_mags, dtype=np.float32)
            saved_hidden_stats[index] = np.asarray(
                hidden_mags, dtype=np.float32
            )
            iperm = np.argsort(-mags)
            sorted_mags = mags[iperm]
            budget = budget_from_magnitudes(sorted_mags)
            uncovered = float(sorted_mags[budget * PANEL])
            emit({
                "kind": "calibration_layer",
                "layer": index,
                "budget": budget,
                "top_channel": int(iperm[0]),
                "top_magnitudes": sorted_mags[:4].tolist(),
                "median_magnitude": float(np.median(mags)),
                "max_uncovered_magnitude": uncovered,
                "hidden_input_max": float(np.max(saved_hidden_stats[index])),
                "hidden_input_median": float(
                    np.median(saved_hidden_stats[index])
                ),
            })
        saved_iperms[index] = iperm
        saved_budgets[index] = budget
        params_hybrid = permute_intermediate(params, jnp.asarray(iperm))
        if global_hperm is not None:
            params_hybrid = permute_hidden_params(params_hybrid, hperm_dev)
        jax.block_until_ready(params_hybrid.gate_up)

        if native_executable is None:
            started = time.perf_counter()
            native_executable = jax.jit(native_layer_fn).lower(
                native_hidden, params
            ).compile()
            emit({"kind": "compile", "compile_s": time.perf_counter() - started})
        hybrid_executable = hybrid_executable_for(
            budget, hybrid_hidden, params_hybrid
        )

        native_hidden = native_executable(native_hidden, params)
        hybrid_hidden = hybrid_executable(hybrid_hidden, params_hybrid)
        jax.block_until_ready((native_hidden, hybrid_hidden))
        hybrid_canonical = (
            hybrid_hidden[:, inv_hperm_dev]
            if global_hperm is not None
            else hybrid_hidden
        )
        error = bench.device_error(hybrid_canonical, native_hidden)
        finite = bool(
            jax.device_get(
                jnp.all(jnp.isfinite(native_hidden))
                & jnp.all(jnp.isfinite(hybrid_hidden))
            )
        )
        all_finite = all_finite and finite
        final_hidden_error = error
        emit({
            "kind": "layer_accuracy",
            "layer": index,
            "finite": finite,
            "hybrid_vs_native": error,
        })
        del params, params_hybrid
        gc.collect()

    if not frozen:
        np.save(IPERM_SAVE, saved_iperms)
        np.save(BUDGET_SAVE, saved_budgets)
        np.save(STATS_SAVE, saved_hidden_stats)
        emit({
            "kind": "calibration_saved",
            "iperms": str(IPERM_SAVE),
            "budgets": saved_budgets.tolist(),
            "extra_panel_layers": int(np.sum(saved_budgets > 1)),
        })

    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    final_norm_hybrid = (
        final_norm[hperm_dev] if global_hperm is not None else final_norm
    )
    native_hidden = layer_bench.rms_norm(native_hidden, final_norm)
    hybrid_hidden = layer_bench.rms_norm(hybrid_hidden, final_norm_hybrid)
    jax.block_until_ready((native_hidden, hybrid_hidden))
    hybrid_post_norm_canonical = (
        hybrid_hidden[:, inv_hperm_dev]
        if global_hperm is not None
        else hybrid_hidden
    )
    normalized_error = bench.device_error(
        hybrid_post_norm_canonical, native_hidden
    )

    lm_head = model_bench.device_matrix(checkpoint, "lm_head.weight")
    lm_head_hybrid = (
        lm_head[hperm_dev, :] if global_hperm is not None else lm_head
    )
    targets = jnp.asarray(token_ids[:, 1:].reshape(-1))
    logits_executable = jax.jit(model_bench.logits_and_loss).lower(
        native_hidden, lm_head, targets
    ).compile()
    native_logits, native_loss = logits_executable(
        native_hidden, lm_head, targets
    )
    hybrid_logits, hybrid_loss = logits_executable(
        hybrid_hidden, lm_head_hybrid, targets
    )
    jax.block_until_ready(
        (native_logits, native_loss, hybrid_logits, hybrid_loss)
    )
    logits_error = bench.device_error(hybrid_logits, native_logits)
    top1_agreement = float(
        jax.device_get(
            jnp.mean(
                jnp.argmax(native_logits, axis=-1)
                == jnp.argmax(hybrid_logits, axis=-1)
            )
        )
    )
    native_loss, hybrid_loss = map(
        float, jax.device_get((native_loss, hybrid_loss))
    )
    absolute_loss_delta = abs(hybrid_loss - native_loss)
    passes = (
        all_finite
        and normalized_error["l2_relative"] <= 0.02
        and normalized_error["maxnorm_relative"] <= 0.02
        and logits_error["l2_relative"] <= 0.02
        and logits_error["maxnorm_relative"] <= 0.02
        and top1_agreement >= 0.95
        and absolute_loss_delta <= 0.01
    )
    emit({
        "kind": "model_accuracy",
        "final_hidden_pre_norm": final_hidden_error,
        "final_hidden_post_norm": normalized_error,
        "logits": logits_error,
        "top1_agreement": top1_agreement,
        "native_next_token_loss": native_loss,
        "hybrid_next_token_loss": hybrid_loss,
        "absolute_loss_delta": absolute_loss_delta,
        "passes": passes,
    })
    emit({"kind": "final", "success": passes})
    if not passes:
        raise RuntimeError("hybrid full-checkpoint quality gate failed")


if __name__ == "__main__":
    main()
