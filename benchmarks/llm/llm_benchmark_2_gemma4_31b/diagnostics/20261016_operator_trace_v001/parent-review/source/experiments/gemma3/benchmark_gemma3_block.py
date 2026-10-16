"""Real-weight Gemma 3 27B transformer block, layer 0.

Gemma differs from Qwen3 in four ways that matter, all verified against the
published checkpoint rather than assumed:

1. Four norms per block, not two.  `post_attention_layernorm` is a genuine
   post-norm here, where the same HF name denotes Qwen's *pre*-MLP norm, so
   reusing the Qwen loader's mapping would be silently wrong.  Because a norm
   sits between each projection and its residual add, the `residual_add`
   epilogue that wins at o and down on Qwen cannot fuse on Gemma.
2. RMSNorm scales by `(1 + weight)`, not `weight`.
3. GeGLU (`gelu_pytorch_tanh`) rather than SwiGLU.
4. Model dimension 5376 = 256 x 21, so q/k/v/gate_up admit only the
   contraction depths 256, 768, 1792 and 5376; the tuned Qwen depth of 512 is
   illegal.

QK-norm is present, so `qk_norm_rope` ports unchanged -- Gemma's `(1 + w)`
convention is absorbed for free by handing the epilogue `1 + q_norm` as its
scale, which is an offline transformation of a vector.

Layer 0 is a `sliding_attention` layer with `rope_theta=10000` and a
1024-token window.  At the pinned sequence length of 1024 that window spans
the whole sequence, so sliding attention and causal attention coincide; the
harness asserts this rather than relying on it silently.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import NamedTuple

os.environ.setdefault("QWEN3_MODEL", "gemma3-27b")
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

RMS_EPS = 1e-6
ROPE_LOCAL_BASE = 10000.0
QUERY_PRE_ATTN_SCALAR = 168
SLIDING_WINDOW = 1024
HEAD_DIM = layer.HEAD_DIM
HEADS, KV_HEADS = layer.HEADS, layer.KV_HEADS

BM, BN, BK = (int(v) for v in os.environ.get(
    "GEMMA_PRODUCT_TILE", "1024,768,5376").split(","))
CUBIC_BM, CUBIC_BN, CUBIC_BK = (int(v) for v in os.environ.get(
    "GEMMA_CUBIC_TILE", f"{BM},{BN},{BK}").split(","))
QK_TILE = tuple(int(v) for v in os.environ.get(
    "GEMMA_QK_TILE", "1024,1024,5376").split(","))
FUSED_QK = os.environ.get("GEMMA_FUSED_QK", "") not in ("", "0")
# Gemma puts a norm between each projection and its residual add, which is
# why plain residual_add cannot fuse at o and down.  norm_residual_add folds
# both in, at the cost of requiring bn == n so the row's sum of squares is
# available in one tile.
NORM_RESIDUAL = os.environ.get("GEMMA_NORM_RESIDUAL", "") not in ("", "0")
O_TILE = tuple(int(v) for v in os.environ.get(
    "GEMMA_O_TILE", "1024,5376,512").split(","))
DOWN_TILE = tuple(int(v) for v in os.environ.get(
    "GEMMA_DOWN_TILE", "1024,5376,512").split(","))
STRASSEN_LIMIT = int(
    os.environ.get("QWEN3_STRASSEN_LIMIT_MIB", "104")) * 1024 * 1024
CUBIC_LIMIT = int(os.environ.get("QWEN3_CUBIC_LIMIT_MIB", "104")) * 1024 * 1024
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
WARMUPS, RUNS = 3, 20
ARMS = ("regular_xla", "cubic_pallas", "strassen_fused",
        "product_strassen_fused")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(f"{RESULTS_DIR}/strassen_gemma3_27b_block{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("GEMMA3_BLOCK_JSON " + line, flush=True)


class GemmaParameters(NamedTuple):
    input_norm: jax.Array
    query: jax.Array
    query_norm: jax.Array
    key: jax.Array
    key_norm: jax.Array
    value: jax.Array
    attention_output: jax.Array
    post_attention_norm: jax.Array
    post_attention_norm_scale: jax.Array
    pre_feedforward_norm: jax.Array
    post_feedforward_norm: jax.Array
    post_feedforward_norm_scale: jax.Array
    gate_up: jax.Array
    gate_up_laid: jax.Array
    mlp_down: jax.Array
    rope_cos: jax.Array
    rope_sin: jax.Array
    query_laid: jax.Array
    key_laid: jax.Array
    query_norm_laid: jax.Array
    key_norm_laid: jax.Array
    rope_table_cos: jax.Array
    rope_table_sin: jax.Array


def gemma_rms_norm(x, weight):
    """Gemma scales by (1 + weight); Qwen scales by weight."""
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(
        jnp.mean(x32 * x32, axis=-1, keepdims=True) + RMS_EPS)
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(jnp.bfloat16)


def rope_values():
    positions = jnp.arange(layer.SEQUENCE, dtype=jnp.float32)[:, None]
    inverse = 1.0 / (
        ROPE_LOCAL_BASE
        ** (jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM))
    angles = jnp.concatenate((positions * inverse[None, :],) * 2, axis=-1)
    return (jnp.cos(angles).astype(jnp.bfloat16),
            jnp.sin(angles).astype(jnp.bfloat16))


def apply_rope(x, cos, sin):
    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    cos = cos[None, :, None, :].astype(jnp.float32)
    sin = sin[None, :, None, :].astype(jnp.float32)
    return (x.astype(jnp.float32) * cos + rotated * sin).astype(jnp.bfloat16)


def activate(full):
    gate, up = jnp.split(full, 2, axis=-1)
    return (
        jax.nn.gelu(gate.astype(jnp.float32), approximate=True)
        * up.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def make_block(arm):
    def qk_fused(site, lhs, params):
        qk_bm, qk_bn, qk_bk = QK_TILE
        if site == "q":
            laid, scale, heads = params.query_laid, params.query_norm_laid, HEADS
        else:
            laid, scale, heads = params.key_laid, params.key_norm_laid, KV_HEADS
        flat = sp.strassen_matmul(
            lhs, laid, bm=qk_bm, bn=qk_bn, bk=qk_bk, interleave_products=True,
            epilogue="qk_norm_rope", rope_cos=params.rope_table_cos,
            rope_sin=params.rope_table_sin, rope_scale=scale,
            rope_head_dim=HEAD_DIM, rope_eps=RMS_EPS,
            vmem_limit_bytes=STRASSEN_LIMIT)
        return layer.rope_layout_to_heads(flat, heads, qk_bn)

    def run(x, params):
        residual = x
        normalized = gemma_rms_norm(x, params.input_norm)
        fuse = FUSED_QK and arm in ("strassen_fused", "product_strassen_fused")
        if fuse:
            query = qk_fused("q", normalized, params)
            key = qk_fused("k", normalized, params)
        else:
            query = sp.native_matmul(normalized, params.query).reshape(
                layer.BATCH, layer.SEQUENCE, HEADS, HEAD_DIM)
            key = sp.native_matmul(normalized, params.key).reshape(
                layer.BATCH, layer.SEQUENCE, KV_HEADS, HEAD_DIM)
            query = gemma_rms_norm(query, params.query_norm)
            key = gemma_rms_norm(key, params.key_norm)
            query = apply_rope(query, params.rope_cos, params.rope_sin)
            key = apply_rope(key, params.rope_cos, params.rope_sin)
        value = sp.native_matmul(normalized, params.value).reshape(
            layer.BATCH, layer.SEQUENCE, KV_HEADS, HEAD_DIM)
        attended = jax.nn.dot_product_attention(
            query, key, value, is_causal=True,
            scale=QUERY_PRE_ATTN_SCALAR ** -0.5, implementation="xla",
        ).reshape(layer.TOKENS, HEADS * HEAD_DIM).astype(jnp.bfloat16)
        fuse_norm = NORM_RESIDUAL and arm in (
            "strassen_fused", "product_strassen_fused")
        if fuse_norm:
            obm, obn, obk = O_TILE
            residual = sp.strassen_matmul(
                attended, params.attention_output, bm=obm, bn=obn, bk=obk,
                interleave_products=True, epilogue="norm_residual_add",
                residual=residual,
                norm_scale=params.post_attention_norm_scale,
                norm_eps=RMS_EPS, vmem_limit_bytes=STRASSEN_LIMIT)
        else:
            attention_output = sp.native_matmul(
                attended, params.attention_output)
            # The norm between the projection and the residual add is why the
            # plain residual_add epilogue cannot fuse at o on Gemma.
            attention_output = gemma_rms_norm(
                attention_output, params.post_attention_norm)
            residual = (
                residual.astype(jnp.float32)
                + attention_output.astype(jnp.float32)
            ).astype(jnp.bfloat16)

        hidden = gemma_rms_norm(residual, params.pre_feedforward_norm)
        if arm == "regular_xla":
            activated = activate(sp.native_matmul(hidden, params.gate_up))
        elif arm == "cubic_pallas":
            activated = activate(cubic.cubic_matmul(
                hidden, params.gate_up, variant="blocked",
                bm=CUBIC_BM, bn=CUBIC_BN, bk=CUBIC_BK,
                vmem_limit_bytes=CUBIC_LIMIT))
        else:
            activated = sp.strassen_matmul(
                hidden, params.gate_up_laid, bm=BM, bn=BN, bk=BK,
                epilogue="geglu", interleave_products=True,
                product_aware_swiglu=(arm == "product_strassen_fused"),
                vmem_limit_bytes=STRASSEN_LIMIT)
        if fuse_norm:
            dbm, dbn, dbk = DOWN_TILE
            return sp.strassen_matmul(
                activated, params.mlp_down, bm=dbm, bn=dbn, bk=dbk,
                interleave_products=True, epilogue="norm_residual_add",
                residual=residual,
                norm_scale=params.post_feedforward_norm_scale,
                norm_eps=RMS_EPS, vmem_limit_bytes=STRASSEN_LIMIT)
        projected = sp.native_matmul(activated, params.mlp_down)
        projected = gemma_rms_norm(projected, params.post_feedforward_norm)
        return (
            residual.astype(jnp.float32) + projected.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return run


def load_params():
    checkpoint = layer.ShardedSafetensors()
    prefix = layer.TENSOR_PREFIX

    def weight(name):
        return jnp.asarray(checkpoint.tensor(f"{prefix}.layers.0.{name}").T)

    def vector(name):
        return jnp.asarray(checkpoint.tensor(f"{prefix}.layers.0.{name}"))

    gate = weight("mlp.gate_proj.weight")
    up = weight("mlp.up_proj.weight")
    gate_up = jnp.concatenate((gate, up), axis=1)
    rope_cos, rope_sin = rope_values()
    query = weight("self_attn.q_proj.weight")
    key = weight("self_attn.k_proj.weight")
    query_norm = vector("self_attn.q_norm.weight")
    key_norm = vector("self_attn.k_norm.weight")

    qk_bn = QK_TILE[1]
    cos = jnp.tile(rope_cos, (layer.BATCH, 1))
    sin = jnp.tile(rope_sin, (layer.BATCH, 1))
    table_cos, table_sin = sp.rope_layout_tables(cos, sin, qk_bn // 2, HEAD_DIM)
    # Gemma's (1 + w) convention folds into the epilogue's scale for free.
    params = GemmaParameters(
        input_norm=vector("input_layernorm.weight"),
        query=query, query_norm=query_norm,
        key=key, key_norm=key_norm,
        value=weight("self_attn.v_proj.weight"),
        attention_output=weight("self_attn.o_proj.weight"),
        post_attention_norm=vector("post_attention_layernorm.weight"),
        post_attention_norm_scale=(
            1.0 + vector("post_attention_layernorm.weight").astype(
                jnp.float32)).astype(jnp.bfloat16),
        pre_feedforward_norm=vector("pre_feedforward_layernorm.weight"),
        post_feedforward_norm=vector("post_feedforward_layernorm.weight"),
        post_feedforward_norm_scale=(
            1.0 + vector("post_feedforward_layernorm.weight").astype(
                jnp.float32)).astype(jnp.bfloat16),
        gate_up=gate_up,
        gate_up_laid=sp.swiglu_weight_layout(gate_up, BN),
        mlp_down=weight("mlp.down_proj.weight"),
        rope_cos=rope_cos, rope_sin=rope_sin,
        query_laid=sp.rope_weight_layout(query, qk_bn, HEAD_DIM),
        key_laid=sp.rope_weight_layout(key, qk_bn, HEAD_DIM),
        query_norm_laid=sp.rope_scale_layout(
            1.0 + query_norm.astype(jnp.float32), qk_bn, HEAD_DIM
        ).astype(jnp.bfloat16),
        key_norm_laid=sp.rope_scale_layout(
            1.0 + key_norm.astype(jnp.float32), qk_bn, HEAD_DIM
        ).astype(jnp.bfloat16),
        rope_table_cos=table_cos, rope_table_sin=table_sin,
    )
    jax.block_until_ready(params)
    return params


def main():
    if layer.SEQUENCE > SLIDING_WINDOW:
        raise ValueError(
            f"sequence {layer.SEQUENCE} exceeds the {SLIDING_WINDOW}-token "
            "sliding window, so layer 0's local attention is no longer "
            "equivalent to causal attention and needs an explicit mask")
    layer.emit = emit
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "tile": [BM, BN, BK], "cubic_tile": [CUBIC_BM, CUBIC_BN, CUBIC_BK],
        "qk_tile": list(QK_TILE), "fused_qk": FUSED_QK,
        "norm_residual": NORM_RESIDUAL,
        "o_tile": list(O_TILE), "down_tile": list(DOWN_TILE),
        "activation": "geglu", "arms": list(ARMS),
        "shape": {"tokens": layer.TOKENS, "model": layer.MODEL_DIM,
                  "intermediate": layer.INTERMEDIATE_DIM,
                  "heads": HEADS, "kv_heads": KV_HEADS},
        "policy": ("gate/up + GeGLU"
                   + ("; fused q/k RMSNorm+RoPE" if FUSED_QK else "")
                   + ("; o and down fused with norm_residual_add"
                      if NORM_RESIDUAL else
                      "; o and down left to XLA because a norm separates "
                      "each projection from its residual add")),
        "scope": "complete real-weight Gemma 3 27B layer-0 block",
    })
    x = layer.deterministic_hidden()
    params = load_params()
    args = (x, params)

    executables, outputs = {}, {}
    for arm in ARMS:
        started = time.perf_counter()
        executables[arm] = jax.jit(make_block(arm)).lower(*args).compile()
        emit({"kind": "compile", "arm": arm,
              "seconds": time.perf_counter() - started})
        outputs[arm] = jax.block_until_ready(executables[arm](*args))

    reference = jnp.asarray(outputs["regular_xla"], jnp.float32)
    emit({"kind": "accuracy", "errors_vs_xla": {
        arm: common.device_error(outputs[arm], reference)
        for arm in ARMS if arm != "regular_xla"}})

    # benchmark_common.interleaved_timings is shaped for two-operand GEMMs
    # and returns summaries; the block needs raw samples, so it rotates the
    # arm order here the same way, alternating direction on odd runs.
    names = tuple(executables)
    for warmup in range(WARMUPS):
        offset = warmup % len(names)
        for arm in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[arm](*args))
    timings = {arm: [] for arm in names}
    for run in range(RUNS):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
        if run % 2:
            order = tuple(reversed(order))
        for arm in order:
            started_ns = time.perf_counter_ns()
            jax.block_until_ready(executables[arm](*args))
            timings[arm].append((time.perf_counter_ns() - started_ns) / 1e6)
    means = {arm: statistics.fmean(v) for arm, v in timings.items()}

    def independent(reference, candidate):
        n = len(reference)
        mean = statistics.fmean(candidate) - statistics.fmean(reference)
        spread = (statistics.variance(candidate) / n
                  + statistics.variance(reference) / n) ** 0.5
        return {"mean_ms": mean,
                "ci95_ms": [mean - 2.093 * spread, mean + 2.093 * spread]}
    best = min((v, a) for a, v in means.items() if "strassen" in a)
    emit({
        "kind": "performance", "mean_ms": means, "samples_ms": timings,
        "best_strassen_arm": best[1],
        "speedup_vs_xla": means["regular_xla"] / best[0],
        "speedup_vs_cubic": means["cubic_pallas"] / best[0],
        "product_minus_standard_strassen": independent(
            timings["strassen_fused"], timings["product_strassen_fused"]),
        "product_minus_xla": independent(
            timings["regular_xla"], timings["product_strassen_fused"]),
    })
    emit({"kind": "verdict", "passes": best[0] < means["regular_xla"],
          "scope": "Gemma 3 27B layer-0 block"})


if __name__ == "__main__":
    main()
