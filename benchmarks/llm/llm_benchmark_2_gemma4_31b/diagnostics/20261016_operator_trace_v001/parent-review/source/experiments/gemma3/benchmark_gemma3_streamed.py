"""Streamed all-layer Gemma 3 27B inference with a task gate.

Streams the 62 layers one at a time, timing each under three arms and
carrying the hidden state forward, then scores the final logits against the
native path.  The structure follows the Qwen streamed harness; what differs
is everything Gemma does differently, each verified against the published
checkpoint:

* Four norms per block, RMSNorm scaling by ``(1 + weight)``, GeGLU.
* Attention type alternates five sliding layers to one full layer.  The two
  kinds use different RoPE bases -- 10000 local, 1000000 global with the
  config's linear factor of 8 -- so the tables are built per layer type
  rather than once.  At the pinned 1024-token sequence the 1024-token
  sliding window spans the whole sequence, so both kinds are causal here;
  the harness refuses longer sequences rather than quietly dropping a mask.
* The embedding is scaled by sqrt(hidden_size) on input, and there is no
  ``lm_head``: Gemma ties the head to ``embed_tokens``.
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
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0"

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as common
import benchmark_cubic_control as cubic
import benchmark_gemma3_block as block
import benchmark_qwen3_32b_layer as layer
import mosaic_compat
import strassen_pallas as sp

HEAD_DIM, HEADS, KV_HEADS = block.HEAD_DIM, block.HEADS, block.KV_HEADS
RMS_EPS = block.RMS_EPS
NUM_LAYERS = int(os.environ.get(
    "GEMMA_LAYERS", layer._MODEL["num_layers"]))
SLIDING_PERIOD = 6          # five sliding layers, then one full
ROPE_LOCAL_BASE, ROPE_GLOBAL_BASE = 10000.0, 1000000.0
ROPE_GLOBAL_FACTOR = 8.0
LOGIT_CHUNK = 256
CORPUS = os.environ.get("GEMMA_CORPUS", "repeated")
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WARMUPS, RUNS = 2, 5
ARMS = ("regular_xla", "gated_cubic", "gated_strassen")
# The block reaches 1.1577x on v5e and 1.1299x on v6e only with q and k
# carrying their per-head RMSNorm and RoPE in kernel; without it both sit
# near 1.078x.  The streamed and downstream harnesses must be able to run
# the same policy, or they certify something the headline does not claim.
FUSED_QK = os.environ.get("GEMMA_FUSED_QK", "") not in ("", "0")
QK_TILE = tuple(int(v) for v in os.environ.get(
    "GEMMA_QK_TILE", "1024,1024,5376").split(","))
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(f"{RESULTS_DIR}/strassen_gemma3_27b_streamed{SUFFIX}.jsonl")

TEXTS = (
    "The capital of France is Paris, a city known for its museums.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A prime number has exactly two distinct positive divisors.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Shakespeare wrote both tragedies and comedies for the stage.",
    "Water freezes at zero degrees Celsius under standard pressure.",
    "The mitochondrion is often called the powerhouse of the cell.",
    "Gravity causes objects with mass to attract one another.",
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("GEMMA3_STREAM_JSON " + line, flush=True)


def is_full_attention(index):
    """Layer types run five sliding then one full, so index 5 is the first."""
    return index % SLIDING_PERIOD == SLIDING_PERIOD - 1


def rope_tables(full_attention):
    base = ROPE_GLOBAL_BASE if full_attention else ROPE_LOCAL_BASE
    positions = jnp.arange(layer.SEQUENCE, dtype=jnp.float32)[:, None]
    if full_attention:
        # The config's rope_scaling is linear with factor 8, applied to the
        # global layers only.
        positions = positions / ROPE_GLOBAL_FACTOR
    inverse = 1.0 / (
        base ** (jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM))
    angles = jnp.concatenate((positions * inverse[None, :],) * 2, axis=-1)
    return (jnp.cos(angles).astype(jnp.bfloat16),
            jnp.sin(angles).astype(jnp.bfloat16))


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        layer.REPOSITORY, revision=layer.REVISION, use_fast=True,
        token=layer._hub_token())


def make_tokens():
    """Either the eight repeated sentences or contiguous WikiText-2.

    The repeated corpus tokenises to about 74 unique ids, which is the same
    narrow input whose apparent drift turned out to be an artifact on Qwen.
    Natural text is the gate that actually binds, so it is selectable and
    the choice is recorded with the token hash.
    """
    import hashlib

    tokenizer = _tokenizer()
    if CORPUS == "wikitext2":
        from datasets import load_dataset

        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
            revision=WIKITEXT_REVISION)
        text = "\n".join(row["text"] for row in dataset)
        flat = tokenizer(text, add_special_tokens=False,
                         return_tensors="np")["input_ids"][0]
        needed = layer.BATCH * layer.SEQUENCE
        if flat.size < needed:
            raise ValueError(f"corpus has {flat.size} tokens, need {needed}")
        ids = np.asarray(
            flat[:needed], dtype=np.int32).reshape(
                layer.BATCH, layer.SEQUENCE)
        label = "wikitext-2-raw-v1/test first contiguous tokens"
    else:
        encoded = tokenizer(
            [(text + "\n") * 100 for text in TEXTS],
            add_special_tokens=True, max_length=layer.SEQUENCE,
            padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="np")
        ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        label = "eight repeated sentences"
    emit({"kind": "tokens", "corpus": label, "shape": list(ids.shape),
          "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
          "unique_tokens": int(np.unique(ids).size)})
    return ids


class LayerParameters(NamedTuple):
    input_norm: jax.Array
    query: jax.Array
    query_norm: jax.Array
    key: jax.Array
    key_norm: jax.Array
    value: jax.Array
    attention_output: jax.Array
    post_attention_norm: jax.Array
    pre_feedforward_norm: jax.Array
    post_feedforward_norm: jax.Array
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


def load_layer(checkpoint, index):
    prefix = f"{layer.TENSOR_PREFIX}.layers.{index}"

    def weight(name):
        return jnp.asarray(checkpoint.tensor(f"{prefix}.{name}").T)

    def vector(name):
        return jnp.asarray(checkpoint.tensor(f"{prefix}.{name}"))

    gate_up = jnp.concatenate(
        (weight("mlp.gate_proj.weight"), weight("mlp.up_proj.weight")), axis=1)
    cos, sin = rope_tables(is_full_attention(index))
    query = weight("self_attn.q_proj.weight")
    key = weight("self_attn.k_proj.weight")
    query_norm = vector("self_attn.q_norm.weight")
    key_norm = vector("self_attn.k_norm.weight")
    if FUSED_QK:
        qk_bn = QK_TILE[1]
        # Per-layer, because sliding and full layers use different RoPE bases.
        table_cos, table_sin = sp.rope_layout_tables(
            jnp.tile(cos, (layer.BATCH, 1)), jnp.tile(sin, (layer.BATCH, 1)),
            qk_bn // 2, HEAD_DIM)
        fused = dict(
            query_laid=sp.rope_weight_layout(query, qk_bn, HEAD_DIM),
            key_laid=sp.rope_weight_layout(key, qk_bn, HEAD_DIM),
            # Gemma's (1 + w) folds into the epilogue scale exactly.
            query_norm_laid=sp.rope_scale_layout(
                1.0 + query_norm.astype(jnp.float32), qk_bn, HEAD_DIM
            ).astype(jnp.bfloat16),
            key_norm_laid=sp.rope_scale_layout(
                1.0 + key_norm.astype(jnp.float32), qk_bn, HEAD_DIM
            ).astype(jnp.bfloat16),
            rope_table_cos=table_cos, rope_table_sin=table_sin)
    else:
        empty = jnp.zeros((1, 1), jnp.bfloat16)
        fused = dict(query_laid=empty, key_laid=empty,
                     query_norm_laid=empty, key_norm_laid=empty,
                     rope_table_cos=empty, rope_table_sin=empty)
    params = LayerParameters(
        input_norm=vector("input_layernorm.weight"),
        query=query, query_norm=query_norm,
        key=key, key_norm=key_norm,
        value=weight("self_attn.v_proj.weight"),
        attention_output=weight("self_attn.o_proj.weight"),
        post_attention_norm=vector("post_attention_layernorm.weight"),
        pre_feedforward_norm=vector("pre_feedforward_layernorm.weight"),
        post_feedforward_norm=vector("post_feedforward_layernorm.weight"),
        gate_up=gate_up,
        gate_up_laid=sp.swiglu_weight_layout(gate_up, block.BN),
        mlp_down=weight("mlp.down_proj.weight"),
        rope_cos=cos, rope_sin=sin, **fused)
    jax.block_until_ready(params)
    return params


def make_layer(arm):
    def run(x, params):
        residual = x
        normalized = block.gemma_rms_norm(x, params.input_norm)
        if FUSED_QK and arm == "gated_strassen":
            qk_bm, qk_bn, qk_bk = QK_TILE

            def fused_qk(laid, scale, heads):
                flat = sp.strassen_matmul(
                    normalized, laid, bm=qk_bm, bn=qk_bn, bk=qk_bk,
                    interleave_products=True, epilogue="qk_norm_rope",
                    rope_cos=params.rope_table_cos,
                    rope_sin=params.rope_table_sin, rope_scale=scale,
                    rope_head_dim=HEAD_DIM, rope_eps=RMS_EPS,
                    vmem_limit_bytes=block.STRASSEN_LIMIT)
                return layer.rope_layout_to_heads(flat, heads, qk_bn)

            query = fused_qk(params.query_laid, params.query_norm_laid, HEADS)
            key = fused_qk(params.key_laid, params.key_norm_laid, KV_HEADS)
        else:
            query = sp.native_matmul(normalized, params.query).reshape(
                layer.BATCH, layer.SEQUENCE, HEADS, HEAD_DIM)
            key = sp.native_matmul(normalized, params.key).reshape(
                layer.BATCH, layer.SEQUENCE, KV_HEADS, HEAD_DIM)
            query = block.gemma_rms_norm(query, params.query_norm)
            key = block.gemma_rms_norm(key, params.key_norm)
            query = block.apply_rope(query, params.rope_cos, params.rope_sin)
            key = block.apply_rope(key, params.rope_cos, params.rope_sin)
        value = sp.native_matmul(normalized, params.value).reshape(
            layer.BATCH, layer.SEQUENCE, KV_HEADS, HEAD_DIM)
        attended = jax.nn.dot_product_attention(
            query, key, value, is_causal=True,
            scale=block.QUERY_PRE_ATTN_SCALAR ** -0.5, implementation="xla",
        ).reshape(layer.TOKENS, HEADS * HEAD_DIM).astype(jnp.bfloat16)
        attention = block.gemma_rms_norm(
            sp.native_matmul(attended, params.attention_output),
            params.post_attention_norm)
        residual = (
            residual.astype(jnp.float32) + attention.astype(jnp.float32)
        ).astype(jnp.bfloat16)

        hidden = block.gemma_rms_norm(residual, params.pre_feedforward_norm)
        if arm == "regular_xla":
            activated = block.activate(
                sp.native_matmul(hidden, params.gate_up))
        elif arm == "gated_cubic":
            activated = block.activate(cubic.cubic_matmul(
                hidden, params.gate_up, variant="blocked",
                bm=block.CUBIC_BM, bn=block.CUBIC_BN, bk=block.CUBIC_BK,
                vmem_limit_bytes=block.CUBIC_LIMIT))
        else:
            activated = sp.strassen_matmul(
                hidden, params.gate_up_laid,
                bm=block.BM, bn=block.BN, bk=block.BK,
                epilogue="geglu", interleave_products=True,
                product_aware_swiglu=True,
                vmem_limit_bytes=block.STRASSEN_LIMIT)
        projected = block.gemma_rms_norm(
            sp.native_matmul(activated, params.mlp_down),
            params.post_feedforward_norm)
        return (
            residual.astype(jnp.float32) + projected.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return run


def task_metrics(states, checkpoint, token_ids):
    """Top-1 agreement, KL and loss delta on the final logits."""
    final_norm = jnp.asarray(
        checkpoint.tensor(f"{layer.TENSOR_PREFIX}.norm.weight"))
    # Gemma ties the head to the embedding table; there is no lm_head.
    head = jnp.asarray(
        checkpoint.tensor(f"{layer.TENSOR_PREFIX}.embed_tokens.weight")).T
    targets = jnp.asarray(token_ids.reshape(-1)[1:], jnp.int32)
    results = {}
    native = None
    for arm in ARMS:
        hidden = block.gemma_rms_norm(states[arm], final_norm)
        totals = {"loss": 0.0, "kl": 0.0, "top1": 0, "agree": 0,
                  "count": 0}
        rows = hidden.shape[0] - 1
        for start in range(0, rows, LOGIT_CHUNK):
            stop = min(start + LOGIT_CHUNK, rows)
            logits = sp.native_matmul(
                hidden[start:stop], head).astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            target = targets[start:stop]
            totals["loss"] += float(
                -jnp.take_along_axis(logp, target[:, None], 1).sum())
            choice = logits.argmax(-1)
            totals["top1"] += int((choice == target).sum())
            totals["count"] += stop - start
            index = start // LOGIT_CHUNK
            if arm == "regular_xla":
                results.setdefault("_native_logp", []).append(logp)
                results.setdefault("_native_choice", []).append(choice)
            else:
                ref = results["_native_logp"][index]
                totals["kl"] += float((jnp.exp(ref) * (ref - logp)).sum())
                # Agreement is with the native arm's choice, not with the
                # target: the gate asks whether routing changed the model's
                # answer, which is a different question from accuracy.
                totals["agree"] += int(
                    (choice == results["_native_choice"][index]).sum())
        entry = {
            "mean_loss": totals["loss"] / totals["count"],
            "top1_fraction": totals["top1"] / totals["count"],
            "positions": totals["count"],
        }
        if arm == "regular_xla":
            native = entry
        else:
            entry["mean_kl_nats"] = totals["kl"] / totals["count"]
            entry["absolute_loss_delta"] = abs(
                entry["mean_loss"] - native["mean_loss"])
            entry["top1_agreement"] = totals["agree"] / totals["count"]
            entry["passes_task_gate"] = (
                entry["absolute_loss_delta"] <= 0.01
                and entry["mean_kl_nats"] <= 0.02
                and entry["top1_agreement"] >= 0.97)
        results[arm] = entry
    results.pop("_native_logp", None)
    results.pop("_native_choice", None)
    return results


def main():
    if layer.SEQUENCE > block.SLIDING_WINDOW:
        raise ValueError("sequence exceeds the sliding window; add a mask")
    layer.emit = emit
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "layers": NUM_LAYERS, "tile": [block.BM, block.BN, block.BK],
        "arms": list(ARMS), "activation": "geglu",
        "fused_qk": FUSED_QK,
        "policy": ("gate/up + GeGLU"
                   + ("; fused q/k RMSNorm+RoPE" if FUSED_QK else "")
                   + "; o and down left to XLA because a norm separates each "
                     "projection from its residual add"),
        "rope": {"local_base": ROPE_LOCAL_BASE,
                 "global_base": ROPE_GLOBAL_BASE,
                 "global_linear_factor": ROPE_GLOBAL_FACTOR,
                 "sliding_period": SLIDING_PERIOD},
        "scope": "streamed all-layer Gemma 3 27B inference with a task gate",
        "corpus": CORPUS,
    })
    token_ids = make_tokens()
    checkpoint = layer.ShardedSafetensors()
    embeddings = checkpoint.tensor(
        f"{layer.TENSOR_PREFIX}.embed_tokens.weight")
    hidden = jnp.asarray(
        np.asarray(embeddings)[token_ids.reshape(-1)], jnp.bfloat16)
    # Gemma scales the embedding by sqrt(hidden_size) before the first block.
    hidden = (hidden.astype(jnp.float32)
              * math.sqrt(layer.MODEL_DIM)).astype(jnp.bfloat16)
    states = {arm: hidden for arm in ARMS}

    for index in range(NUM_LAYERS):
        started = time.perf_counter()
        params = load_layer(checkpoint, index)
        transfer = time.perf_counter() - started
        executables = {
            arm: jax.jit(make_layer(arm)).lower(states[arm], params).compile()
            for arm in ARMS}
        for _ in range(WARMUPS):
            for arm in ARMS:
                jax.block_until_ready(executables[arm](states[arm], params))
        samples = {arm: [] for arm in ARMS}
        for run in range(RUNS):
            order = ARMS if run % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                start_ns = time.perf_counter_ns()
                out = executables[arm](states[arm], params)
                jax.block_until_ready(out)
                samples[arm].append((time.perf_counter_ns() - start_ns) / 1e6)
        for arm in ARMS:
            states[arm] = jax.block_until_ready(
                executables[arm](states[arm], params))
        # Per-layer divergence from the native path, so the growth curve is
        # visible.  A gate can only say pass or fail; this says where the
        # divergence comes from, which is what the Gemma failure needed.
        native = jnp.asarray(states["regular_xla"], jnp.float32)
        divergence = {
            arm: float(jnp.linalg.norm(jnp.asarray(states[arm], jnp.float32)
                                       - native) / jnp.linalg.norm(native))
            for arm in ARMS if arm != "regular_xla"}
        emit({
            "kind": "layer", "layer": index,
            "l2_relative_vs_xla": divergence,
            "full_attention": is_full_attention(index),
            "transfer_s": transfer,
            "mean_ms": {a: statistics.fmean(v) for a, v in samples.items()},
            "samples_ms": samples,
            "finite": all(bool(jnp.isfinite(states[a]).all()) for a in ARMS),
        })
        del params, executables

    results = task_metrics(states, checkpoint, token_ids)
    emit({"kind": "task", "results": results})
    # This previously reported passes=True unconditionally, so the verdict
    # said nothing about the gate it was summarising.  A gate that always
    # passes is worse than no gate.
    gate_passes = all(
        entry.get("passes_task_gate", True) for entry in results.values())
    emit({"kind": "verdict", "passes": gate_passes,
          "task_passes": gate_passes,
          "scope": "streamed Gemma 3 27B; per-layer timings and task gate"})


if __name__ == "__main__":
    main()
