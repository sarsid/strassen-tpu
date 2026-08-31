"""Stream all 64 real Qwen3-32B layers through three TPU policy arms.

The checkpoint cannot reside in one v5e's 16 GB HBM, so one layer is loaded at
a time.  Transfer is recorded but excluded from the registered performance
result.  All arms use ordinary XLA except that the frozen gate/up-only policy
sends the combined MLP gate/up projection through matched cubic or Strassen
Pallas.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import NamedTuple

os.environ["QWEN3_STRASSEN_POLICY"] = "up_only"
os.environ.setdefault("STRASSEN_VMEM_PROFILE", "max48")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as common
import benchmark_cubic_control as cubic
import benchmark_qwen3_32b_layer as layer
import mosaic_compat
import strassen_pallas as sp


PRODUCT_AWARE = os.environ.get("QWEN3_STREAM_PRODUCT_AWARE", "0") == "1"
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
OUTPUT = Path(
    f"/content/results/strassen_qwen3_{layer.MODEL_NAME}"
    f"_streamed_product_inference{SUFFIX}.jsonl"
    if PRODUCT_AWARE else
    f"/content/results/strassen_qwen3_{layer.MODEL_NAME}"
    f"_streamed_inference{SUFFIX}.jsonl")
NUM_LAYERS = layer.NUM_LAYERS
WARMUPS, RUNS = 2, 5
ARMS = ("regular_xla", "gated_cubic", "gated_strassen")
PRODUCT_TILE = tuple(int(v) for v in os.environ.get(
    "QWEN3_PRODUCT_TILE", "2048,2048,512").split(","))
CUBIC_TILE = tuple(int(v) for v in os.environ.get(
    "QWEN3_CUBIC_TILE", ",".join(str(v) for v in PRODUCT_TILE)).split(","))
LOGIT_CHUNK = 256
TEXTS = (
    "The history of mathematics is a conversation between practical problems and abstract ideas. ",
    "A careful scientific experiment changes one condition at a time and records the complete procedure. ",
    "In the early morning the harbor was quiet, and the first ferry moved toward the far shore. ",
    "Computer systems are shaped by arithmetic, memory, communication, scheduling, and numerical formats. ",
    "When readers evaluate an argument, they ask whether the evidence supports the conclusion. ",
    "A forest connects water, soil, fungi, insects, weather, and time beneath the visible landscape. ",
    "Good software leaves a trail: inputs are identified, decisions documented, and failures preserved. ",
    "Music creates expectation through repetition and change, while silence gives shape to the next sound. ",
)


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_STREAM_JSON " + line, flush=True)


def make_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        layer.REPOSITORY, revision=layer.REVISION, use_fast=True, token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [(text + "\n") * 100 for text in TEXTS],
        add_special_tokens=True,
        max_length=layer.SEQUENCE,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    if not np.all(encoded["attention_mask"] == 1):
        raise RuntimeError("text did not fill all registered positions")
    ids = encoded["input_ids"].astype(np.int32)
    emit({
        "kind": "tokens", "shape": list(ids.shape),
        "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "unique_tokens": int(np.unique(ids).size),
    })
    return ids


def device_matrix(checkpoint, name):
    value = jnp.asarray(checkpoint.tensor(name).T)
    jax.block_until_ready(value)
    return value


class StreamParameters(NamedTuple):
    attention_norm: jax.Array
    query: jax.Array
    query_norm: jax.Array
    key: jax.Array
    key_norm: jax.Array
    value: jax.Array
    attention_output: jax.Array
    mlp_norm: jax.Array
    gate_up: jax.Array
    gate_up_laid: jax.Array
    mlp_down: jax.Array
    rope_cos: jax.Array
    rope_sin: jax.Array


def load_layer(checkpoint, index, rope_cos, rope_sin):
    prefix = f"model.layers.{index}"
    gate = device_matrix(checkpoint, f"{prefix}.mlp.gate_proj.weight")
    up = device_matrix(checkpoint, f"{prefix}.mlp.up_proj.weight")
    gate_up = jnp.concatenate((gate, up), axis=1)
    values = dict(
        attention_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.input_layernorm.weight")),
        query=device_matrix(checkpoint, f"{prefix}.self_attn.q_proj.weight"),
        query_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.self_attn.q_norm.weight")),
        key=device_matrix(checkpoint, f"{prefix}.self_attn.k_proj.weight"),
        key_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.self_attn.k_norm.weight")),
        value=device_matrix(checkpoint, f"{prefix}.self_attn.v_proj.weight"),
        attention_output=device_matrix(
            checkpoint, f"{prefix}.self_attn.o_proj.weight"),
        mlp_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")),
        gate_up=gate_up,
        mlp_down=device_matrix(checkpoint, f"{prefix}.mlp.down_proj.weight"),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    if PRODUCT_AWARE:
        params = StreamParameters(
            **values,
            gate_up_laid=sp.swiglu_weight_layout(gate_up, PRODUCT_TILE[1]))
    else:
        params = layer.Parameters(**values)
    jax.block_until_ready(params)
    return params


def make_product_layer(arm):
    bm, bn, bk = PRODUCT_TILE

    def run(x, params):
        residual, normalized = layer.attention_prefix(x, params)
        if arm == "regular_xla":
            full = sp.native_matmul(normalized, params.gate_up)
            gate, up = jnp.split(full, 2, axis=-1)
            activated = (
                jax.nn.silu(gate.astype(jnp.float32))
                * up.astype(jnp.float32)
            ).astype(jnp.bfloat16)
        elif arm == "gated_cubic":
            cubic_bm, cubic_bn, cubic_bk = CUBIC_TILE
            full = cubic.cubic_matmul(
                normalized, params.gate_up, variant="blocked",
                bm=cubic_bm, bn=cubic_bn, bk=cubic_bk,
                vmem_limit_bytes=48 * 1024 * 1024)
            gate, up = jnp.split(full, 2, axis=-1)
            activated = (
                jax.nn.silu(gate.astype(jnp.float32))
                * up.astype(jnp.float32)
            ).astype(jnp.bfloat16)
        else:
            activated = sp.strassen_matmul(
                normalized, params.gate_up_laid,
                bm=bm, bn=bn, bk=bk,
                epilogue="swiglu", interleave_products=True,
                product_aware_swiglu=True,
                vmem_limit_bytes=47 * 1024 * 1024)
        projected = sp.native_matmul(activated, params.mlp_down)
        return (
            residual.astype(jnp.float32) + projected.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return run


def bootstrap_sum_interval(deltas, seed):
    values = np.asarray(deltas, np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True)
    sums = np.sum(draws, axis=1)
    low, high = np.quantile(sums, [0.025, 0.975])
    return {"sum_ms": float(np.sum(values)), "ci95_ms": [float(low), float(high)]}


def logit_chunk_metrics(native_h, cubic_h, strassen_h, lm_head, targets, mask):
    logits = tuple(
        sp.native_matmul(hidden, lm_head).astype(jnp.float32)
        for hidden in (native_h, cubic_h, strassen_h)
    )
    log_probs = tuple(jax.nn.log_softmax(value, axis=-1) for value in logits)
    selected = tuple(
        jnp.take_along_axis(value, targets[:, None], axis=1)[:, 0]
        for value in log_probs
    )
    mask32 = mask.astype(jnp.float32)
    native_probs = jnp.exp(log_probs[0])

    def candidate(index):
        delta = logits[index] - logits[0]
        return (
            -jnp.sum(selected[index] * mask32),
            jnp.sum(
                jnp.sum(native_probs * (log_probs[0] - log_probs[index]), axis=1)
                * mask32),
            jnp.sum(
                (jnp.argmax(logits[index], axis=1) == jnp.argmax(logits[0], axis=1))
                * mask),
            jnp.sum(delta * delta * mask32[:, None]),
            jnp.max(jnp.abs(delta) * mask32[:, None]),
        )

    return (
        -jnp.sum(selected[0] * mask32),
        jnp.sum(logits[0] * logits[0] * mask32[:, None]),
        jnp.max(jnp.abs(logits[0]) * mask32[:, None]),
        candidate(1),
        candidate(2),
        jnp.sum(mask),
    )


def task_metrics(hidden_states, checkpoint, token_ids):
    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    normalized = {
        arm: layer.rms_norm(hidden, final_norm) for arm, hidden in hidden_states.items()
    }
    jax.block_until_ready(normalized)
    lm_head = device_matrix(checkpoint, "lm_head.weight")
    rows = {
        arm: hidden.reshape(layer.BATCH, layer.SEQUENCE, layer.MODEL_DIM)[:, :-1]
        .reshape(-1, layer.MODEL_DIM)
        for arm, hidden in normalized.items()
    }
    targets = token_ids[:, 1:].reshape(-1)
    metric_exec = jax.jit(logit_chunk_metrics).lower(
        *[rows[arm][:LOGIT_CHUNK] for arm in ARMS],
        lm_head,
        jnp.asarray(targets[:LOGIT_CHUNK]),
        jnp.ones((LOGIT_CHUNK,), jnp.int32),
    ).compile()
    totals = {
        "native_loss": 0.0, "native_sq": 0.0, "native_max": 0.0, "count": 0,
        "gated_cubic": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gated_strassen": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    for start in range(0, len(targets), LOGIT_CHUNK):
        stop = min(start + LOGIT_CHUNK, len(targets))
        valid = stop - start
        padded_rows = []
        for arm in ARMS:
            part = rows[arm][start:stop]
            padded_rows.append(jnp.pad(part, ((0, LOGIT_CHUNK - valid), (0, 0))))
        target = np.zeros((LOGIT_CHUNK,), np.int32)
        target[:valid] = targets[start:stop]
        mask = np.zeros((LOGIT_CHUNK,), np.int32)
        mask[:valid] = 1
        result = jax.device_get(metric_exec(
            *padded_rows, lm_head, jnp.asarray(target), jnp.asarray(mask)))
        native_loss, native_sq, native_max, cubic_values, strassen_values, count = result
        totals["native_loss"] += float(native_loss)
        totals["native_sq"] += float(native_sq)
        totals["native_max"] = max(totals["native_max"], float(native_max))
        totals["count"] += int(count)
        for name, values in (("gated_cubic", cubic_values),
                             ("gated_strassen", strassen_values)):
            for index, value in enumerate(values):
                if index == 4:
                    totals[name][index] = max(totals[name][index], float(value))
                else:
                    totals[name][index] += float(value)
    count = totals["count"]
    native_loss = totals["native_loss"] / count
    records = {}
    for name in ("gated_cubic", "gated_strassen"):
        loss, kl, top1, delta_sq, max_delta = totals[name]
        records[name] = {
            "native_loss": native_loss,
            "candidate_loss": loss / count,
            "absolute_loss_delta": abs(loss / count - native_loss),
            "mean_kl_nats": kl / count,
            "top1_agreement": top1 / count,
            "logit_l2_relative": math.sqrt(delta_sq / max(totals["native_sq"], 1e-30)),
            "logit_maxnorm_relative": max_delta / max(totals["native_max"], 1e-30),
            "positions": count,
        }
        records[name]["passes_task_gate"] = (
            records[name]["absolute_loss_delta"] <= 0.01
            and records[name]["top1_agreement"] >= 0.99
            and records[name]["mean_kl_nats"] <= 0.005
        )
    return records


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata", "repository": layer.REPOSITORY,
        "revision": layer.REVISION, "layers": NUM_LAYERS,
        "model": layer.MODEL_NAME,
        "cubic_tile": list(CUBIC_TILE) if PRODUCT_AWARE else None,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(sp.MOSAIC_IR_V7_COMPAT),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "arms": list(ARMS), "policy": (
            "product-aware fused gate/up+SwiGLU; XLA down"
            if PRODUCT_AWARE else "combined gate/up only"),
        "product_tile": list(PRODUCT_TILE) if PRODUCT_AWARE else None,
        "performance_scope": "aggregate resident layer compute; transfer excluded",
        "warmups_per_layer": WARMUPS, "runs_per_layer": RUNS,
    })
    token_ids = make_tokens()
    checkpoint = layer.ShardedSafetensors()
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(layer.TOKENS, layer.MODEL_DIM).copy()
    del embeddings
    hidden = {arm: jnp.asarray(initial) for arm in ARMS}
    jax.block_until_ready(hidden)
    del initial
    rope_cos, rope_sin = layer.rope_values()
    first = load_layer(checkpoint, 0, rope_cos, rope_sin)
    executables = {}
    for arm in ARMS:
        started = time.perf_counter()
        layer_function = (
            make_product_layer(arm) if PRODUCT_AWARE else layer.make_layer(arm))
        lowered = jax.jit(layer_function).lower(hidden[arm], first)
        stablehlo = lowered.as_text()
        executables[arm] = lowered.compile()
        emit({
            "kind": "compile", "arm": arm,
            "seconds": time.perf_counter() - started,
            "stablehlo_custom_calls": stablehlo.count("stablehlo.custom_call"),
            "stablehlo_dots": stablehlo.count("stablehlo.dot_general"),
        })

    per_layer = {arm: [] for arm in ARMS}
    all_finite = True
    for index in range(NUM_LAYERS):
        transfer_started = time.perf_counter()
        params = first if index == 0 else load_layer(checkpoint, index, rope_cos, rope_sin)
        transfer_s = time.perf_counter() - transfer_started
        for warmup in range(WARMUPS):
            order = ARMS[(index + warmup) % 3 :] + ARMS[: (index + warmup) % 3]
            for arm in order:
                jax.block_until_ready(executables[arm](hidden[arm], params))
        samples = {arm: [] for arm in ARMS}
        next_hidden = {}
        for run in range(RUNS):
            order = ARMS[(index + run) % 3 :] + ARMS[: (index + run) % 3]
            if run % 2:
                order = tuple(reversed(order))
            for arm in order:
                started = time.perf_counter_ns()
                value = executables[arm](hidden[arm], params)
                jax.block_until_ready(value)
                samples[arm].append((time.perf_counter_ns() - started) / 1e6)
                next_hidden[arm] = value
        hidden = next_hidden
        finite = bool(jax.device_get(jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value)) for value in hidden.values()
        ]))))
        all_finite = all_finite and finite
        errors = {
            arm: common.device_error(value, hidden["regular_xla"])
            for arm, value in hidden.items() if arm != "regular_xla"
        }
        means = {arm: statistics.fmean(values) for arm, values in samples.items()}
        for arm in ARMS:
            per_layer[arm].append(means[arm])
        emit({
            "kind": "layer", "layer": index, "transfer_s": transfer_s,
            "mean_ms": means, "samples_ms": samples,
            "errors_vs_regular_xla": errors, "finite": finite,
        })
        del params
        if index == 0:
            del first
        gc.collect()

    aggregate_ms = {arm: float(sum(values)) for arm, values in per_layer.items()}
    deltas_xla = [s - x for x, s in zip(
        per_layer["regular_xla"], per_layer["gated_strassen"], strict=True)]
    deltas_cubic = [s - c for c, s in zip(
        per_layer["gated_cubic"], per_layer["gated_strassen"], strict=True)]
    performance = {
        "aggregate_ms": aggregate_ms,
        "speedup_strassen_vs_xla": aggregate_ms["regular_xla"]
        / aggregate_ms["gated_strassen"],
        "speedup_strassen_vs_cubic": aggregate_ms["gated_cubic"]
        / aggregate_ms["gated_strassen"],
        "strassen_minus_xla": bootstrap_sum_interval(deltas_xla, 20260829),
        "strassen_minus_cubic": bootstrap_sum_interval(deltas_cubic, 20260830),
    }
    performance["passes"] = (
        performance["strassen_minus_xla"]["ci95_ms"][1] < 0
        and performance["strassen_minus_cubic"]["ci95_ms"][1] < 0
    )
    emit({"kind": "aggregate_performance", **performance})
    tasks = task_metrics(hidden, checkpoint, token_ids)
    emit({"kind": "task", "results": tasks})
    passes = all_finite and performance["passes"] and tasks[
        "gated_strassen"]["passes_task_gate"]
    emit({
        "kind": "verdict", "passes": passes,
        "performance_passes": performance["passes"],
        "task_passes": tasks["gated_strassen"]["passes_task_gate"],
        "all_finite": all_finite,
    })


if __name__ == "__main__":
    main()
