"""Real-checkpoint Mistral layer gate for the qualified TPU MLP shapes.

The layer uses pinned Mistral-7B-v0.1 BF16 weights and tokenizer-produced
language tokens.  The controlled comparator combines the real gate/up weights
for both native and Strassen; a standard two-projection native policy is also
reported as a model-faithful reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import NamedTuple


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import strassen_pallas as sp

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import requests

import benchmark_common as bench


OUTPUT = Path("/content/results/strassen_checkpoint_layer.jsonl")
REPOSITORY = "mistralai/Mistral-7B-v0.1"
REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
SHARD = "model-00001-of-00002.safetensors"
SHARD_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{SHARD}"

BATCH = 8
SEQUENCE = 1024
TOKENS = BATCH * SEQUENCE
MODEL_DIM = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = MODEL_DIM // HEADS
INTERMEDIATE_DIM = 14336
WARMUPS = 10
RUNS = 20

POLICIES = {
    "checkpoint_native": {"combine_gate_up": False, "strassen": frozenset()},
    "matched_native": {"combine_gate_up": True, "strassen": frozenset()},
    "mlp_up": {"combine_gate_up": True, "strassen": frozenset({"mlp_up"})},
    "mlp_down": {
        "combine_gate_up": True,
        "strassen": frozenset({"mlp_down"}),
    },
    "both": {
        "combine_gate_up": True,
        "strassen": frozenset({"mlp_up", "mlp_down"}),
    },
}

TEXTS = (
    "The history of mathematics is a conversation between practical problems "
    "and abstract ideas. A method invented for navigation may later become a "
    "theorem, while a theorem may eventually guide an engineer.",
    "A careful scientific experiment changes one condition at a time, records "
    "the complete procedure, and treats an unexpected result as information "
    "rather than an inconvenience.",
    "In the early morning the harbor was quiet. Ropes tapped against the masts, "
    "a gull crossed the pale sky, and the first ferry moved slowly toward the "
    "far shore.",
    "Computer systems are shaped by several limits at once: arithmetic units, "
    "memory capacity, communication bandwidth, scheduling, numerical formats, "
    "and the structure of the program.",
    "When readers evaluate an argument, they ask whether the evidence supports "
    "the conclusion, whether alternatives were tested, and whether uncertainty "
    "has been described plainly.",
    "A forest is not merely a collection of trees. Water, soil, fungi, insects, "
    "weather, and time connect the visible landscape to processes beneath it.",
    "Good software leaves a trail that another person can follow. Inputs are "
    "identified, decisions are documented, tests are repeatable, and failures "
    "remain available as evidence.",
    "Music creates expectation through repetition and change. A phrase returns "
    "with a different harmony, a rhythm pauses, and silence gives shape to the "
    "next sound.",
)


class Parameters(NamedTuple):
    attention_norm: jax.Array
    query: jax.Array
    key: jax.Array
    value: jax.Array
    attention_output: jax.Array
    mlp_norm: jax.Array
    gate_up: jax.Array
    mlp_down: jax.Array
    rope_cos: jax.Array
    rope_sin: jax.Array


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_LAYER_JSON " + line, flush=True)


class RemoteSafetensors:
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        prefix = self._range(0, 7)
        header_length = int.from_bytes(prefix, "little")
        header = self._range(8, 7 + header_length)
        self.metadata = json.loads(header)
        self.data_start = 8 + header_length

    def _range(self, start, end):
        last_error = None
        for attempt in range(3):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 600),
                )
                if response.status_code != 206:
                    raise RuntimeError(
                        f"range {start}-{end} returned {response.status_code}"
                    )
                expected = end - start + 1
                if len(response.content) != expected:
                    raise RuntimeError(
                        f"range {start}-{end} returned {len(response.content)} bytes"
                    )
                return response.content
            except (requests.RequestException, RuntimeError) as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"checkpoint range download failed: {last_error}")

    def tensor(self, name):
        metadata = self.metadata[name]
        if metadata["dtype"] != "BF16":
            raise ValueError(f"{name} has unsupported dtype {metadata['dtype']}")
        first, last = metadata["data_offsets"]
        raw = self._range(self.data_start + first, self.data_start + last - 1)
        digest = hashlib.sha256(raw).hexdigest()
        array = np.frombuffer(raw, dtype=ml_dtypes.bfloat16)
        array = array.reshape(metadata["shape"]).copy()
        emit({
            "kind": "tensor",
            "name": name,
            "shape": metadata["shape"],
            "bytes": len(raw),
            "sha256": digest,
        })
        return array


def make_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        REPOSITORY,
        revision=REVISION,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    expanded = [(text + "\n") * 80 for text in TEXTS]
    encoded = tokenizer(
        expanded,
        add_special_tokens=True,
        max_length=SEQUENCE,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    if not np.all(encoded["attention_mask"] == 1):
        raise RuntimeError("text batch did not fill every sequence position")
    token_ids = encoded["input_ids"].astype(np.int32)
    emit({
        "kind": "tokens",
        "shape": list(token_ids.shape),
        "sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "unique_tokens": int(np.unique(token_ids).size),
    })
    return token_ids


def device_matrix(checkpoint, name):
    host = checkpoint.tensor(name)
    value = jnp.asarray(host.T)
    jax.block_until_ready(value)
    return value


def load_inputs():
    token_ids = make_tokens()
    checkpoint = RemoteSafetensors(SHARD_URL)
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    x = jnp.asarray(embeddings[token_ids].reshape(TOKENS, MODEL_DIM))

    gate = device_matrix(checkpoint, "model.layers.0.mlp.gate_proj.weight")
    up = device_matrix(checkpoint, "model.layers.0.mlp.up_proj.weight")
    gate_up = jnp.concatenate((gate, up), axis=1)
    rope_cos, rope_sin = rope_values()
    params = Parameters(
        attention_norm=jnp.asarray(
            checkpoint.tensor("model.layers.0.input_layernorm.weight")
        ),
        query=device_matrix(
            checkpoint, "model.layers.0.self_attn.q_proj.weight"
        ),
        key=device_matrix(checkpoint, "model.layers.0.self_attn.k_proj.weight"),
        value=device_matrix(
            checkpoint, "model.layers.0.self_attn.v_proj.weight"
        ),
        attention_output=device_matrix(
            checkpoint, "model.layers.0.self_attn.o_proj.weight"
        ),
        mlp_norm=jnp.asarray(
            checkpoint.tensor("model.layers.0.post_attention_layernorm.weight")
        ),
        gate_up=gate_up,
        mlp_down=device_matrix(
            checkpoint, "model.layers.0.mlp.down_proj.weight"
        ),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    jax.block_until_ready((x, params))
    return x, params


def rope_values():
    positions = jnp.arange(SEQUENCE, dtype=jnp.float32)[:, None]
    inverse_frequency = 1.0 / (
        10000.0
        ** (jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM)
    )
    frequencies = positions * inverse_frequency[None, :]
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)
    return (
        jnp.cos(angles).astype(jnp.bfloat16),
        jnp.sin(angles).astype(jnp.bfloat16),
    )


def rms_norm(x, scale):
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + 1e-5) * scale).astype(jnp.bfloat16)


def apply_rope(x, cos, sin):
    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    cos = cos[None, :, None, :].astype(jnp.float32)
    sin = sin[None, :, None, :].astype(jnp.float32)
    return (x.astype(jnp.float32) * cos + rotated * sin).astype(jnp.bfloat16)


def gemm(lhs, rhs, use_strassen=False):
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    output = (
        sp.tuned_matmul(lhs, rhs)
        if use_strassen
        else sp.native_matmul(lhs, rhs)
    )
    return jax.lax.optimization_barrier(output)


def make_layer(policy_name):
    policy = POLICIES[policy_name]
    selected = policy["strassen"]

    def layer(x, params):
        normalized = rms_norm(x, params.attention_norm)
        query = gemm(normalized, params.query)
        key = gemm(normalized, params.key)
        value = gemm(normalized, params.value)
        query = query.reshape(BATCH, SEQUENCE, HEADS, HEAD_DIM)
        key = key.reshape(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
        value = value.reshape(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
        query = apply_rope(query, params.rope_cos, params.rope_sin)
        key = apply_rope(key, params.rope_cos, params.rope_sin)
        attended = jax.nn.dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            implementation="xla",
        )
        attended = attended.reshape(TOKENS, MODEL_DIM).astype(jnp.bfloat16)
        attention_output = gemm(attended, params.attention_output)
        residual = (
            x.astype(jnp.float32) + attention_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

        normalized = rms_norm(residual, params.mlp_norm)
        if policy["combine_gate_up"]:
            gate_up = gemm(
                normalized, params.gate_up, "mlp_up" in selected
            )
            gate, up = jnp.split(gate_up, 2, axis=-1)
        else:
            gate = gemm(normalized, params.gate_up[:, :INTERMEDIATE_DIM])
            up = gemm(normalized, params.gate_up[:, INTERMEDIATE_DIM:])
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        mlp_output = gemm(
            activated, params.mlp_down, "mlp_down" in selected
        )
        return (
            residual.astype(jnp.float32) + mlp_output.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def lowered_audit(lowered):
    text = lowered.as_text()
    terms = (
        "stablehlo.dot_general",
        "stablehlo.optimization_barrier",
        "stablehlo.custom_call",
    )
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text),
        "term_counts": {term: text.count(term) for term in terms},
    }


def compile_policies(x, params):
    executables = {}
    for name in POLICIES:
        started = time.perf_counter()
        lowered = jax.jit(make_layer(name)).lower(x, params)
        audit = lowered_audit(lowered)
        executables[name] = lowered.compile()
        emit({
            "kind": "compile",
            "policy": name,
            "compile_s": time.perf_counter() - started,
            "lowered_audit": audit,
        })
    return executables


def accuracy(executables, x, params):
    outputs = {
        name: executable(x, params)
        for name, executable in executables.items()
    }
    jax.block_until_ready(outputs)
    checkpoint_reference = outputs["checkpoint_native"]
    matched_reference = outputs["matched_native"]
    records = {}
    for name, output in outputs.items():
        vs_checkpoint = bench.device_error(output, checkpoint_reference)
        vs_matched = bench.device_error(output, matched_reference)
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(output))))
        if name == "matched_native":
            passes = (
                finite
                and vs_checkpoint["maxnorm_relative"] <= 0.001
                and vs_checkpoint["l2_relative"] <= 0.001
            )
        else:
            passes = (
                finite
                and vs_matched["maxnorm_relative"] <= 0.01
                and vs_matched["l2_relative"] <= 0.01
            )
        records[name] = {
            "finite": finite,
            "vs_checkpoint_native": vs_checkpoint,
            "vs_matched_native": vs_matched,
            "passes": passes,
        }
        emit({"kind": "accuracy", "policy": name, **records[name]})
    return records


def bootstrap_delta_interval(reference, candidate, seed):
    deltas = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(deltas, size=(20000, len(deltas)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(deltas)), float(low), float(high)


def timing(executables, x, params):
    names = tuple(executables)
    for warmup in range(WARMUPS):
        offset = warmup % len(names)
        for name in names[offset:] + names[:offset]:
            jax.block_until_ready(executables[name](x, params))

    samples = {name: [] for name in names}
    for run in range(RUNS):
        offset = run % len(names)
        order = names[offset:] + names[:offset]
        if run % 2:
            order = tuple(reversed(order))
        for name in order:
            started = time.perf_counter_ns()
            jax.block_until_ready(executables[name](x, params))
            samples[name].append((time.perf_counter_ns() - started) / 1e6)

    reference_mean = statistics.fmean(samples["matched_native"])
    records = {}
    for index, name in enumerate(names):
        values = samples[name]
        mean_ms = statistics.fmean(values)
        delta, low, high = bootstrap_delta_interval(
            samples["matched_native"], values, 20260817 + index
        )
        records[name] = {
            "mean_ms": mean_ms,
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "std_ms": statistics.pstdev(values),
            "samples_ms": values,
            "speedup_vs_matched_native": reference_mean / mean_ms,
            "candidate_minus_matched_mean_ms": delta,
            "candidate_minus_matched_bootstrap_95ci_ms": [low, high],
            "statistically_faster": name != "matched_native" and high < 0.0,
        }
        emit({"kind": "performance", "policy": name, **records[name]})
    return records


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "repository": REPOSITORY,
        "revision": REVISION,
        "checkpoint_contract": "Mistral-7B-v0.1 BF16 layer 0",
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "profile": sp.VMEM_PROFILE,
        "tile": [sp.TUNED_BM, sp.TUNED_BN, sp.TUNED_BK],
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "model_dim": MODEL_DIM,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "intermediate_dim": INTERMEDIATE_DIM,
        },
        "policies": {
            name: {
                "combine_gate_up": value["combine_gate_up"],
                "strassen": sorted(value["strassen"]),
            }
            for name, value in POLICIES.items()
        },
        "warmups": WARMUPS,
        "runs": RUNS,
    })
    x, params = load_inputs()
    executables = compile_policies(x, params)
    accuracy_records = accuracy(executables, x, params)
    performance_records = timing(executables, x, params)
    accuracy_passes = all(record["passes"] for record in accuracy_records.values())
    primary_performance_passes = performance_records["both"]["statistically_faster"]
    success = accuracy_passes and primary_performance_passes
    emit({
        "kind": "final",
        "success": success,
        "accuracy_passes": accuracy_passes,
        "primary_performance_passes": primary_performance_passes,
    })
    if not success:
        raise RuntimeError("checkpoint-layer gate failed; inspect emitted records")


if __name__ == "__main__":
    main()
