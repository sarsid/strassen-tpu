"""Real-weight Qwen3-32B layer-0 three-arm TPU boundary gate.

The checkpoint weights are loaded by HTTP range from an immutable revision.
Attention and all non-MLP work remain ordinary XLA.  The registered arms are
regular XLA, gated rank-8 cubic Pallas, and gated rank-7 Strassen Pallas at the
combined gate/up and down projections.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import NamedTuple


os.environ.setdefault("STRASSEN_VMEM_PROFILE", "max48")

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import requests

import benchmark_common as common
import benchmark_cubic_control as cubic
import mosaic_compat
import strassen_pallas as sp


POLICY = os.environ.get("QWEN3_STRASSEN_POLICY", "both")
if POLICY not in ("both", "up_only"):
    raise ValueError(f"unknown QWEN3_STRASSEN_POLICY={POLICY!r}")
# The dense Qwen3 series shares one architecture (BF16 sharded safetensors,
# q/k RMS norm, rope theta 1e6, head dim 128, untied embeddings), so one
# registry entry per model is the complete difference between study points.
# Revisions are the repository main heads pinned on 2026-08-30.
MODELS = {
    "8b": {
        "repository": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "model_dim": 4096, "intermediate_dim": 12288,
        "heads": 32, "kv_heads": 8, "num_layers": 36,
    },
    "14b": {
        "repository": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "model_dim": 5120, "intermediate_dim": 17408,
        "heads": 40, "kv_heads": 8, "num_layers": 40,
    },
    "32b": {
        "repository": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "model_dim": 5120, "intermediate_dim": 25600,
        "heads": 64, "kv_heads": 8, "num_layers": 64,
    },
}
MODEL_NAME = os.environ.get("QWEN3_MODEL", "32b")
if MODEL_NAME not in MODELS:
    raise ValueError(f"unknown QWEN3_MODEL={MODEL_NAME!r}")
_MODEL = MODELS[MODEL_NAME]
OUTPUT = Path(
    f"/content/results/strassen_qwen3_{MODEL_NAME}_layer_up_only.jsonl"
    if POLICY == "up_only"
    else f"/content/results/strassen_qwen3_{MODEL_NAME}_layer.jsonl"
)
REPOSITORY = _MODEL["repository"]
REVISION = _MODEL["revision"]
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
BATCH, SEQUENCE = 8, 1024
TOKENS = BATCH * SEQUENCE
MODEL_DIM, INTERMEDIATE_DIM = _MODEL["model_dim"], _MODEL["intermediate_dim"]
HEADS, KV_HEADS, HEAD_DIM = _MODEL["heads"], _MODEL["kv_heads"], 128
NUM_LAYERS = _MODEL["num_layers"]
RMS_EPS, ROPE_THETA = 1e-6, 1e6
UP_TILE = (2048, 2048, 512)
DOWN_TILE = (2048, 1024, 512)
CUBIC_LIMIT = 48 * 1024 * 1024
STRASSEN_LIMIT = 47 * 1024 * 1024
WARMUPS, RUNS = 10, 20


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("QWEN3_LAYER_JSON " + line, flush=True)


class ShardedSafetensors:
    def __init__(self):
        self.session = requests.Session()
        index = self._get(f"{BASE}/model.safetensors.index.json").json()
        self.shard_of = index["weight_map"]
        self.headers = {}
        emit({
            "kind": "checkpoint_index",
            "total_size": index.get("metadata", {}).get("total_size"),
            "revision": REVISION,
        })

    def _get(self, url, byte_range=None):
        headers = {"Range": byte_range} if byte_range else {}
        last = None
        for attempt in range(5):
            try:
                response = self.session.get(
                    url, headers=headers, timeout=(30, 900)
                )
                if response.status_code in (200, 206):
                    return response
                raise RuntimeError(f"HTTP status {response.status_code}")
            except (requests.RequestException, RuntimeError) as error:
                last = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"download failed: {last}")

    def _header(self, shard):
        if shard not in self.headers:
            url = f"{BASE}/{shard}"
            prefix = self._get(url, "bytes=0-7").content
            length = int.from_bytes(prefix, "little")
            header = self._get(url, f"bytes=8-{7 + length}").content
            self.headers[shard] = (json.loads(header), 8 + length)
        return self.headers[shard]

    def tensor(self, name):
        shard = self.shard_of[name]
        metadata, data_start = self._header(shard)
        info = metadata[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name} has dtype {info['dtype']}")
        first, last = info["data_offsets"]
        raw = self._get(
            f"{BASE}/{shard}",
            f"bytes={data_start + first}-{data_start + last - 1}",
        ).content
        if len(raw) != last - first:
            raise RuntimeError(
                f"short range for {name}: {len(raw)} != {last - first}"
            )
        emit({
            "kind": "tensor",
            "name": name,
            "shape": info["shape"],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        return np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(
            info["shape"]
        ).copy()


class Parameters(NamedTuple):
    attention_norm: jax.Array
    query: jax.Array
    query_norm: jax.Array
    key: jax.Array
    key_norm: jax.Array
    value: jax.Array
    attention_output: jax.Array
    mlp_norm: jax.Array
    gate_up: jax.Array
    mlp_down: jax.Array
    rope_cos: jax.Array
    rope_sin: jax.Array


def deterministic_hidden():
    row = jnp.arange(TOKENS, dtype=jnp.int32)[:, None]
    col = jnp.arange(MODEL_DIM, dtype=jnp.int32)[None, :]
    x = (((row * 17 + col * 11) % 127).astype(jnp.float32) - 63.0) / 256.0
    x *= 0.75 + (row % 13).astype(jnp.float32) / 32.0
    x = x.astype(jnp.bfloat16)
    jax.block_until_ready(x)
    emit({
        "kind": "input",
        "source": "deterministic nonuniform synthetic embedding-scale state",
        "shape": list(x.shape),
        "sha256": hashlib.sha256(np.asarray(x).tobytes()).hexdigest(),
    })
    return x


def rope_values():
    positions = jnp.arange(SEQUENCE, dtype=jnp.float32)[:, None]
    inverse = 1.0 / (
        ROPE_THETA
        ** (jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM)
    )
    angles = jnp.concatenate((positions * inverse[None, :],) * 2, axis=-1)
    return jnp.cos(angles).astype(jnp.bfloat16), jnp.sin(angles).astype(
        jnp.bfloat16
    )


def rms_norm(x, scale):
    x32 = x.astype(jnp.float32)
    return (
        x32
        * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + RMS_EPS)
        * scale.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def apply_rope(x, cos, sin):
    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    cos = cos[None, :, None, :].astype(jnp.float32)
    sin = sin[None, :, None, :].astype(jnp.float32)
    return (x.astype(jnp.float32) * cos + rotated * sin).astype(jnp.bfloat16)


def native_projection(lhs, rhs):
    return sp.native_matmul(lhs, rhs)


def selected_gemm(lhs, rhs, *, arm, direction):
    if arm == "regular_xla" or (POLICY == "up_only" and direction == "down"):
        return native_projection(lhs, rhs)
    lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    tile = UP_TILE if direction == "up" else DOWN_TILE
    bm, bn, bk = tile
    if arm == "gated_cubic":
        output = cubic.cubic_matmul(
            lhs,
            rhs,
            variant="blocked",
            bm=bm,
            bn=bn,
            bk=bk,
            vmem_limit_bytes=CUBIC_LIMIT,
        )
    elif arm == "gated_strassen":
        output = sp.strassen_matmul(
            lhs,
            rhs,
            bm=bm,
            bn=bn,
            bk=bk,
            interleave_products=(direction == "up"),
            vmem_limit_bytes=STRASSEN_LIMIT,
        )
    else:
        raise ValueError(arm)
    return jax.lax.optimization_barrier(output)


def attention_prefix(x, params):
    normalized = rms_norm(x, params.attention_norm)
    query = native_projection(normalized, params.query).reshape(
        BATCH, SEQUENCE, HEADS, HEAD_DIM
    )
    key = native_projection(normalized, params.key).reshape(
        BATCH, SEQUENCE, KV_HEADS, HEAD_DIM
    )
    value = native_projection(normalized, params.value).reshape(
        BATCH, SEQUENCE, KV_HEADS, HEAD_DIM
    )
    query = rms_norm(query, params.query_norm)
    key = rms_norm(key, params.key_norm)
    query = apply_rope(query, params.rope_cos, params.rope_sin)
    key = apply_rope(key, params.rope_cos, params.rope_sin)
    attended = jax.nn.dot_product_attention(
        query, key, value, is_causal=True, implementation="xla"
    ).reshape(TOKENS, HEADS * HEAD_DIM).astype(jnp.bfloat16)
    attention_output = native_projection(attended, params.attention_output)
    residual = (
        x.astype(jnp.float32) + attention_output.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    return residual, rms_norm(residual, params.mlp_norm)


def make_layer(arm):
    def layer(x, params):
        residual, normalized = attention_prefix(x, params)
        gate_up = selected_gemm(
            normalized, params.gate_up, arm=arm, direction="up"
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = (
            jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        projected = selected_gemm(
            activated, params.mlp_down, arm=arm, direction="down"
        )
        return (
            residual.astype(jnp.float32) + projected.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    return layer


def load_params():
    checkpoint = ShardedSafetensors()

    def weight(name):
        return jnp.asarray(checkpoint.tensor(f"model.layers.0.{name}").T)

    def vector(name):
        return jnp.asarray(checkpoint.tensor(f"model.layers.0.{name}"))

    gate = weight("mlp.gate_proj.weight")
    up = weight("mlp.up_proj.weight")
    rope_cos, rope_sin = rope_values()
    params = Parameters(
        attention_norm=vector("input_layernorm.weight"),
        query=weight("self_attn.q_proj.weight"),
        query_norm=vector("self_attn.q_norm.weight"),
        key=weight("self_attn.k_proj.weight"),
        key_norm=vector("self_attn.k_norm.weight"),
        value=weight("self_attn.v_proj.weight"),
        attention_output=weight("self_attn.o_proj.weight"),
        mlp_norm=vector("post_attention_layernorm.weight"),
        gate_up=jnp.concatenate((gate, up), axis=1),
        mlp_down=weight("mlp.down_proj.weight"),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    jax.block_until_ready(params)
    return params


def paired_interval(reference, candidate):
    deltas = [y - x for x, y in zip(reference, candidate, strict=True)]
    mean = statistics.fmean(deltas)
    half = 2.093 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {"mean_ms": mean, "ci95_ms": [mean - half, mean + half]}


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "repository": REPOSITORY,
        "revision": REVISION,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT
        ),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "dims": {
            "tokens": TOKENS,
            "model": MODEL_DIM,
            "intermediate": INTERMEDIATE_DIM,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
        },
        "arms": ["regular_xla", "gated_cubic", "gated_strassen"],
        "policy": POLICY,
        "scope": "real layer-0 weights; deterministic synthetic input state",
    })
    x = deterministic_hidden()
    params = load_params()
    arms = ("regular_xla", "gated_cubic", "gated_strassen")
    executables = {}
    for arm in arms:
        started = time.perf_counter()
        lowered = jax.jit(make_layer(arm)).lower(x, params)
        stablehlo = lowered.as_text()
        executables[arm] = lowered.compile()
        emit({
            "kind": "compile",
            "arm": arm,
            "seconds": time.perf_counter() - started,
            "stablehlo_custom_calls": stablehlo.count("stablehlo.custom_call"),
            "stablehlo_dots": stablehlo.count("stablehlo.dot_general"),
        })

    outputs = {arm: executable(x, params) for arm, executable in executables.items()}
    jax.block_until_ready(tuple(outputs.values()))
    errors = {
        arm: common.device_error(value, outputs["regular_xla"])
        for arm, value in outputs.items()
        if arm != "regular_xla"
    }
    emit({"kind": "accuracy", "errors_vs_regular_xla": errors})

    for warmup in range(WARMUPS):
        order = arms[warmup % 3 :] + arms[: warmup % 3]
        for arm in order:
            jax.block_until_ready(executables[arm](x, params))
    samples = {arm: [] for arm in arms}
    for run in range(RUNS):
        order = arms[run % 3 :] + arms[: run % 3]
        if run % 2:
            order = tuple(reversed(order))
        for arm in order:
            started = time.perf_counter_ns()
            jax.block_until_ready(executables[arm](x, params))
            samples[arm].append((time.perf_counter_ns() - started) / 1e6)

    timings = {
        arm: common.summarize_samples(values, (TOKENS, 1, 1))
        for arm, values in samples.items()
    }
    # The placeholder shape above makes mean_tflops meaningless for a full
    # layer; remove it rather than imply a layer FLOP model.
    for value in timings.values():
        value.pop("mean_tflops")
    pairs = {
        "gated_cubic_minus_regular_xla": paired_interval(
            samples["regular_xla"], samples["gated_cubic"]
        ),
        "gated_strassen_minus_gated_cubic": paired_interval(
            samples["gated_cubic"], samples["gated_strassen"]
        ),
        "gated_strassen_minus_regular_xla": paired_interval(
            samples["regular_xla"], samples["gated_strassen"]
        ),
    }
    means = {arm: timings[arm]["mean_ms"] for arm in arms}
    passes = (
        pairs["gated_strassen_minus_gated_cubic"]["ci95_ms"][1] < 0
        and pairs["gated_strassen_minus_regular_xla"]["ci95_ms"][1] < 0
    )
    emit({
        "kind": "performance",
        "timings": timings,
        "paired_deltas": pairs,
        "speedups": {
            "strassen_vs_xla": means["regular_xla"]
            / means["gated_strassen"],
            "strassen_vs_cubic": means["gated_cubic"]
            / means["gated_strassen"],
            "cubic_vs_xla": means["regular_xla"] / means["gated_cubic"],
        },
        "passes": passes,
    })
    emit({
        "kind": "verdict",
        "passes": passes,
        "scope": "complete real-weight layer boundary; task gate not run",
    })


if __name__ == "__main__":
    main()
