"""Gemma 3 text full-prefill adapter, qualified separately before any claim.

Architecture and rounding contract follow Hugging Face Transformers 4.56.2:
https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py
https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/configuration_gemma3.py
https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/masking_utils.py
Those reference implementations are Copyright Google/Hugging Face, Apache-2.0.

Version 002 adds the official Gemma3-12B text backbone from a multimodal checkpoint,
including linear global RoPE and unchanged local RoPE. Image inputs are not evaluated.

Only the gate/up and down matrix products may use the frozen custom MM kernels.
All projections accumulate in FP32 and round to BF16 before their next model
operation. Gemma offset RMS normalization, local/global attention, and tanh GELU
are distinct from the earlier Qwen/Mistral adapter. In particular, down output
is normalized BEFORE residual addition; N8's fused residual kernel is not used.

Scope: evaluation, one unpadded sequence, complete prefill starting at position
zero. This computes the initial-prefill values with or without an initially
empty KV cache, but does not expose or update a cache. Incremental decoding,
nonzero position offsets, padding masks, training, multimodal models, nondefault
RoPE scaling and quantized checkpoints are unsupported. The caller streams one
layer's weights at a time. No accuracy or speed assertion is made by this file.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from .kernels_v002 import make_matmul
from .model_n9_v001 import compare_logits


VERSION = "model_gemma_v002"
REFERENCE_TRANSFORMERS_VERSION = "4.56.2"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_config(config):
    """Return a normalized copy; never change the verified official config."""
    c = dict(config.get("text_config", config))
    if config.get("model_type") == "gemma3":
        c.setdefault("torch_dtype", config.get("torch_dtype", "bfloat16"))
    c.setdefault("vocab_size", 262208)
    if c.get("model_type") != "gemma3_text" or c.get("vision_config") is not None:
        raise ValueError("Only the Gemma 3 text causal language model is supported")
    defaults = {
        "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 131072, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True, "rope_theta": 1000000.0,
        "attention_bias": False, "attention_dropout": 0.0,
        "query_pre_attn_scalar": 256, "sliding_window": 4096,
        "rope_local_base_freq": 10000.0, "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
    }
    for name, value in defaults.items():
        c.setdefault(name, value)
    for name in ("hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim",
                 "vocab_size", "max_position_embeddings", "sliding_window"):
        _positive_int(c.get(name), name)
    if c["head_dim"] % 2 or c["num_attention_heads"] % c["num_key_value_heads"]:
        raise ValueError("RoPE requires even head_dim and integral grouped-query heads")
    for name in ("rms_norm_eps", "rope_theta", "rope_local_base_freq", "query_pre_attn_scalar"):
        c[name] = _positive_float(c[name], name)
    if c["hidden_activation"] != "gelu_pytorch_tanh":
        raise ValueError("Only Gemma's gelu_pytorch_tanh activation is implemented")
    if c["attention_bias"] or c.get("mlp_bias", False):
        raise ValueError("Projection biases are unsupported")
    if c["attention_dropout"] != 0 or c.get("is_encoder_decoder", False):
        raise ValueError("Only dropout-free decoder inference is implemented")
    if c.get("quantization_config") is not None or c.get("rope_parameters") is not None:
        raise ValueError("Quantization and alternate RoPE parameter schemas are unsupported")
    scaling = c.get("rope_scaling")
    if scaling not in (None, {}, {"rope_type": "default"}, {"type": "default"}):
        if (not isinstance(scaling, dict) or set(scaling) != {"rope_type", "factor"}
                or scaling["rope_type"] != "linear"):
            raise ValueError("Only default or linear global RoPE is implemented")
        _positive_float(scaling["factor"], "rope_scaling.factor")
    if c.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Partial rotary dimensions are unsupported")
    if c.get("torch_dtype", c.get("dtype", "bfloat16")) not in (None, "bfloat16"):
        raise ValueError("The model contract requires BF16 activations and matrices")
    if type(c["tie_word_embeddings"]) is not bool:
        raise ValueError("tie_word_embeddings must be boolean")
    # The pinned Gemma3Attention stores this setting but does not pass `softcap`
    # to eager_attention_forward. Do not silently implement a different model.
    if c["attn_logit_softcapping"] is not None:
        raise ValueError("Non-null attention softcap is outside pinned Transformers 4.56.2 semantics")
    if c["final_logit_softcapping"] is not None:
        c["final_logit_softcapping"] = _positive_float(c["final_logit_softcapping"], "final_logit_softcapping")
    types = c.get("layer_types")
    if types is None:
        pattern = c.get("sliding_window_pattern", c.get("_sliding_window_pattern", 6))
        _positive_int(pattern, "sliding_window_pattern")
        types = ["sliding_attention" if (i + 1) % pattern else "full_attention"
                 for i in range(c["num_hidden_layers"])]
    if not isinstance(types, (list, tuple)) or len(types) != c["num_hidden_layers"]:
        raise ValueError("layer_types must name every decoder layer")
    if any(value not in ("sliding_attention", "full_attention") for value in types):
        raise ValueError("Unsupported attention layer type")
    c["layer_types"] = list(types)
    return c


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


class Checkpoint:
    """Verified, memory-mapped official tensors; no full checkpoint device copy."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(), object_pairs_hook=_unique_object)
        self.root = Path(self.manifest["cache_dir"]).resolve()
        self.raw_config = self.manifest["config"]
        self.config = validate_config(self.raw_config)
        self.normalized_config = self.config
        self.locations = {}
        self.verified = []
        if sys.byteorder != "little":
            raise ValueError("Exact BF16 loading requires a little-endian host")
        if not re.fullmatch(r"[0-9a-f]{40}", self.manifest.get("revision", "")):
            raise ValueError("Checkpoint revision must be immutable 40-hex")
        paths = set()
        config_verified = False
        for row in self.manifest["files"]:
            relative = row["path"]
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or relative in paths:
                raise ValueError("Duplicate or escaping checkpoint path")
            paths.add(relative)
            if type(row["bytes"]) is not int or row["bytes"] < 0:
                raise ValueError("Invalid manifest byte count")
            if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError("Invalid manifest SHA256")
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("Checkpoint file size/hash mismatch: " + relative)
            if relative == "config.json":
                if json.loads(path.read_text(), object_pairs_hook=_unique_object) != self.raw_config:
                    raise ValueError("Manifest config differs from verified official config")
                config_verified = True
            self.verified.append(dict(row))
            if path.suffix == ".safetensors":
                self._index_file(path)
        if not config_verified or not self.locations:
            raise ValueError("Verified config.json and safetensors weights are required")
        self._check_tensor_shapes()

    def _index_file(self, path):
        size_on_disk = path.stat().st_size
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors length")
            header_size = struct.unpack("<Q", raw)[0]
            if header_size > 64 * 1024**2 or 8 + header_size > size_on_disk:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(handle.read(header_size), object_pairs_hook=_unique_object)
        intervals = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            # HF's original multimodal serialization prefixes every text tensor.
            if name.startswith("language_model."):
                name = name.removeprefix("language_model.")
            elif name.startswith("model.language_model."):
                name = "model." + name.removeprefix("model.language_model.")
            if name in self.locations:
                raise ValueError("Duplicate checkpoint tensor: " + name)
            shape = info["shape"]
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("Invalid tensor shape")
            dtype = info["dtype"]
            if dtype not in ("BF16", "F32") or (len(shape) > 1 and dtype != "BF16"):
                raise ValueError("Matrices/embeddings must be BF16; only scalar/vector F32 is allowed")
            offsets = info["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError("Invalid safetensors offsets")
            lo, hi = offsets
            width = 2 if dtype == "BF16" else 4
            if lo < 0 or hi - lo != math.prod(shape) * width or 8 + header_size + hi > size_on_disk:
                raise ValueError("Invalid tensor byte bounds")
            self.locations[name] = (path, 8 + header_size + lo, shape, dtype)
            intervals.append((lo, hi))
        cursor = 0
        for lo, hi in sorted(intervals):
            if lo != cursor:
                raise ValueError("Overlapping or unaccounted safetensors payload bytes")
            cursor = hi
        if 8 + header_size + cursor != size_on_disk:
            raise ValueError("Unaccounted trailing safetensors bytes")

    def _check_tensor_shapes(self):
        c = self.normalized_config
        h, inner, dim = c["hidden_size"], c["intermediate_size"], c["head_dim"]
        qwidth, kvwidth = c["num_attention_heads"] * dim, c["num_key_value_heads"] * dim
        shapes = {"model.embed_tokens.weight": [c["vocab_size"], h], "model.norm.weight": [h]}
        if not c["tie_word_embeddings"] or "lm_head.weight" in self.locations:
            shapes["lm_head.weight"] = [c["vocab_size"], h]
        layer_shapes = {
            "self_attn.q_proj.weight": [qwidth, h], "self_attn.k_proj.weight": [kvwidth, h],
            "self_attn.v_proj.weight": [kvwidth, h], "self_attn.o_proj.weight": [h, qwidth],
            "self_attn.q_norm.weight": [dim], "self_attn.k_norm.weight": [dim],
            "mlp.gate_proj.weight": [inner, h], "mlp.up_proj.weight": [inner, h],
            "mlp.down_proj.weight": [h, inner], "input_layernorm.weight": [h],
            "post_attention_layernorm.weight": [h], "pre_feedforward_layernorm.weight": [h],
            "post_feedforward_layernorm.weight": [h],
        }
        for index in range(c["num_hidden_layers"]):
            shapes.update({f"model.layers.{index}.{name}": shape for name, shape in layer_shapes.items()})
        for name, shape in shapes.items():
            if name not in self.locations or self.locations[name][2] != shape:
                raise ValueError("Missing or incorrectly shaped Gemma tensor: " + name)
            if self.locations[name][3] != "BF16":
                # The official qualification loads this checkpoint in BF16.
                # Retaining an F32 norm would silently use a different weight.
                raise ValueError("Required Gemma parameters must be stored BF16: " + name)

    def tensor(self, name):
        path, offset, shape, dtype = self.locations[name]
        if dtype == "BF16":
            return np.memmap(path, mode="r", offset=offset, dtype="<u2", shape=tuple(shape)).view(ml_dtypes.bfloat16)
        return np.memmap(path, mode="r", offset=offset, dtype="<f4", shape=tuple(shape))

    def layer(self, index):
        if type(index) is not int or not 0 <= index < self.normalized_config["num_hidden_layers"]:
            raise ValueError("Layer index out of range")
        prefix = f"model.layers.{index}."
        weights = {}
        for key, suffix in (("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                            ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                            ("down", "mlp.down_proj.weight")):
            weights[key] = np.ascontiguousarray(self.tensor(prefix + suffix).T)
        weights["gateup"] = np.ascontiguousarray(np.concatenate((
            self.tensor(prefix + "mlp.gate_proj.weight"),
            self.tensor(prefix + "mlp.up_proj.weight")), axis=0).T)
        for key, suffix in (("norm1", "input_layernorm.weight"),
                            ("norm2", "post_attention_layernorm.weight"),
                            ("norm3", "pre_feedforward_layernorm.weight"),
                            ("norm4", "post_feedforward_layernorm.weight"),
                            ("qnorm", "self_attn.q_norm.weight"), ("knorm", "self_attn.k_norm.weight")):
            weights[key] = np.array(self.tensor(prefix + suffix), copy=True)
        return weights

    def embeddings(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= self.normalized_config["vocab_size"]):
            raise ValueError("Token IDs must be integers inside the vocabulary")
        values = np.array(self.tensor("model.embed_tokens.weight")[tokens], copy=True)
        scale = np.asarray(math.sqrt(self.normalized_config["hidden_size"]), dtype=ml_dtypes.bfloat16)
        return (values.astype(np.float32) * scale.astype(np.float32)).astype(ml_dtypes.bfloat16)

    def head(self):
        name = "model.embed_tokens.weight" if self.normalized_config["tie_word_embeddings"] else "lm_head.weight"
        # The tied head uses the raw parameter, never the embedding activation scale.
        return np.ascontiguousarray(self.tensor(name).T)


def rounded(x):
    return x.astype(jnp.bfloat16)


def dot(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.DEFAULT,
                      preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def rms(x, weight, eps):
    """Gemma offset RMS: all normalization/weight arithmetic is FP32."""
    value = x.astype(jnp.float32)
    value = value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def gelu(x):
    """PyTorch's tanh-approximate GELU, FP32 computation then input dtype."""
    return jax.nn.gelu(x.astype(jnp.float32), approximate=True).astype(x.dtype)


def scaled_embeddings(values, hidden_size):
    """Scale already-gathered BF16 embeddings by a BF16-rounded sqrt(H)."""
    scale = jnp.asarray(math.sqrt(hidden_size), dtype=jnp.bfloat16)
    return (values.astype(jnp.float32) * scale.astype(jnp.float32)).astype(jnp.bfloat16)


def rotate_half(x):
    return jnp.concatenate((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), axis=-1)


def rope(x, theta, factor=1.0):
    """Unscaled full-prefill RoPE for [sequence, heads, even head_dim]."""
    length, _, dim = x.shape
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inverse = 1.0 / (float(theta) ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    inverse = inverse / float(factor)
    frequency = jnp.arange(length, dtype=jnp.float32)[:, None] * inverse[None, :]
    angle = jnp.concatenate((frequency, frequency), axis=-1)[:, None, :]
    cosine, sine = jnp.cos(angle).astype(x.dtype), jnp.sin(angle).astype(x.dtype)
    # Eager PyTorch materializes both BF16 products before their BF16 sum.
    left = (x * cosine).astype(x.dtype)
    right = (rotate_half(x) * sine).astype(x.dtype)
    return (left + right).astype(x.dtype)


def attention_mask(sequence_length, sliding_window=None):
    """Boolean allowed keys: causal, and key > query - window for local layers."""
    _positive_int(sequence_length, "sequence_length")
    query = jnp.arange(sequence_length)[:, None]
    key = jnp.arange(sequence_length)[None, :]
    allowed = key <= query
    if sliding_window is not None:
        _positive_int(sliding_window, "sliding_window")
        allowed = allowed & (key > query - sliding_window)
    return allowed


def softcap(x, cap):
    """Pinned eager softcap boundaries: divide, tanh, multiply in input dtype."""
    if cap is None:
        return x
    value = (x.astype(jnp.float32) / cap).astype(x.dtype)
    value = jnp.tanh(value.astype(jnp.float32)).astype(x.dtype)
    return (value.astype(jnp.float32) * cap).astype(x.dtype)


def native_policy():
    return {key: {"algorithm": "native", "variant": "plain", "tile": None,
                  "fused": False, "early": False, "packed": False}
            for key in ("gateup", "down")}


def build_layer(config, sequence_length, policy, layer_index=0, interpret=False):
    """Build one JIT layer; the explicit index determines local/global attention."""
    c = validate_config(config)
    _positive_int(sequence_length, "sequence_length")
    if sequence_length > c["max_position_embeddings"]:
        raise ValueError("Sequence exceeds the configured position limit")
    if type(layer_index) is not int or not 0 <= layer_index < c["num_hidden_layers"]:
        raise ValueError("Layer index out of range")
    if set(policy) != {"gateup", "down"}:
        raise ValueError("Policy must specify only gateup and down projections")
    hidden, inner = c["hidden_size"], c["intermediate_size"]
    heads, kvheads, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps = c["rms_norm_eps"]
    is_local = c["layer_types"][layer_index] == "sliding_attention"
    theta = c["rope_local_base_freq"] if is_local else c["rope_theta"]
    allowed = attention_mask(sequence_length, c["sliding_window"] if is_local else None)
    products = {}
    for name, shape in (("gateup", (sequence_length, hidden, 2 * inner)),
                        ("down", (sequence_length, inner, hidden))):
        choice = policy[name]
        if set(choice) - {"algorithm", "variant", "tile", "fused", "early", "packed", "vmem_limit_bytes", "candidate_id"}:
            raise ValueError("Unknown projection policy fields")
        if any(choice.get(option, False) for option in ("fused", "early", "packed")):
            raise ValueError("Gemma requires plain gate/up layout and separate GELU/offset norms; N8 epilogues are invalid")
        algorithm = choice["algorithm"]
        if algorithm not in ("native", "cubic_full", "cubic_quadrant", "strassen"):
            raise ValueError("Unsupported matrix algorithm")
        tile = choice.get("tile")
        if algorithm != "native" and tile is None:
            raise ValueError("Custom projection requires an explicit tile")
        products[name] = make_matmul(algorithm, shape, None if tile is None else tuple(tile),
            variant=choice.get("variant", "plain"), interpret=interpret,
            vmem_limit_bytes=None if algorithm == "native" else choice.get("vmem_limit_bytes", 48 * 1024**2))

    def layer(x, weights):
        if x.shape != (sequence_length, hidden) or x.dtype != jnp.bfloat16:
            raise ValueError("Layer input must be one unpadded [sequence, hidden] BF16 prefill")
        expected = {"q": (hidden, heads * dim), "k": (hidden, kvheads * dim),
                    "v": (hidden, kvheads * dim), "o": (heads * dim, hidden),
                    "gateup": (hidden, 2 * inner), "down": (inner, hidden),
                    "norm1": (hidden,), "norm2": (hidden,), "norm3": (hidden,), "norm4": (hidden,),
                    "qnorm": (dim,), "knorm": (dim,)}
        if set(weights) != set(expected):
            raise ValueError("Layer requires all Gemma projections and six offset RMS weights")
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ValueError("Incorrect layer weight shape: " + name)
            if weights[name].dtype not in (jnp.bfloat16, jnp.float32) or (len(shape) == 2 and weights[name].dtype != jnp.bfloat16):
                raise ValueError("Layer matrices must be BF16; normalization weights BF16 or FP32")
        normalized = rms(x, weights["norm1"], eps)
        q = rms(dot(normalized, weights["q"]).reshape(sequence_length, heads, dim), weights["qnorm"], eps)
        k = rms(dot(normalized, weights["k"]).reshape(sequence_length, kvheads, dim), weights["knorm"], eps)
        v = dot(normalized, weights["v"]).reshape(sequence_length, kvheads, dim)
        factor = 1.0 if is_local else (c.get("rope_scaling") or {}).get("factor", 1.0)
        q, k = rope(q, theta, factor), rope(k, theta, factor)
        k = jnp.repeat(k, heads // kvheads, axis=1)
        v = jnp.repeat(v, heads // kvheads, axis=1)
        scores = jnp.einsum("thd,shd->hts", q, k, precision=jax.lax.Precision.DEFAULT,
                            preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        scores = (scores.astype(jnp.float32) * (c["query_pre_attn_scalar"] ** -0.5)).astype(jnp.bfloat16)
        # Match additive eager mask semantics, rather than changing blocked values
        # by selection. For finite normalized scores, BF16 min remains finite.
        mask = jnp.where(allowed, jnp.asarray(0, jnp.bfloat16), jnp.finfo(jnp.bfloat16).min)
        scores = (scores + mask[None, :, :]).astype(jnp.bfloat16)
        probability = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        value = jnp.einsum("hts,shd->thd", probability, v, precision=jax.lax.Precision.DEFAULT,
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16).reshape(sequence_length, heads * dim)
        residual = (x + rms(dot(value, weights["o"]), weights["norm2"], eps)).astype(jnp.bfloat16)
        normalized = rms(residual, weights["norm3"], eps)
        gate, up = jnp.split(products["gateup"](normalized, weights["gateup"]).astype(jnp.bfloat16), 2, axis=-1)
        activated = (gelu(gate) * up).astype(jnp.bfloat16)
        down = products["down"](activated, weights["down"]).astype(jnp.bfloat16)
        return (residual + rms(down, weights["norm4"], eps)).astype(jnp.bfloat16)

    result = jax.jit(layer)
    result.metadata = {"model_adapter": VERSION, "reference_transformers": REFERENCE_TRANSFORMERS_VERSION,
                       "layer_index": layer_index, "attention_type": c["layer_types"][layer_index],
                       "rope_theta": theta, "query_scaling": c["query_pre_attn_scalar"] ** -0.5,
                       "scope": "single_sequence_full_prefill_position_zero_no_padding",
                       "input_dtype": "bfloat16", "output_dtype": "bfloat16",
                       "projection_metadata": {name: fn.metadata for name, fn in products.items()}}
    return result


@functools.partial(jax.jit, static_argnames=("eps", "cap"))
def _head_logits(hidden, norm, weight, *, eps, cap):
    return softcap(dot(rms(hidden, norm, eps), weight), cap)


def head_logits(hidden, norm, weight, config):
    """Final offset RMS and BF16 head; weight columns may be vocabulary chunks."""
    c = validate_config(config)
    h = c["hidden_size"]
    if hidden.ndim != 2 or hidden.shape[1] != h or hidden.dtype != jnp.bfloat16:
        raise ValueError("Head input must be [positions, hidden] BF16")
    if norm.shape != (h,) or norm.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("Head requires the Gemma final offset RMS weight")
    if weight.ndim != 2 or weight.shape[0] != h or weight.dtype != jnp.bfloat16:
        raise ValueError("Head weight must be transposed raw BF16 vocabulary weights")
    if not 0 < weight.shape[1] <= c["vocab_size"]:
        raise ValueError("Invalid vocabulary chunk width")
    return _head_logits(hidden, norm, weight, eps=c["rms_norm_eps"], cap=c["final_logit_softcapping"])
