"""Native Gemma4 dense text prefill for independent numerical qualification.

BF16 model boundaries, FP32 matrix accumulation, DEFAULT dot precision. This
version does not provide custom kernels, checkpoint loading, cache integration,
decode, padding masks, multimodal inputs, or performance evidence.
Reference: pinned Transformers 5.5.0, retained in third_party.
"""
import math
import jax
import jax.numpy as jnp
from .gemma4_contract_v001 import validate_config, layer_geometry
from .fusion_v003 import bf32, store

VERSION = "model_gemma4_v001"


def rounded(x):
    return store(x, jnp.bfloat16)


def dot(a, b):
    return rounded(jnp.matmul(a, b, precision=jax.lax.Precision.DEFAULT,
                              preferred_element_type=jnp.float32))


def rms(x, weight, eps):
    """Direct FP32 learned scale, no Gemma3 offset or Qwen intermediate cast."""
    value = x.astype(jnp.float32)
    value = value * jnp.power(jnp.mean(value * value, axis=-1, keepdims=True) + eps, -0.5)
    if weight is not None:
        value = value * weight.astype(jnp.float32)
    return rounded(value)


def rotary_tables(config, layer_index, positions):
    c = validate_config(config)
    g = layer_geometry(c, layer_index)
    dim = g["head_dim"]
    rp = c["rope_parameters"][g["layer_type"]]
    angles = dim // 2 if g["layer_type"] == "sliding_attention" else int(rp["partial_rotary_factor"] * dim // 2)
    # Proportional RoPE divides exponents by the full width and pads frequencies
    # with zeros. The rotated pairs straddle the two full-width halves.
    inverse = 1.0 / (rp["rope_theta"] ** (jnp.arange(0, 2 * angles, 2, dtype=jnp.float32) / dim))
    inverse = jnp.pad(inverse, (0, dim // 2 - angles))
    phase = jnp.asarray(positions, jnp.float32)[..., None] * inverse
    phase = jnp.concatenate((phase, phase), axis=-1)
    return rounded(jnp.cos(phase)), rounded(jnp.sin(phase))


def apply_rotary(x, cosine, sine):
    cosine, sine = cosine[..., None, :].astype(jnp.float32), sine[..., None, :].astype(jnp.float32)
    x = x.astype(jnp.float32)
    rotated = jnp.concatenate((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), axis=-1)
    return rounded(bf32(x * cosine) + bf32(rotated * sine))


def attention_mask(sequence, window=None):
    position = jnp.arange(sequence)
    allowed = position[None, :] <= position[:, None]
    if window is not None:
        allowed &= position[None, :] > position[:, None] - window
    return jnp.where(allowed, jnp.asarray(0, jnp.bfloat16), jnp.finfo(jnp.bfloat16).min)


def scaled_embeddings(values, hidden_size):
    return rounded(values.astype(jnp.float32) * bf32(jnp.asarray(math.sqrt(hidden_size), jnp.float32)))


def head_logits(hidden, norm, transposed_weight, config):
    c = validate_config(config)
    value = dot(rms(hidden, norm, c["rms_norm_eps"]), transposed_weight)
    cap = c.get("final_logit_softcapping")
    if cap is not None:
        value = rounded(value.astype(jnp.float32) / cap)
        value = rounded(jnp.tanh(value.astype(jnp.float32)))
        value = rounded(value.astype(jnp.float32) * cap)
    return value


def build_layer(config, sequence_length, *, batch_size=1, layer_index=0):
    """One native jitted layer, input [B,S,H], canonical transposed weights."""
    c = validate_config(config)
    g = layer_geometry(c, layer_index, batch_size, sequence_length)
    h, inner = c["hidden_size"], c["intermediate_size"]
    heads, kv, dim = g["query_heads"], g["kv_heads"], g["head_dim"]
    expected = {site: tuple(shape[1:]) for site, shape in g["projection_shapes_mkn"].items()}
    expected.update({k: (h,) for k in ("norm1", "norm2", "norm3", "norm4")})
    expected.update(qnorm=(dim,), knorm=(dim,), layer_scalar=(1,))
    cosine, sine = rotary_tables(c, layer_index, jnp.arange(sequence_length)[None, :])
    mask = attention_mask(sequence_length, g["sliding_window"])[None, None, :, :]
    eps = c["rms_norm_eps"]

    def layer(x, weights):
        if x.shape != (batch_size, sequence_length, h) or x.dtype != jnp.bfloat16:
            raise ValueError("Expected fixed [batch, sequence, hidden] BF16 input")
        if set(weights) != set(expected):
            raise ValueError("Missing or unexpected Gemma4 layer state")
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ValueError("Incorrect Gemma4 weight shape: " + name)
            if weights[name].dtype != jnp.bfloat16 and not (name == "layer_scalar" and weights[name].dtype == jnp.float32):
                raise ValueError("Gemma4 parameters must be BF16; scalar may be FP32")
        value = rms(x, weights["norm1"], eps)
        q = dot(value, weights["q"]).reshape(batch_size, sequence_length, heads, dim)
        raw_k = dot(value, weights["k"]).reshape(batch_size, sequence_length, kv, dim)
        raw_v = raw_k if g["shared_raw_kv"] else dot(value, weights["v"]).reshape(batch_size, sequence_length, kv, dim)
        q = apply_rotary(rms(q, weights["qnorm"], eps), cosine, sine)
        k = apply_rotary(rms(raw_k, weights["knorm"], eps), cosine, sine)
        v = rms(raw_v, None, eps)
        k = jnp.repeat(k, heads // kv, axis=2)
        v = jnp.repeat(v, heads // kv, axis=2)
        scores = rounded(jnp.einsum("bthd,bshd->bhts", q, k,
                         precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32))
        # Gemma4 attention scale is exactly 1.0.
        scores = rounded(scores.astype(jnp.float32) + mask.astype(jnp.float32))
        probability = rounded(jax.nn.softmax(scores.astype(jnp.float32), axis=-1))
        value = rounded(jnp.einsum("bhts,bshd->bthd", probability, v,
                        precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32))
        value = value.reshape(batch_size, sequence_length, heads * dim)
        residual = rounded(x.astype(jnp.float32) + rms(dot(value, weights["o"]), weights["norm2"], eps).astype(jnp.float32))
        value = rms(residual, weights["norm3"], eps)
        gate, up = jnp.split(dot(value, weights["gateup"]), 2, axis=-1)
        activated = rounded(bf32(jax.nn.gelu(gate.astype(jnp.float32), approximate=True)) * up.astype(jnp.float32))
        value = rms(dot(activated, weights["down"]), weights["norm4"], eps)
        value = rounded(residual.astype(jnp.float32) + value.astype(jnp.float32))
        return rounded(value.astype(jnp.float32) * weights["layer_scalar"].astype(jnp.float32))

    result = jax.jit(layer)
    result.metadata = dict(adapter=VERSION, **g, output_dtype="bfloat16", precision="DEFAULT",
                           scope="native_numerical_qualification_only")
    return result
