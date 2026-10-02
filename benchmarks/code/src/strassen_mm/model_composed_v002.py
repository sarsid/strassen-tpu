"""Explicit real-model execution with independently compiled projection calls.

Reuses the preserved qualified model primitives. All policies share the same
native attention prefix, activation, post-projection normalization and residual.
The composition boundary is intentional: per-projection Native compiler options
remain effective, and whole-layer times include identical Python dispatch.
"""
import jax
import jax.numpy as jnp
from . import model_n9_v001 as qm
from . import model_gemma_v002 as gm


def adapter(config):
    return gm if config['model_type'] == 'gemma3_text' else qm


def prefix(config, length, layer_index=0):
    gemma = config['model_type'] == 'gemma3_text'
    mod = adapter(config)
    c = gm.validate_config(config) if gemma else config
    heads, kv = c['num_attention_heads'], c['num_key_value_heads']
    dim = c.get('head_dim', c['hidden_size']//heads)
    eps = c['rms_norm_eps']
    local = gemma and c['layer_types'][layer_index] == 'sliding_attention'
    theta = c['rope_local_base_freq'] if local else c['rope_theta']
    allowed = gm.attention_mask(length, c['sliding_window'] if local else None)

    def call(x, w):
        normalized = mod.rms(x, w['norm1'], eps)
        q = mod.dot(normalized, w['q']).reshape(length, heads, dim)
        k = mod.dot(normalized, w['k']).reshape(length, kv, dim)
        v = mod.dot(normalized, w['v']).reshape(length, kv, dim)
        if gemma or c['model_type'] == 'qwen3':
            q, k = mod.rms(q, w['qnorm'], eps), mod.rms(k, w['knorm'], eps)
        if gemma:
            factor = 1.0 if local else (c.get('rope_scaling') or {}).get('factor', 1.0)
            q, k = mod.rope(q, theta, factor), mod.rope(k, theta, factor)
        else:
            q, k = mod.rope(q, theta), mod.rope(k, theta)
        k, v = jnp.repeat(k, heads//kv, 1), jnp.repeat(v, heads//kv, 1)
        scores = jnp.einsum('thd,shd->hts', q, k, precision=jax.lax.Precision.DEFAULT,
                            preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        scale = c['query_pre_attn_scalar']**-0.5 if gemma else dim**-0.5
        scores = ((scores.astype(jnp.float32)*scale) if gemma else (scores*scale)).astype(jnp.bfloat16)
        if gemma:
            mask = jnp.where(allowed, jnp.asarray(0, jnp.bfloat16), jnp.finfo(jnp.bfloat16).min)
            scores = (scores+mask[None]).astype(jnp.bfloat16)
        else:
            scores = jnp.where(allowed[None], scores, jnp.finfo(jnp.bfloat16).min)
        prob = jax.nn.softmax(scores.astype(jnp.float32), -1).astype(jnp.bfloat16)
        value = jnp.einsum('hts,shd->thd', prob, v, precision=jax.lax.Precision.DEFAULT,
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16).reshape(length, heads*dim)
        projected = mod.dot(value, w['o'])
        if gemma:
            projected = mod.rms(projected, w['norm2'], eps)
        residual = (x+projected).astype(jnp.bfloat16)
        normalized = mod.rms(residual, w['norm3' if gemma else 'norm2'], eps)
        return residual, normalized
    return jax.jit(call)


def activation(config):
    gemma = config['model_type'] == 'gemma3_text'
    def call(output):
        gate, up = jnp.split(output.astype(jnp.bfloat16), 2, -1)
        if gemma:
            gate = gm.gelu(gate)
        else:
            # Match the preserved N8 activation's BF16 SiLU boundary.
            gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        return (gate*up).astype(jnp.bfloat16)
    return jax.jit(call)


def suffix(config):
    gemma = config['model_type'] == 'gemma3_text'
    def call(output, residual, w):
        value = output.astype(jnp.bfloat16)
        if gemma:
            value = gm.rms(value, w['norm4'], config['rms_norm_eps'])
        return (residual+value).astype(jnp.bfloat16)
    return jax.jit(call)


class Layer:
    def __init__(self, prefix_fn, gateup, activate_fn, down, suffix_fn):
        self.prefix, self.gateup, self.activate, self.down, self.suffix = prefix_fn, gateup, activate_fn, down, suffix_fn

    def __call__(self, x, weights):
        residual, normalized = self.prefix(x, weights)
        inner = self.activate(self.gateup(normalized, weights['gateup']))
        return self.suffix(self.down(inner, weights['down']), residual, weights)


def head(checkpoint, hidden, norm, weight):
    c = checkpoint.config
    if c['model_type'] == 'gemma3_text':
        return gm.head_logits(hidden, norm, weight, c)
    return qm.head_logits(hidden, norm, weight, c['rms_norm_eps'])
