"""Native BF16 model epilogues retain BF16 operations in the XLA graph.

The v001 arithmetic contract is unchanged. Explicit BF16 operations prevent
the CPU compiler from treating BF16 round-trip expressions as a different
fusion opportunity at the residual/next-normalization boundary.
"""
import jax
import jax.numpy as jnp
from .fusion_v001 import Epilogue, reference as previous_reference


def reference(projected, spec, output_dtype, **aux):
    if spec.rounding != 'model' or jnp.dtype(output_dtype) != jnp.dtype(jnp.bfloat16):
        return previous_reference(projected,spec,output_dtype,**aux)
    if set(aux) != set(spec.auxiliaries):
        raise ValueError('Incorrect epilogue auxiliary operands')
    x = projected.astype(jnp.bfloat16)
    if spec.kind == 'none':
        return x
    if spec.gated:
        first,second = jnp.split(x,2,-1)
        value = jax.nn.gelu(first.astype(jnp.float32),approximate=True) if spec.kind=='geglu' else jax.nn.silu(first.astype(jnp.float32))
        return (value.astype(jnp.bfloat16)*second).astype(jnp.bfloat16)
    if spec.kind in ('residual_add','bias_add'):
        return (x+aux['residual' if spec.kind=='residual_add' else 'bias']).astype(jnp.bfloat16)
    def rms(value,weight):
        f = value.astype(jnp.float32)
        normal = f*jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+spec.eps)
        if spec.norm == 'gemma':
            return (normal*(1+weight.astype(jnp.float32))).astype(jnp.bfloat16)
        return (normal.astype(jnp.bfloat16)*weight.astype(jnp.bfloat16)).astype(jnp.bfloat16)
    if spec.kind == 'norm_residual_add':
        return (rms(x,aux['scale'])+aux['residual']).astype(jnp.bfloat16)
    heads = x.reshape(x.shape[0],-1,spec.head_dim)
    if spec.kind == 'qk_norm_rope':
        heads = rms(heads,aux['scale'])
    first,second = jnp.split(heads,2,-1)
    rotated = jnp.concatenate((-second,first),-1)
    cos = jnp.concatenate((aux['cos'],aux['cos']),-1)[:,None,:].astype(jnp.bfloat16)
    sin = jnp.concatenate((aux['sin'],aux['sin']),-1)[:,None,:].astype(jnp.bfloat16)
    result = ((heads*cos).astype(jnp.bfloat16)+(rotated*sin).astype(jnp.bfloat16)).astype(jnp.bfloat16)
    return result.reshape(x.shape)
