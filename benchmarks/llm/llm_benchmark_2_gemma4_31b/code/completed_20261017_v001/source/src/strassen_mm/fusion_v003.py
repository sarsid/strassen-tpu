"""Version 003: enforce XLA BF16 rounding with reduce_precision.

Adapted from public-preview strassen_pallas.py at 95be1fb088656a89813b04492e1d77c66b36ccf9.
No parent imports, environment mutation, allocation, or compatibility monkeypatch.
`model` preserves existing BF16 intermediate boundaries; `accumulator` is a
separate numerical experiment that evaluates epilogues on FP32 accumulators.
The requested output dtype always controls the final store.
"""
from dataclasses import dataclass, asdict
import math
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Epilogue:
    kind: str = 'none'
    rounding: str = 'model'
    norm: str = 'qwen'
    head_dim: int = 128
    eps: float = 1e-6
    early: bool = False

    def __post_init__(self):
        if self.kind not in ('none', 'swiglu', 'geglu', 'residual_add', 'bias_add',
                             'rope', 'qk_norm_rope', 'norm_residual_add'):
            raise ValueError('Unknown epilogue')
        if self.rounding not in ('model', 'accumulator') or self.norm not in ('qwen', 'gemma'):
            raise ValueError('Explicit rounding and normalization conventions required')
        if type(self.head_dim) is not int or self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError('head_dim must be positive and even')
        if not math.isfinite(self.eps) or self.eps <= 0 or type(self.early) is not bool:
            raise ValueError('Invalid epsilon or early-finalization flag')

    @property
    def gated(self):
        return self.kind in ('swiglu', 'geglu')

    @property
    def rotary(self):
        return self.kind in ('rope', 'qk_norm_rope')

    @property
    def auxiliaries(self):
        return {'none': (), 'swiglu': (), 'geglu': (), 'residual_add': ('residual',),
                'bias_add': ('bias',), 'rope': ('cos', 'sin'),
                'qk_norm_rope': ('cos', 'sin', 'scale'),
                'norm_residual_add': ('residual', 'scale')}[self.kind]


def bf32(x):
    # A cast-and-widen round trip may be elided by XLA's excess precision.
    return jax.lax.reduce_precision(x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7)


def store(x, dtype):
    return (bf32(x) if jnp.dtype(dtype)==jnp.dtype(jnp.bfloat16) else x).astype(dtype)


def norm_parts(first, second, inverse_rms, scale_first, scale_second, spec):
    if spec.norm == 'gemma':
        first = first * inverse_rms * (1 + scale_first.astype(jnp.float32))
        second = second * inverse_rms * (1 + scale_second.astype(jnp.float32))
    elif spec.rounding == 'model':
        first = bf32(first * inverse_rms) * bf32(scale_first)
        second = bf32(second * inverse_rms) * bf32(scale_second)
    else:
        first = first * inverse_rms * scale_first.astype(jnp.float32)
        second = second * inverse_rms * scale_second.astype(jnp.float32)
    return (bf32(first), bf32(second)) if spec.rounding == 'model' else (first, second)


def gate(first, second, spec):
    if spec.rounding == 'model':
        first, second = bf32(first), bf32(second)
    value = (jax.nn.gelu(first, approximate=True) if spec.kind == 'geglu' else jax.nn.silu(first))
    if spec.rounding == 'model':
        value = bf32(value)
    return value * second


def rotate(first, second, cosine, sine, spec):
    if spec.rounding == 'model':
        cosine, sine = bf32(cosine), bf32(sine)
        return (bf32(first*cosine) - bf32(second*sine),
                bf32(second*cosine) + bf32(first*sine))
    return first*cosine-second*sine, second*cosine+first*sine


def reference(projected, spec, output_dtype, **aux):
    """Natural-layout XLA epilogue, suitable for a complete jitted baseline."""
    if set(aux) != set(spec.auxiliaries):
        raise ValueError('Incorrect epilogue auxiliary operands')
    x = projected.astype(jnp.float32)
    if spec.kind == 'none':
        return store(x,output_dtype)
    if spec.gated:
        first, second = jnp.split(x, 2, -1)
        return store(gate(first, second, spec),output_dtype)
    if spec.rounding == 'model':
        x = bf32(x)
    if spec.kind == 'residual_add':
        return store(x + aux['residual'].astype(jnp.float32),output_dtype)
    if spec.kind == 'bias_add':
        return store(x + aux['bias'].astype(jnp.float32),output_dtype)
    if spec.kind == 'norm_residual_add':
        inv = jax.lax.rsqrt(jnp.mean(x*x, axis=-1, keepdims=True)+spec.eps)
        # Reuse normalization conventions, without imposing a tile split.
        value, _ = norm_parts(x, x, inv, aux['scale'], aux['scale'], spec)
        return store(value + aux['residual'].astype(jnp.float32),output_dtype)
    heads = x.reshape(x.shape[0], -1, spec.head_dim)
    if spec.kind == 'qk_norm_rope':
        inv = jax.lax.rsqrt(jnp.mean(heads*heads, axis=-1, keepdims=True)+spec.eps)
        heads, _ = norm_parts(heads, heads, inv, aux['scale'], aux['scale'], spec)
    first, second = jnp.split(heads, 2, -1)
    left, right = rotate(first, second, aux['cos'][:, None, :], aux['sin'][:, None, :], spec)
    return store(jnp.concatenate((left, right), -1).reshape(x.shape),output_dtype)


def pack_weights(weight, shape, padded, bn, spec):
    """Pad each paired half independently, then place pairs in one N tile."""
    m, k, n = shape
    mp, kp, npad = padded
    if spec.gated:
        first, second = jnp.split(weight, 2, -1)
        half = npad//2
        first = jnp.pad(first, ((0, kp-k), (0, half-n//2)))
        second = jnp.pad(second, ((0, kp-k), (0, half-n//2)))
    else:
        value = jnp.pad(weight, ((0, kp-k), (0, npad-n)))
        if not spec.rotary:
            return value
        heads = value.reshape(kp, npad//spec.head_dim, spec.head_dim)
        first, second = jnp.split(heads, 2, -1)
        first, second = first.reshape(kp, -1), second.reshape(kp, -1)
    return jnp.stack((first.reshape(kp, -1, bn//2), second.reshape(kp, -1, bn//2)), axis=2).reshape(kp, npad)


def unpack_rotary(value, bn, head_dim):
    rows, width = value.shape
    halves = value.reshape(rows, -1, 2, bn//2).transpose(0, 2, 1, 3)
    return halves.reshape(rows, 2, width//head_dim, head_dim//2).transpose(0, 2, 1, 3).reshape(rows, width)


def plan(arm, shape, spec):
    """Validate a study arm against the fused operation; no compilation."""
    from .kernels_v001 import _positive_triple
    shape = _positive_triple(shape, 'shape (M,K,N)')
    m, k, n = shape
    if arm.get('architecture') != 'v6e':
        raise ValueError('This integration requires an explicit v6e arm')
    dtype = arm['output_dtype']
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('Only separate FP32/BF16 output contracts are supported')
    impl, depth = arm['implementation'], arm['depth']
    if impl not in ('native', 'cubic', 'cubic_full', 'current'):
        raise ValueError('Unsupported fused matrix implementation')
    if (impl == 'current' and depth not in (1, 2)) or (impl != 'current' and depth != 0):
        raise ValueError('Only Native, cubic, S1 and S2 are supported')
    if spec.gated and n % 2:
        raise ValueError('Gate/up projection width must be even')
    if spec.rotary and n % spec.head_dim:
        raise ValueError('Q/K width must contain whole heads')
    mode = arm.get('accumulator', 'products')
    if mode not in ('products', 'outputs'):
        raise ValueError('Unsupported accumulator strategy')
    if spec.early and (impl != 'current' or depth != 1 or mode != 'outputs' or not spec.gated):
        raise ValueError('Early finalization is an explicit S1/output-accumulator gated candidate only')
    tile = None if impl == 'native' else _positive_triple(arm['tile'], 'tile (BM,BN,BK)')
    padded, scratch, extra = shape, (), 0
    if tile:
        bm, bn, bk = tile
        divisor = 2**depth if depth else 2
        alignment = (8*divisor, 128*divisor, 128 if impl == 'cubic_full' else 128*divisor)
        if any(v % a for v, a in zip(tile, alignment)):
            raise ValueError('Tile fails recursive leaf/epilogue alignment')
        if spec.rotary and (bn % (2*spec.head_dim) or bn//spec.head_dim > 128):
            raise ValueError('Rotary tile must pair complete heads and at most 128 heads')
        if spec.kind == 'norm_residual_add' and bn < n:
            raise ValueError('Post-projection RMSNorm needs one tile spanning the full row')
        buffers = arm['buffers']
        if buffers not in (1, 2) or type(buffers) is not int:
            raise ValueError('Fused kernels require explicit one/two input buffers')
        if type(arm['vmem_limit_bytes']) is not int or arm['vmem_limit_bytes'] <= 0:
            raise ValueError('A positive per-kernel VMEM allowance is required')
        padded = tuple(math.ceil(v/t)*t for v, t in zip(shape, (bm, bk, bn)))
        scratch = (bm, bn) if impl == 'cubic_full' else (7 if impl == 'current' and mode == 'products' else 4, bm//2, bn//2)
        extra = 4*bm*bn if 'residual' in spec.auxiliaries else 0
        extra += 4*bn if 'scale' in spec.auxiliaries or 'bias' in spec.auxiliaries else 0
        if spec.rotary:
            extra += 4*bm*bn  # two FP32 half-width position tables
        if spec.kind == 'qk_norm_rope':
            extra += 4*bn*128  # padded reduction and broadcast matrices
        estimated = 4*math.prod(scratch) + buffers*2*(bm*bk+bk*bn) + extra + 2*(2 if dtype == 'bfloat16' else 4)*bm*(bn//2 if spec.gated else bn)
    else:
        estimated = None
    return dict(kernel_version='kernels_fused_v001', architecture='v6e',
        parent_source='95be1fb088656a89813b04492e1d77c66b36ccf9',
        shape_mkn=list(shape), padded_shape_mkn=list(padded), tile_bm_bn_bk=list(tile) if tile else None,
        implementation=impl, depth=depth, accumulator=mode, epilogue=asdict(spec),
        input_dtype='bfloat16', pre_add_dtype='bfloat16', accumulation_dtype='float32', output_dtype=dtype,
        scratch_shape=list(scratch), estimated_vmem_bytes=estimated,
        memory_estimate_is_heuristic=True, vmem_limit_bytes=arm.get('vmem_limit_bytes') if tile else None,
        pipeline_buffers=arm.get('buffers') if tile else None,
        full_contraction=bool(tile and tile[2] == padded[1]),
        norm_reduction_precision='HIGHEST' if spec.kind == 'qk_norm_rope' else 'FP32',
        prepared_call_scope='activation/auxiliary padding, MM, epilogue, crop and rotary output relayout; excludes offline weight preparation',
        complete_call_scope='prepared call plus weight padding/relayout',
        compiler_options=dict(arm.get('compiler_options', {})), process_flags_modified=False)
