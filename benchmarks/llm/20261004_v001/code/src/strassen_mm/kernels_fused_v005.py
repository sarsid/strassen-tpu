"""Campaign v6e fusion: DEFAULT dots, explicit model rounding, S1/S2 only.

Version 005 adds Gemma4 direct-scale RMS semantics via fusion_v004.
Native/custom arithmetic, pipeline choices and tuning controls are unchanged.
This is not deployed to the active Qwen campaign.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base
from . import kernels_v6e_v001 as arithmetic
from .kernels_two_level_v001 import ORDER, TARGETS, quarters, operands
from .fusion_v004 import Epilogue, bf32, gate, rotate, norm_parts, reference, pack_weights, unpack_rotary, plan


def _pair(first, second, aux, spec, width):
    if spec.gated:
        return (gate(first, second, spec),)
    if spec.kind != 'none' and spec.rounding == 'model':
        first, second = bf32(first), bf32(second)
    half = first.shape[1]
    if spec.kind == 'residual_add':
        return first+aux['residual'][:, :half], second+aux['residual'][:, half:]
    if spec.kind == 'bias_add':
        return first+aux['bias'][:, :half], second+aux['bias'][:, half:]
    if spec.kind == 'norm_residual_add':
        squares = jnp.sum(first*first, 1, keepdims=True)+jnp.sum(second*second, 1, keepdims=True)
        inv = jax.lax.rsqrt(squares/width+spec.eps)
        first, second = norm_parts(first, second, inv, aux['scale'][:, :half], aux['scale'][:, half:], spec)
        return first+aux['residual'][:, :half], second+aux['residual'][:, half:]
    if spec.kind == 'qk_norm_rope':
        # Match parent DEFAULT dot precision, retaining FP32 accumulation.
        # The rounding difference is measured, not treated as exact algebra.
        dot = functools.partial(jnp.dot, precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32)
        sums = dot(first*first, aux['segment'])+dot(second*second, aux['segment'])
        inv = dot(jax.lax.rsqrt(sums/spec.head_dim+spec.eps), aux['segment_t'])
        first, second = norm_parts(first, second, inv, aux['scale'][:, :half], aux['scale'][:, half:], spec)
    if spec.rotary:
        return rotate(first, second, aux['cos'], aux['sin'], spec)
    return first, second


def _kernel(a_ref, b_ref, *refs, impl, depth, mode, nk, spec, aux_names, width):
    out_ref, acc_ref = refs[-2:]
    aux_refs = dict(zip(aux_names, refs[:-2]))
    bm, bk = a_ref.shape
    bn = b_ref.shape[1]
    hm, hn = bm//2, bn//2
    step = pl.program_id(2)
    @pl.when(step == 0)
    def zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    def finish_pair(first, second, top):
        rows = slice(0, hm) if top else slice(hm, bm)
        aux = {name: (ref[rows, :].astype(jnp.float32) if name in ('residual', 'cos', 'sin') else ref[...].astype(jnp.float32))
               for name, ref in aux_refs.items()}
        values = _pair(first, second, aux, spec, width)
        if spec.gated:
            out_ref[rows, :] = values[0].astype(out_ref.dtype)
        else:
            out_ref[rows, :hn] = values[0].astype(out_ref.dtype)
            out_ref[rows, hn:] = values[1].astype(out_ref.dtype)

    if impl == 'cubic_full':
        acc_ref[...] += base._dot(a_ref[...], b_ref[...])
    else:
        aq, bq = quarters(a_ref[...]), quarters(b_ref[...])
        if impl == 'cubic':
            # Same dependency-spaced eight-product schedule as the study control.
            for quadrant, ai, bi in ((0,0,0),(1,0,1),(2,2,0),(3,2,1),(0,1,2),(1,1,3),(2,3,2),(3,3,3)):
                acc_ref[quadrant] += base._dot(aq[ai], bq[bi])
        else:
            def product(p):
                left, right = operands(aq, bq, p)
                value = base._dot(left, right) if depth == 1 else arithmetic.panel_product(left, right, 1, ORDER)
                if mode == 'products':
                    acc_ref[p-1] += value
                else:
                    for quadrant, sign in TARGETS[p]:
                        if sign == 1:
                            acc_ref[quadrant] += value
                        else:
                            acc_ref[quadrant] -= value
            if spec.early:
                @pl.when(step != nk-1)
                def ordinary():
                    for p in ORDER:
                        product(p)
                @pl.when(step == nk-1)
                def early():
                    for p in (4,5,7,3,1):
                        product(p)
                    finish_pair(acc_ref[0], acc_ref[1], True)
                    for p in (6,2):
                        product(p)
                    finish_pair(acc_ref[2], acc_ref[3], False)
            else:
                for p in ORDER:
                    product(p)
    if not spec.early:
        @pl.when(step == nk-1)
        def finish():
            if impl == 'cubic_full':
                q = quarters(acc_ref[...])
            elif impl == 'current' and mode == 'products':
                q = arithmetic.combine({p: acc_ref[p-1] for p in ORDER}, ORDER)
            else:
                q = [acc_ref[i] for i in range(4)]
            finish_pair(q[0], q[1], True)
            finish_pair(q[2], q[3], False)


def make_projection(arm, shape, spec=Epilogue(), *, interpret=False):
    """Return complete(a, natural_weight, **aux) and prepared-weight entry points.

    prepare_weights must be rerun when tile or model weights change. Model
    adapters call it once per selected layer/profile and time `prepared`.
    Native stays an ordinary XLA expression so whole-block XLA fusion remains
    available; `compiler_options` are caller-owned compilation settings.
    """
    meta = plan(arm, shape, spec)
    meta.update(kernel_version='kernels_fused_v005', dot_precision='DEFAULT',
                norm_reduction_precision='DEFAULT' if spec.kind=='qk_norm_rope' else 'FP32',
                rounding_enforcement='reduce_precision_8_7')
    shape = tuple(meta['shape_mkn']); m, k, n = shape
    padded = tuple(meta['padded_shape_mkn']); mp, kp, npad = padded
    dtype = jnp.dtype(arm['output_dtype'])
    impl = arm['implementation']

    def check_aux(aux):
        expected = {'residual': (m,n), 'bias': (n,), 'cos': (m,spec.head_dim//2),
                    'sin': (m,spec.head_dim//2), 'scale': (spec.head_dim,) if spec.rotary else (n,)}
        if set(aux) != set(spec.auxiliaries):
            raise ValueError('Incorrect epilogue auxiliary operands')
        for name, value in aux.items():
            if value.shape != expected[name] or value.dtype not in (jnp.bfloat16, jnp.float32):
                raise ValueError('Incorrect shape/dtype for '+name)

    if impl == 'native':
        def prepare_weights(b):
            if b.shape != (k,n) or b.dtype != jnp.bfloat16:
                raise ValueError('Expected natural BF16 weights')
            return b
        def prepared(a, b, **aux):
            base._check_inputs(a,b,shape); check_aux(aux)
            return reference(base._dot(a,b), spec, dtype, **aux)
    else:
        bm, bn, bk = arm['tile']
        names = list(spec.auxiliaries)
        options = {} if interpret else {'pipeline_mode': pl.Buffered(buffer_count=arm['buffers'])}
        inputs = [pl.BlockSpec((bm,bk),lambda i,j,s:(i,s),**options),
                  pl.BlockSpec((bk,bn),lambda i,j,s:(s,j),**options)]
        for name in names:
            if name == 'residual':
                inputs.append(pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)))
            elif name in ('cos','sin'):
                inputs.append(pl.BlockSpec((bm,bn//2),lambda i,j,s:(i,0)))
            else:
                inputs.append(pl.BlockSpec((1,bn),lambda i,j,s:(0,0) if spec.rotary else (0,j)))
        if spec.kind == 'qk_norm_rope':
            names += ['segment','segment_t']
            inputs += [pl.BlockSpec((bn//2,128),lambda i,j,s:(0,0)),
                       pl.BlockSpec((128,bn//2),lambda i,j,s:(0,0))]
        out_width = npad//2 if spec.gated else npad
        out_tile = bn//2 if spec.gated else bn
        call = pl.pallas_call(functools.partial(_kernel, impl=impl, depth=arm['depth'],
            mode=arm.get('accumulator','products'), nk=kp//bk, spec=spec, aux_names=tuple(names), width=n),
            grid=(mp//bm,npad//bn,kp//bk), in_specs=inputs,
            out_specs=pl.BlockSpec((bm,out_tile),lambda i,j,s:(i,j)),
            out_shape=jax.ShapeDtypeStruct((mp,out_width),dtype),
            scratch_shapes=[pltpu.VMEM(tuple(meta['scratch_shape']),jnp.float32)],
            compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),
                vmem_limit_bytes=arm['vmem_limit_bytes']), interpret=interpret,
            name=f'fused_v005_{impl}_s{arm["depth"]}_{spec.kind}_{spec.rounding}_{arm["output_dtype"]}')

        def prepare_weights(b):
            if b.shape != (k,n) or b.dtype != jnp.bfloat16:
                raise ValueError('Expected natural BF16 weights')
            return pack_weights(b, shape, padded, bn, spec)

        def prepared(a, b, **aux):
            if a.shape != (m,k) or a.dtype != jnp.bfloat16 or b.shape != (kp,npad) or b.dtype != jnp.bfloat16:
                raise ValueError('Expected natural BF16 activations and weights prepared for this tile')
            check_aux(aux)
            values = [jnp.pad(a,((0,mp-m),(0,kp-k))), b]
            for name in spec.auxiliaries:
                value = aux[name].astype(jnp.float32)
                if name == 'residual':
                    value = jnp.pad(value,((0,mp-m),(0,npad-n)))
                elif name in ('cos','sin'):
                    value = jnp.tile(value,(1,bn//spec.head_dim))
                    value = jnp.pad(value,((0,mp-m),(0,0)))
                elif spec.rotary:
                    first, second = jnp.split(value,2)
                    value = jnp.concatenate((jnp.tile(first,bn//spec.head_dim),jnp.tile(second,bn//spec.head_dim)))[None,:]
                else:
                    value = jnp.pad(value,(0,npad-n))[None,:]
                values.append(value)
            if spec.kind == 'qk_norm_rope':
                columns = jnp.arange(bn//2)
                segment = jnp.zeros((bn//2,128),jnp.float32).at[columns, columns//(spec.head_dim//2)].set(1)
                values += [segment,segment.T]
            result = call(*values)
            if spec.rotary:
                result = unpack_rotary(result,bn,spec.head_dim)
            return result[:m, :n//2 if spec.gated else n]

    def complete(a,b,**aux):
        base._check_inputs(a,b,shape)
        return prepared(a,prepare_weights(b),**aux)
    complete.prepare_weights = prepare_weights
    complete.prepared = prepared
    complete.metadata = meta
    return complete
