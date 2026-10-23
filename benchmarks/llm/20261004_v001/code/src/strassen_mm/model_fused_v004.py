"""Whole-prefill-layer integration of v6e fused projections.

Consumes the existing Qwen3/Mistral/Gemma3 checkpoint.layer() dictionaries.
Supports fixed unpadded (B,S,H), or (S,H) for B=1, starting at position zero.
No incremental KV-cache update, multimodal input, dropout or quantization.
Packing is explicit setup; `prepared` includes output-layout restoration.
"""
import jax
import jax.numpy as jnp
from . import model_n9_v001 as qm
from . import model_gemma_v002 as gm
from .fusion_v003 import Epilogue
from .fusion_v003 import reference, bf32, store
from .kernels_fused_v004 import make_projection
from .tuner_joint_v001 import arm

SITES = ('q', 'k', 'v', 'o', 'gateup', 'down')


def validate_config(config):
    if config.get('model_type') in ('gemma3', 'gemma3_text'):
        return gm.validate_config(config)
    c = dict(config)
    if c.get('model_type') not in ('qwen3', 'mistral'):
        raise ValueError('Only Qwen3, Mistral and Gemma3 text are supported')
    for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads'):
        gm._positive_int(c.get(key), key)
    c.setdefault('head_dim', c['hidden_size']//c['num_attention_heads'])
    gm._positive_int(c['head_dim'], 'head_dim')
    if c['head_dim'] % 2 or c['num_attention_heads'] % c['num_key_value_heads']:
        raise ValueError('Even head dimensions and integral grouped-query heads required')
    for key in ('rms_norm_eps', 'rope_theta'):
        c[key] = gm._positive_float(c.get(key), key)
    if c.get('hidden_act', 'silu') != 'silu':
        raise ValueError('Qwen/Mistral requires SiLU')
    if c.get('attention_bias', False) or c.get('mlp_bias', False):
        raise ValueError('Biased model projections are not integrated')
    if c.get('attention_dropout', 0) != 0 or c.get('is_encoder_decoder', False):
        raise ValueError('Only dropout-free decoder inference is supported')
    if c.get('quantization_config') or c.get('rope_parameters'):
        raise ValueError('Quantization/alternate RoPE schemas unsupported')
    if c.get('rope_scaling') not in (None, {}, {'rope_type':'default'}, {'type':'default'}):
        raise ValueError('Qwen/Mistral nondefault RoPE scaling unsupported')
    if c.get('partial_rotary_factor', 1.0) != 1 or c.get('use_sliding_window', False):
        raise ValueError('Partial RoPE or Qwen hybrid sliding attention unsupported')
    if c.get('torch_dtype', c.get('dtype', 'bfloat16')) not in (None, 'bfloat16'):
        raise ValueError('BF16 checkpoint matrices required')
    if c.get('sliding_window') is not None:
        gm._positive_int(c['sliding_window'], 'sliding_window')
    return c


def native_policy(output_dtype='bfloat16'):
    return {site: dict(arm(dtype=output_dtype), architecture='v6e') for site in SITES}


def build_layer(config, sequence_length, policy=None, *, batch_size=1, layer_index=0,
                output_dtype='bfloat16', rounding='model', fused_sites=SITES,
                early_sites=(), compiler_options=None, interpret=False, attention_backend='xla'):
    if attention_backend not in ('xla', 'explicit'):
        raise ValueError('Unknown attention backend')
    c = validate_config(config)
    for key, value in (('sequence_length', sequence_length), ('batch_size', batch_size)):
        gm._positive_int(value, key)
    if type(layer_index) is not int or not 0 <= layer_index < c['num_hidden_layers']:
        raise ValueError('Invalid layer index')
    if sequence_length > c.get('max_position_embeddings', sequence_length):
        raise ValueError('Sequence exceeds configured maximum positions')
    if not set(fused_sites) <= set(SITES) or not set(early_sites) <= set(fused_sites):
        raise ValueError('Unknown/unfused site requested')
    chosen = native_policy(output_dtype)
    if policy is not None:
        if not set(policy) <= set(SITES):
            raise ValueError('Unknown projection in policy')
        chosen.update(policy)
    compile_opts = dict(compiler_options or {})
    for a in chosen.values():
        if a['output_dtype'] != output_dtype:
            raise ValueError('One output-dtype comparison per layer policy')
        if a.get('compiler_options') and a['compiler_options'] != compile_opts:
            raise ValueError('Per-projection compiler options cannot silently become whole-layer options')
    gemma = c['model_type'] == 'gemma3_text'
    mod = gm if gemma else qm
    hidden, inner = c['hidden_size'], c['intermediate_size']
    heads, kv, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    bsz, length, m = batch_size, sequence_length, batch_size*sequence_length
    eps = c['rms_norm_eps']
    norm = 'gemma' if gemma else 'qwen'
    def rms(x, weight, eps):
        f=x.astype(jnp.float32)
        value=f*jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+eps)
        if gemma:
            value=value*(1+weight.astype(jnp.float32))
        else:
            value=bf32(value)*bf32(weight)
        return store(value,jnp.bfloat16)
    shapes = dict(q=(m,hidden,heads*dim), k=(m,hidden,kv*dim), v=(m,hidden,kv*dim),
                  o=(m,heads*dim,hidden), gateup=(m,hidden,2*inner), down=(m,inner,hidden))
    kinds = dict(q='qk_norm_rope' if gemma or c['model_type']=='qwen3' else 'rope',
                 v='none', o='norm_residual_add' if gemma else 'residual_add',
                 gateup='geglu' if gemma else 'swiglu',
                 down='norm_residual_add' if gemma else 'residual_add')
    kinds['k'] = kinds['q']
    specs = {site: Epilogue(kinds[site], rounding, norm, dim, eps, site in early_sites) for site in SITES}
    projections = {}
    for site in SITES:
        a = chosen[site]
        spec = specs[site]
        if site not in fused_sites:
            # Unfused control retains the FP32 intermediate before applying
            # the SAME epilogue/rounding contract, then the requested store.
            a, spec = dict(a, output_dtype='float32'), Epilogue()
        projections[site] = make_projection(a, shapes[site], spec, interpret=interpret)
    local = gemma and c['layer_types'][layer_index] == 'sliding_attention'
    window = c.get('sliding_window') if local or (not gemma and c['model_type']=='mistral') else None
    allowed = gm.attention_mask(length, window)
    theta = c['rope_local_base_freq'] if local else c['rope_theta']
    factor = 1.0 if not gemma or local else (c.get('rope_scaling') or {}).get('factor', 1.0)

    def position_tables():
        inverse = 1.0/(theta**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))
        inverse = inverse/factor
        angle = jnp.arange(length,dtype=jnp.float32)[:,None]*inverse[None,:]
        return {'cos':jnp.tile(store(jnp.cos(angle),jnp.bfloat16),(bsz,1)),
                'sin':jnp.tile(store(jnp.sin(angle),jnp.bfloat16),(bsz,1))}

    def prepare_weights(w):
        expected = set(SITES) | {'norm1','norm2'}
        if gemma:
            expected |= {'norm3','norm4','qnorm','knorm'}
        elif c['model_type'] == 'qwen3':
            expected |= {'qnorm','knorm'}
        if set(w) != expected:
            raise ValueError('Layer weight keys differ from the checkpoint contract')
        for name in expected-set(SITES):
            size = dim if name in ('qnorm','knorm') else hidden
            if w[name].shape != (size,) or w[name].dtype not in (jnp.bfloat16,jnp.float32):
                raise ValueError('Incorrect normalization weights: '+name)
        return {name: projections[name].prepare_weights(value) if name in SITES else value for name,value in w.items()}

    def call(x,w):
        squeeze = x.ndim == 2
        if x.shape != ((length,hidden) if squeeze and bsz==1 else (bsz,length,hidden)) or x.dtype != jnp.bfloat16:
            raise ValueError('Expected unpadded BF16 activations for the selected (B,S,H) profile')
        x = x.reshape(m,hidden)
        def project(site, a, **aux):
            p = projections[site]
            y = p.prepared(a,w[site],**aux) if site in fused_sites else reference(p.prepared(a,w[site]),specs[site],jnp.dtype(output_dtype),**aux)
            # Consumers retain the model BF16 contract even in the distinct
            # FP32 projection-output experiment; this cast is timed.
            return store(y,jnp.bfloat16)
        normalized = rms(x,w['norm1'],eps)
        position = position_tables()
        qa = dict(position); ka = dict(position)
        if gemma or c['model_type'] == 'qwen3':
            qa['scale'], ka['scale'] = w['qnorm'], w['knorm']
        q = project('q',normalized,**qa).reshape(bsz,length,heads,dim)
        k = project('k',normalized,**ka).reshape(bsz,length,kv,dim)
        v = project('v',normalized).reshape(bsz,length,kv,dim)
        if attention_backend == 'xla':
            value = jax.nn.dot_product_attention(q,k,v,
                mask=allowed[None,None],
                scale=c['query_pre_attn_scalar']**-0.5 if gemma else dim**-0.5,
                implementation='xla')
            value = store(value,jnp.bfloat16).reshape(m,heads*dim)
        else:
            k,v = jnp.repeat(k,heads//kv,2),jnp.repeat(v,heads//kv,2)
            scores = jnp.einsum('bthd,bshd->bhts',q,k,precision=jax.lax.Precision.DEFAULT,
                                 preferred_element_type=jnp.float32)
            scores = store(scores,jnp.bfloat16)
            scale = c['query_pre_attn_scalar']**-0.5 if gemma else dim**-0.5
            scores = store(scores.astype(jnp.float32)*(scale if gemma else bf32(jnp.asarray(scale,jnp.float32))),jnp.bfloat16)
            if gemma:
                mask = jnp.where(allowed,jnp.asarray(0,jnp.bfloat16),jnp.finfo(jnp.bfloat16).min)
                scores = (scores+mask[None,None]).astype(jnp.bfloat16)
            else:
                scores = jnp.where(allowed[None,None],scores,jnp.finfo(jnp.bfloat16).min)
            prob = store(jax.nn.softmax(scores.astype(jnp.float32),-1),jnp.bfloat16)
            value = jnp.einsum('bhts,bshd->bthd',prob,v,precision=jax.lax.Precision.DEFAULT,
                                preferred_element_type=jnp.float32)
            value = store(value,jnp.bfloat16).reshape(m,heads*dim)
        oa = {'residual':x}
        if gemma:
            oa['scale'] = w['norm2']
        residual = project('o',value,**oa)
        normalized = rms(residual,w['norm3' if gemma else 'norm2'],eps)
        middle = project('gateup',normalized)
        da = {'residual':residual}
        if gemma:
            da['scale'] = w['norm4']
        y = project('down',middle,**da).reshape(bsz,length,hidden)
        return y[0] if squeeze else y

    jit_options = {'compiler_options':compile_opts} if compile_opts else {}
    def complete(x,w):
        return call(x,prepare_weights(w))
    complete = jax.jit(complete,**jit_options)
    complete.prepare_weights = prepare_weights
    complete.prepared = jax.jit(call,**jit_options)
    complete.metadata = dict(version='model_fused_v004', attention_backend=attention_backend,model_type=c['model_type'],batch_size=bsz,
        sequence_length=length,layer_index=layer_index,output_dtype=output_dtype,rounding=rounding,
        fused_sites=list(fused_sites),early_sites=list(early_sites),compiler_options=compile_opts,
        attention_scope='whole unpadded prefill; no KV-cache updates',
        consumer_dtype='bfloat16',rounding_enforcement='reduce_precision_8_7_in_XLA',projections={site:p.metadata for site,p in projections.items()},
        policy=chosen,scope='whole layer, including layout restoration and consumer casts; prepared excludes weight packing')
    return complete
