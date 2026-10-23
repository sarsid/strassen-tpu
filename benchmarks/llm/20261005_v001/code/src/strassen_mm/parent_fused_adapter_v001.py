"""Our v6e projections inside the frozen parent's Qwen model computation.

The parent module is supplied explicitly. This adapter does not replace its
attention/RMSNorm/RoPE or impose the strict model_fused_v003 rounding contract.
Native sites retain the parent's BF16 projection boundaries. Custom sites use
FP32 accumulator epilogues, as in the parent's fused treatment.
"""
from dataclasses import asdict
import jax
import jax.numpy as jnp
from .fusion_v001 import Epilogue
from .kernels_fused_v003 import make_projection

SITES=('q','k','v','o','gateup','down')
FIELDS=dict(q='query',k='key',v='value',o='attention_output',gateup='gate_up',down='mlp_down')


def shapes(layer):
    m,h,i,d=layer.TOKENS,layer.MODEL_DIM,layer.INTERMEDIATE_DIM,layer.HEAD_DIM
    return dict(q=(m,h,layer.HEADS*d),k=(m,h,layer.KV_HEADS*d),v=(m,h,layer.KV_HEADS*d),
                o=(m,layer.HEADS*d,h),gateup=(m,h,2*i),down=(m,i,h))


def specification(site,early=False):
    return Epilogue(dict(q='qk_norm_rope',k='qk_norm_rope',v='none',o='residual_add',
                         gateup='swiglu',down='residual_add')[site],rounding='accumulator',early=early)


def build(layer,policy,*,interpret=False):
    if set(policy)!=set(SITES):raise ValueError('A frozen arm is required for every projection')
    specs={site:specification(site,policy[site].get('early',False)) for site in SITES}
    projections={site:make_projection(policy[site],shapes(layer)[site],specs[site],interpret=interpret)
                 for site in SITES if policy[site]['implementation']!='native'}

    def prepare(params,cache=None):
        cache={} if cache is None else cache
        packed={name:getattr(params,name) for name in ('attention_norm','query_norm','key_norm','mlp_norm','rope_cos','rope_sin')}
        for site in SITES:
            weight=getattr(params,FIELDS[site])
            if site in projections:
                # The cache is strictly per layer. Identical layouts share HBM;
                # no packing result is reused across different model weights.
                meta=projections[site].metadata
                key=(site,tuple(meta['padded_shape_mkn']),tuple(meta['tile_bm_bn_bk']),specs[site].kind)
                if key not in cache:cache[key]=projections[site].prepare_weights(weight)
                packed[site]=cache[key]
            else:packed[site]=weight
        return packed

    def call(x,w):
        b,s,h,d=layer.BATCH,layer.SEQUENCE,layer.HEADS,layer.HEAD_DIM
        def project(site,a,**aux):
            if site in projections:
                return projections[site].prepared(a,w[site],**aux).astype(jnp.bfloat16)
            return layer.native_projection(a,w[site])
        normalized=layer.rms_norm(x,w['attention_norm'])
        rotary=dict(cos=jnp.tile(w['rope_cos'][:,:d//2],(b,1)),sin=jnp.tile(w['rope_sin'][:,:d//2],(b,1)))
        values={}
        for site,heads,norm in [('q',h,'query_norm'),('k',layer.KV_HEADS,'key_norm')]:
            if site in projections:
                values[site]=project(site,normalized,**rotary,scale=w[norm]).reshape(b,s,heads,d)
            else:
                value=project(site,normalized).reshape(b,s,heads,d)
                values[site]=layer.apply_rope(layer.rms_norm(value,w[norm]),w['rope_cos'],w['rope_sin'])
        value=project('v',normalized).reshape(b,s,layer.KV_HEADS,d)
        attended=jax.nn.dot_product_attention(values['q'],values['k'],value,is_causal=True,implementation='xla').reshape(layer.TOKENS,h*d).astype(jnp.bfloat16)
        if 'o' in projections:
            residual=project('o',attended,residual=x)
        else:
            residual=(x.astype(jnp.float32)+project('o',attended).astype(jnp.float32)).astype(jnp.bfloat16)
        normalized=layer.rms_norm(residual,w['mlp_norm'])
        if 'gateup' in projections:
            middle=project('gateup',normalized)
        else:
            gate,up=jnp.split(project('gateup',normalized),2,-1)
            middle=(jax.nn.silu(gate.astype(jnp.float32))*up.astype(jnp.float32)).astype(jnp.bfloat16)
        if 'down' in projections:
            return project('down',middle,residual=residual)
        return (residual.astype(jnp.float32)+project('down',middle).astype(jnp.float32)).astype(jnp.bfloat16)

    call.prepare_weights=prepare
    call.metadata=dict(version='parent_fused_adapter_v001',policy=policy,
                       projections={k:v.metadata for k,v in projections.items()},
                       native_semantics='exact parent BF16 projection and epilogue boundaries',
                       custom_semantics='FP32 accumulator epilogues; output store then BF16 consumer',
                       additional_fusion='O residual and cubic Q/K epilogues receive equal opportunity',
                       attention='parent jax.nn.dot_product_attention implementation=xla')
    return call


def tuning_operands(layer,params,x):
    """Real layer-0 activations from the parent Native computation."""
    b,s,d=layer.BATCH,layer.SEQUENCE,layer.HEAD_DIM
    normalized=layer.rms_norm(x,params.attention_norm)
    q=layer.native_projection(normalized,params.query).reshape(b,s,layer.HEADS,d)
    k=layer.native_projection(normalized,params.key).reshape(b,s,layer.KV_HEADS,d)
    v=layer.native_projection(normalized,params.value).reshape(b,s,layer.KV_HEADS,d)
    q=layer.apply_rope(layer.rms_norm(q,params.query_norm),params.rope_cos,params.rope_sin)
    k=layer.apply_rope(layer.rms_norm(k,params.key_norm),params.rope_cos,params.rope_sin)
    attended=jax.nn.dot_product_attention(q,k,v,is_causal=True,implementation='xla').reshape(layer.TOKENS,layer.HEADS*d).astype(jnp.bfloat16)
    residual=(x.astype(jnp.float32)+layer.native_projection(attended,params.attention_output).astype(jnp.float32)).astype(jnp.bfloat16)
    mlp=layer.rms_norm(residual,params.mlp_norm)
    gate,up=jnp.split(layer.native_projection(mlp,params.gate_up),2,-1)
    middle=(jax.nn.silu(gate.astype(jnp.float32))*up.astype(jnp.float32)).astype(jnp.bfloat16)
    rotary=dict(cos=jnp.tile(params.rope_cos[:,:d//2],(b,1)),sin=jnp.tile(params.rope_sin[:,:d//2],(b,1)))
    return dict(q=(normalized,dict(rotary,scale=params.query_norm)),k=(normalized,dict(rotary,scale=params.key_norm)),
                v=(normalized,{}),o=(attended,dict(residual=x)),gateup=(mlp,{}),down=(middle,dict(residual=residual)))
