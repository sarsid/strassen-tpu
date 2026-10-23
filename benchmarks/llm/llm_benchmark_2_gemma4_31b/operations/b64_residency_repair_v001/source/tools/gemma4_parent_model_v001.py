"""Gemma4 dense text semantics with immutable parent Cubic and S1 kernels.

Prepared weights are common to every arm. Q/K and GeGLU fuse only for S1;
Gemma row-norm/residual fusion is a separately recorded per-site tuning choice.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'third_party/strassen_tpu_public_95be1fb_v001'
MODEL='google/gemma-4-31B'
REVISION='5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89'
BASE=f'https://huggingface.co/{MODEL}/resolve/{REVISION}'
PREFIX='model.language_model'
ARMS=('native','native_tuned','cubic_tuned','strassen_tuned')


def setup():
    global jax,jnp,np,sp,cubic,native,geometry,config
    for path in (ROOT/'src',PARENT,PARENT/'experiments/qwen3'):sys.path.insert(0,str(path))
    os.environ.setdefault('STRASSEN_MANAGE_CEILING','0')
    os.environ.setdefault('LIBTPU_INIT_ARGS','--xla_tpu_use_enhanced_launch_barrier=true')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import strassen_pallas as sp
    import benchmark_cubic_control as cubic
    from strassen_mm import model_gemma4_v001 as native
    from strassen_mm.gemma4_contract_v001 import layer_geometry as geometry,validate_config
    config=validate_config(json.loads((ROOT/'plans/parent_llm_models_20261013_v001/metadata/gemma4_31b.config.json').read_text()))
    return config


class Checkpoint:
    """Public immutable HTTP ranges with content-verified private byte cache."""
    def __init__(self,cache,emit):
        import requests
        from parent_stage_v001 import install_transport
        transport=types.SimpleNamespace(BASE=BASE,ShardedSafetensors=type('Transport',(),{}))
        install_transport(transport,Path(cache),emit)
        self.reader=transport.ShardedSafetensors();self.reader.session=requests.Session()
        self.emit=emit;self.headers={}
        payload=self.reader._get(BASE+'/model.safetensors.index.json').content
        if hashlib.sha256(payload).hexdigest()!='d4aff3b976d69c123a29d1c085d7ba4de1ac3f4ca1726a7f81e1b11462a64ea2':
            raise ValueError('Pinned Gemma checkpoint index mismatch')
        self.index=json.loads(payload)['weight_map']

    def tensor(self,name):
        import ml_dtypes
        if name=='lm_head.weight':name=PREFIX+'.embed_tokens.weight'
        elif name.startswith('model.'):name=PREFIX+name[len('model'):]
        shard=self.index[name]
        if shard not in self.headers:
            length=int.from_bytes(self.reader._get(BASE+'/'+shard,'bytes=0-7').content,'little')
            if not 0<length<8*1024**2:raise ValueError('Invalid safetensors header length')
            header=self.reader._get(BASE+'/'+shard,f'bytes=8-{7+length}').json()
            self.headers[shard]=(header,8+length)
        header,offset=self.headers[shard];info=header[name]
        dtype={'BF16':ml_dtypes.bfloat16,'F32':np.float32}[info['dtype']]
        first,last=info['data_offsets'];size=last-first
        if size!=math.prod(info['shape'])*np.dtype(dtype).itemsize:raise ValueError('Tensor byte count mismatch')
        result=np.empty(math.prod(info['shape']),dtype=dtype);target=result.view(np.uint8)
        digest=hashlib.sha256()
        # The tied 2.6 GiB embedding exceeds the parent's 2 GiB request ceiling.
        # Chunk only transport; preserve exact checkpoint bytes and tensor layout.
        for begin in range(first,last,128*1024**2):
            end=min(last,begin+128*1024**2)
            raw=self.reader._get(BASE+'/'+shard,f'bytes={offset+begin}-{offset+end-1}').content
            target[begin-first:end-first]=np.frombuffer(raw,dtype=np.uint8);digest.update(raw)
        self.emit(dict(kind='tensor',name=name,shape=info['shape'],dtype=info['dtype'],bytes=size,sha256=digest.hexdigest()))
        return result.reshape(info['shape'])


def raw_layer(checkpoint,index):
    p=f'model.layers.{index}.';g=geometry(config,index)
    w={site:checkpoint.tensor(p+'self_attn.'+site+'_proj.weight').T.copy() for site in ('q','k','o')}
    if not g['shared_raw_kv']:w['v']=checkpoint.tensor(p+'self_attn.v_proj.weight').T.copy()
    w['gateup']=np.concatenate([checkpoint.tensor(p+'mlp.'+s+'_proj.weight') for s in ('gate','up')]).T.copy()
    w['down']=checkpoint.tensor(p+'mlp.down_proj.weight').T.copy()
    for key,name in [('norm1','input_layernorm'),('norm2','post_attention_layernorm'),('norm3','pre_feedforward_layernorm'),('norm4','post_feedforward_layernorm')]:
        w[key]=checkpoint.tensor(p+name+'.weight')
    w['qnorm']=checkpoint.tensor(p+'self_attn.q_norm.weight');w['knorm']=checkpoint.tensor(p+'self_attn.k_norm.weight')
    w['layer_scalar']=checkpoint.tensor(p+'layer_scalar')
    return w


def default_profile(batch,sequence):
    bm=min(1024,batch*sequence)
    return {'native_options':{},'strassen':{'q':{'tile':[bm,1024,768],'fused':True},'k':{'tile':[bm,1024,768],'fused':True},'gateup':{'tile':[bm,1024,768],'fused':True},'o':{'tile':[bm,768,1024],'fused':False},'down':{'tile':[bm,768,768],'fused':False}},'cubic':{'gateup':{'tile':[bm,1024,768]},'o':{'tile':[bm,768,1024]},'down':{'tile':[bm,768,768]}}}


def prepare_weights(raw,profile,index,batch,sequence):
    g=geometry(config,index);dim=g['head_dim']
    p={k:jnp.asarray(v) for k,v in raw.items()}
    cos,sin=native.rotary_tables(config,index,jnp.arange(sequence)[None])
    p.update(cos=cos,sin=sin)
    fullcos=jnp.tile(cos[0],(batch,1));fullsin=jnp.tile(sin[0],(batch,1))
    for site in ('q','k'):
        if site=='k' and g['shared_raw_kv']:continue
        bn=profile['strassen'][site]['tile'][1]
        p[site+'_laid']=sp.rope_weight_layout(p[site],bn,dim)
        p[site+'_scale']=sp.rope_scale_layout(p[site+'norm'],bn,dim)
        p[site+'_cos'],p[site+'_sin']=sp.rope_layout_tables(fullcos,fullsin,bn//2,dim)
    p['gateup_laid']=sp.swiglu_weight_layout(p['gateup'],profile['strassen']['gateup']['tile'][1])
    jax.block_until_ready(p)
    return p


def load_layer(checkpoint,index,profile,batch,sequence):
    return prepare_weights(raw_layer(checkpoint,index),profile,index,batch,sequence)


def restore_rope(flat,heads,bn,dim,batch,sequence):
    m=flat.shape[0];half=bn//2;blocks=flat.shape[1]//bn
    halves=flat.reshape(m,blocks,2,half)
    first=halves[:,:,0,:].reshape(m,heads,dim//2)
    second=halves[:,:,1,:].reshape(m,heads,dim//2)
    return jnp.concatenate((first,second),axis=-1).reshape(batch,sequence,heads,dim)


def projection(site,arm,profile,index,batch,sequence,*,interpret=False):
    g=geometry(config,index);eps=config['rms_norm_eps'];dim=g['head_dim']
    def call(x,p,residual=None):
        if site in ('q','k'):
            heads=g['query_heads'] if site=='q' else g['kv_heads']
            # Global K supplies both raw K and V: normalize/rotate after sharing.
            if site=='k' and g['shared_raw_kv']:return native.dot(x,p['k'])
            if arm=='strassen_tuned':
                bm,bn,bk=profile['strassen'][site]['tile']
                y=sp.strassen_matmul(x,p[site+'_laid'],bm=bm,bn=bn,bk=bk,interleave_products=True,
                    epilogue='qk_norm_rope',rope_cos=p[site+'_cos'],rope_sin=p[site+'_sin'],
                    rope_scale=p[site+'_scale'],rope_head_dim=dim,rope_eps=eps,
                    vmem_limit_bytes=120*1024**2,interpret=interpret)
                return restore_rope(y,heads,bn,dim,batch,sequence).reshape(batch*sequence,-1)
            y=native.dot(x,p[site]).reshape(batch,sequence,heads,dim)
            return native.apply_rotary(native.rms(y,p[site+'norm'],eps),p['cos'],p['sin']).reshape(batch*sequence,-1)
        if site=='v':return native.dot(x,p['v'])
        entry=profile['strassen' if arm=='strassen_tuned' else 'cubic'][site] if arm in ('strassen_tuned','cubic_tuned') else None
        if entry is None:y=native.dot(x,p[site])
        else:
            bm,bn,bk=entry['tile']
            if arm=='strassen_tuned':
                if site=='gateup':
                    return sp.strassen_matmul(x,p['gateup_laid'],bm=bm,bn=bn,bk=bk,interleave_products=True,
                        epilogue='geglu',product_aware_swiglu=True,vmem_limit_bytes=120*1024**2,interpret=interpret)
                if entry.get('fused'):
                    return sp.strassen_matmul(x,p[site],bm=bm,bn=bn,bk=bk,interleave_products=True,
                        epilogue='norm_residual_add',residual=residual,norm_scale=p['norm2' if site=='o' else 'norm4'],
                        norm_eps=eps,vmem_limit_bytes=120*1024**2,interpret=interpret)
                y=sp.strassen_matmul(x,p[site],bm=bm,bn=bn,bk=bk,interleave_products=True,
                    vmem_limit_bytes=120*1024**2,interpret=interpret)
            else:y=cubic.cubic_matmul(x,p[site],variant='blocked',bm=bm,bn=bn,bk=bk,vmem_limit_bytes=124*1024**2,interpret=interpret)
        if site=='gateup':
            gate,up=jnp.split(y,2,axis=-1)
            return native.rounded(native.rounded(jax.nn.gelu(gate.astype(jnp.float32),approximate=True)).astype(jnp.float32)*up.astype(jnp.float32))
        y=native.rms(y,p['norm2' if site=='o' else 'norm4'],eps)
        return native.rounded(residual.astype(jnp.float32)+y.astype(jnp.float32))
    return call


def components(arm,profile,index,batch,sequence,*,interpret=False):
    g=geometry(config,index);h=config['hidden_size'];dim=g['head_dim'];eps=config['rms_norm_eps'];m=batch*sequence
    project={s:projection(s,arm,profile,index,batch,sequence,interpret=interpret) for s in g['projection_shapes_mkn']}
    allowed=native.attention_mask(sequence,g['sliding_window'])==0
    def qkv(x,p):
        normalized=native.rms(x,p['norm1'],eps)
        q=project['q'](normalized,p).reshape(batch,sequence,g['query_heads'],dim)
        if g['shared_raw_kv']:
            raw=project['k'](normalized,p).reshape(batch,sequence,g['kv_heads'],dim)
            k=native.apply_rotary(native.rms(raw,p['knorm'],eps),p['cos'],p['sin']);v=native.rms(raw,None,eps)
        else:
            k=project['k'](normalized,p).reshape(batch,sequence,g['kv_heads'],dim)
            v=native.rms(project['v'](normalized,p).reshape(batch,sequence,g['kv_heads'],dim),None,eps)
        return normalized,q,k,v
    def attention(q,k,v):
        return native.rounded(jax.nn.dot_product_attention(q,k,v,mask=allowed[None,None],scale=1.0,implementation='xla')).reshape(m,-1)
    def postattention(x,attended,p):
        residual=project['o'](attended,p,x)
        return residual,native.rms(residual,p['norm3'],eps)
    def gate(value,p):return project['gateup'](value,p)
    def finish(value,residual,p):
        y=project['down'](value,p,residual)
        return native.rounded(y.astype(jnp.float32)*p['layer_scalar'].astype(jnp.float32))
    return qkv,attention,postattention,gate,finish


def build_layer(arm,profile,index,batch,sequence,*,interpret=False):
    qkv,attention,post,gate,finish=components(arm,profile,index,batch,sequence,interpret=interpret)
    def call(x,p):
        _,q,k,v=qkv(x,p);attended=attention(q,k,v)
        residual,normalized=post(x,attended,p)
        return finish(gate(normalized,p),residual,p)
    return call


def arm_options(arm,profile):
    return {} if arm=='native' else profile['native_options'] if arm=='native_tuned' else {'xla_tpu_scoped_vmem_limit_kib':48*1024}


def initial_hidden(checkpoint,ids):
    embedding=checkpoint.tensor('model.embed_tokens.weight')
    out=native.scaled_embeddings(jnp.asarray(embedding[ids].reshape(-1,config['hidden_size']).copy()),config['hidden_size'])
    jax.block_until_ready(out);return out
