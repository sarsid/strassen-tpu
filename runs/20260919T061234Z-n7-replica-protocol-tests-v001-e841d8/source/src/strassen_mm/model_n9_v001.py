"""Streamed Qwen3/Mistral inference from immutable official safetensors.

Architecture follows official Transformers 4.56.2 modeling_qwen3/modeling_mistral.
BF16 tensor boundaries follow its eager inference implementation. This module
is independently qualified against that implementation before any quality claim.
Only MLP gate/up and down projections are replaceable; attention is always XLA.
Historical project streamed inference informed the memory strategy, not quality
evidence: new checkpoints, corpus, reference and results are recorded separately.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from .kernels_n8_v001 import make_epilogue, activation


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


class Checkpoint:
    """Exact little-endian BF16 tensors, no conversion via float16 or PyTorch."""
    def __init__(self,manifest_path):
        self.manifest_path=Path(manifest_path)
        self.manifest=json.loads(self.manifest_path.read_text())
        self.root=Path(self.manifest['cache_dir']).resolve()
        self.config=self.manifest['config']
        self.tables={};self.locations={};self.verified=[]
        if self.config['model_type'] not in ('qwen3','mistral'):
            raise ValueError('Only Qwen3 and Mistral architectures are implemented')
        if self.config.get('rope_scaling') or self.config.get('sliding_window'):
            raise ValueError('This fixed study implements unscaled full causal attention only')
        if self.config.get('attention_bias',False) or self.config.get('hidden_act')!='silu':
            raise ValueError('Unsupported projection bias or activation')
        revision=self.manifest['revision']
        if len(revision)!=40 or any(x not in '0123456789abcdef' for x in revision):
            raise ValueError('Model revision must be immutable 40-hex commit')
        for row in self.manifest['files']:
            path=(self.root/row['path']).resolve()
            if not path.is_relative_to(self.root): raise ValueError('Checkpoint path escapes cache')
            if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
                raise ValueError('Checkpoint file size/hash mismatch: '+row['path'])
            self.verified.append(dict(row))
            if path.suffix!='.safetensors': continue
            with path.open('rb') as handle:
                raw=handle.read(8)
                if len(raw)!=8: raise ValueError('Truncated safetensors length')
                size=struct.unpack('<Q',raw)[0]
                if size>64*1024**2: raise ValueError('Unreasonable safetensors header size')
                header=json.loads(handle.read(size))
            for name,info in header.items():
                if name=='__metadata__': continue
                if name in self.locations: raise ValueError('Duplicate checkpoint tensor')
                lo,hi=info['data_offsets'];shape=info['shape']
                widths={'BF16':2,'F32':4}
                if info['dtype'] not in widths: raise ValueError('Unsupported tensor dtype '+info['dtype'])
                if lo<0 or hi-lo!=int(np.prod(shape))*widths[info['dtype']] or 8+size+hi>path.stat().st_size:
                    raise ValueError('Invalid tensor byte bounds')
                self.locations[name]=(path,8+size+lo,shape,info['dtype'])
        if not self.locations: raise ValueError('Checkpoint contains no safetensors tensors')

    def tensor(self,name):
        path,offset,shape,dtype=self.locations[name]
        if dtype=='BF16':
            return np.memmap(path,mode='r',offset=offset,dtype='<u2',shape=tuple(shape)).view(ml_dtypes.bfloat16)
        return np.memmap(path,mode='r',offset=offset,dtype='<f4',shape=tuple(shape))

    def layer(self,index):
        prefix=f'model.layers.{index}.'
        weights={}
        for short,long in (('q','self_attn.q_proj.weight'),('k','self_attn.k_proj.weight'),
            ('v','self_attn.v_proj.weight'),('o','self_attn.o_proj.weight'),
            ('down','mlp.down_proj.weight')):
            weights[short]=np.ascontiguousarray(self.tensor(prefix+long).T)
        weights['gateup']=np.ascontiguousarray(np.concatenate((
            self.tensor(prefix+'mlp.gate_proj.weight'),self.tensor(prefix+'mlp.up_proj.weight')),axis=0).T)
        weights['norm1']=np.array(self.tensor(prefix+'input_layernorm.weight'),copy=True)
        weights['norm2']=np.array(self.tensor(prefix+'post_attention_layernorm.weight'),copy=True)
        if self.config['model_type']=='qwen3':
            weights['qnorm']=np.array(self.tensor(prefix+'self_attn.q_norm.weight'),copy=True)
            weights['knorm']=np.array(self.tensor(prefix+'self_attn.k_norm.weight'),copy=True)
        return weights

    def embeddings(self,tokens):
        return np.array(self.tensor('model.embed_tokens.weight')[np.asarray(tokens)],copy=True)

    def head(self):
        name='lm_head.weight'
        if name not in self.locations:
            if not self.config.get('tie_word_embeddings'): raise ValueError('Missing untied LM head')
            name='model.embed_tokens.weight'
        return np.ascontiguousarray(self.tensor(name).T)


def rounded(x): return x.astype(jnp.bfloat16)


def dot(a,b):
    return jnp.matmul(a,b,precision=jax.lax.Precision.DEFAULT,
                      preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def rms(x,weight,eps):
    value=x.astype(jnp.float32)
    normalized=(value*jax.lax.rsqrt(jnp.mean(value*value,axis=-1,keepdims=True)+eps)).astype(jnp.bfloat16)
    return (normalized*weight.astype(jnp.bfloat16)).astype(jnp.bfloat16)


def rope(x,theta):
    length=x.shape[0];dim=x.shape[-1]
    inverse=1.0/(float(theta)**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))
    freq=jnp.arange(length,dtype=jnp.float32)[:,None]*inverse[None,:]
    angle=jnp.concatenate((freq,freq),axis=-1)[:,None,:]
    cosine=jnp.cos(angle).astype(jnp.bfloat16);sine=jnp.sin(angle).astype(jnp.bfloat16)
    rotated=jnp.concatenate((-x[...,dim//2:],x[...,:dim//2]),axis=-1)
    left=(x*cosine).astype(jnp.bfloat16);right=(rotated*sine).astype(jnp.bfloat16)
    return (left+right).astype(jnp.bfloat16)


def build_layer(config,sequence_length,policy):
    hidden=config['hidden_size'];intermediate=config['intermediate_size']
    heads=config['num_attention_heads'];kvheads=config['num_key_value_heads']
    dim=config.get('head_dim',hidden//heads);eps=config['rms_norm_eps']
    if heads%kvheads: raise ValueError('Query head count must be divisible by KV heads')
    mmfns={}
    for name,shape,kind in (('gateup',(sequence_length,hidden,intermediate*2),'swiglu'),
                            ('down',(sequence_length,intermediate,hidden),'residual')):
        choice=policy[name]
        mmfns[name]=make_epilogue(choice['algorithm'],shape,tuple(choice.get('tile') or (1024,1024,512)),
             kind=kind,fused=choice.get('fused',False),early=choice.get('early',False),
             packed=choice.get('packed',False),variant=choice.get('variant','plain'))

    def layer(x,weights):
        normalized=rms(x,weights['norm1'],eps)
        q=dot(normalized,weights['q']).reshape(sequence_length,heads,dim)
        k=dot(normalized,weights['k']).reshape(sequence_length,kvheads,dim)
        v=dot(normalized,weights['v']).reshape(sequence_length,kvheads,dim)
        if config['model_type']=='qwen3':
            q=rms(q,weights['qnorm'],eps);k=rms(k,weights['knorm'],eps)
        q=rope(q,config['rope_theta']);k=rope(k,config['rope_theta'])
        k=jnp.repeat(k,heads//kvheads,axis=1);v=jnp.repeat(v,heads//kvheads,axis=1)
        scores=jnp.einsum('thd,shd->hts',q,k,precision=jax.lax.Precision.DEFAULT,
                          preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        scores=(scores*(dim**-0.5)).astype(jnp.bfloat16)
        mask=jnp.arange(sequence_length)[:,None]>=jnp.arange(sequence_length)[None,:]
        scores=jnp.where(mask[None,:,:],scores,jnp.finfo(jnp.bfloat16).min)
        probability=jax.nn.softmax(scores.astype(jnp.float32),axis=-1).astype(jnp.bfloat16)
        value=jnp.einsum('hts,shd->thd',probability,v,precision=jax.lax.Precision.DEFAULT,
                         preferred_element_type=jnp.float32).astype(jnp.bfloat16).reshape(sequence_length,heads*dim)
        residual=(x+dot(value,weights['o'])).astype(jnp.bfloat16)
        normalized=rms(residual,weights['norm2'],eps)
        product=mmfns['gateup'](normalized,weights['gateup'])
        return mmfns['down'](product,weights['down'],residual)
    return jax.jit(layer)


def native_policy():
    return {key:{'algorithm':'native','variant':'plain','tile':None,'fused':False}
            for key in ('gateup','down')}


@jax.jit
def head_logits(hidden,norm,weight,eps):
    return dot(rms(hidden,norm,eps),weight)


@jax.jit
def compare_logits(reference,candidate,targets):
    """Aggregate exact same scored positions; returns sums, no batch averaging."""
    reference=reference.astype(jnp.float32);candidate=candidate.astype(jnp.float32)
    logp=jax.nn.log_softmax(reference,axis=-1);logq=jax.nn.log_softmax(candidate,axis=-1)
    index=jnp.arange(reference.shape[0])
    return {
      'positions':jnp.asarray(reference.shape[0],jnp.int32),
      'reference_nll_sum':-jnp.sum(logp[index,targets]),
      'candidate_nll_sum':-jnp.sum(logq[index,targets]),
      'kl_sum':jnp.sum(jnp.sum(jnp.exp(logp)*(logp-logq),axis=-1)),
      'top1_equal':jnp.sum(jnp.argmax(reference,axis=-1)==jnp.argmax(candidate,axis=-1)),
      'squared_error_sum':jnp.sum((reference-candidate)**2),
      'reference_squared_sum':jnp.sum(reference**2),
      'max_abs_error':jnp.max(jnp.abs(reference-candidate)),
      'all_finite':jnp.all(jnp.isfinite(reference)) & jnp.all(jnp.isfinite(candidate)),
    }
