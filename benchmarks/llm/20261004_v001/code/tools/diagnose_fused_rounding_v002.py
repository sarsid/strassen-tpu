"""Separate epilogue semantic differences from whole-JIT/backend rounding."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm import model_n9_v001 as qm, model_gemma_v002 as gm, model_composed_v002 as composed
from strassen_mm.model_fused_v002 import build_layer
from strassen_mm.fusion_v001 import Epilogue, reference

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
rng=np.random.default_rng(1002002); rows=[]
def record(name,a,b):
    a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
    row=dict(name=name,max_abs=float(abs(a-b).max()),relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),different=int(np.count_nonzero(a!=b)),elements=a.size)
    rows.append(row);print(json.dumps(row),flush=True)
def rand(shape,scale=1):return jnp.asarray(rng.normal(size=shape)*scale,jnp.bfloat16)
length,hidden,dim=7,512,128
projection=jnp.asarray(rng.normal(size=(length,256)),jnp.float32)
residual=rand(projection.shape);scale=rand((dim,),.1)+1
freq=jnp.arange(length,dtype=jnp.float32)[:,None]/10000.**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim)[None,:]
aux=dict(cos=jnp.cos(freq).astype(jnp.bfloat16),sin=jnp.sin(freq).astype(jnp.bfloat16),scale=scale)
for model in ('qwen','gemma'):
    mod=qm if model=='qwen' else gm
    x=projection.astype(jnp.bfloat16).reshape(length,2,dim)
    expected=mod.rope(mod.rms(x,scale,1e-6),10000.).reshape(length,256)
    spec=Epilogue('qk_norm_rope',norm=model)
    actual=reference(projection,spec,jnp.bfloat16,**aux)
    record(model+'_qk_eager',actual,expected)
    actual=jax.jit(lambda p:reference(p,spec,jnp.bfloat16,**aux))(projection)
    record(model+'_qk_jit',actual,expected)
    w=rand((256,),.1)+1
    expected=(mod.rms(projection.astype(jnp.bfloat16),w,1e-6)+residual).astype(jnp.bfloat16)
    actual=reference(projection,Epilogue('norm_residual_add',norm=model),jnp.bfloat16,residual=residual,scale=w)
    record(model+'_normresidual',actual,expected)
for kind in ('swiglu','geglu'):
    gate,up=jnp.split(projection.astype(jnp.bfloat16),2,-1)
    value=(gm.gelu(gate) if kind=='geglu' else jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16))*up
    actual=reference(projection,Epilogue(kind),jnp.bfloat16)
    record(kind,actual,value)
c=dict(model_type='qwen3',hidden_size=hidden,intermediate_size=384,num_hidden_layers=1,
       num_attention_heads=2,num_key_value_heads=1,head_dim=dim,rms_norm_eps=1e-6,rope_theta=10000.)
w={name:rand(shape,1/np.sqrt(shape[0])) for name,shape in dict(q=(hidden,2*dim),k=(hidden,dim),v=(hidden,dim),o=(2*dim,hidden),gateup=(hidden,768),down=(384,hidden)).items()}
w.update(norm1=rand((hidden,),.1)+1,norm2=rand((hidden,),.1)+1,qnorm=scale,knorm=scale)
x=rand((2,length,hidden))
new=build_layer(c,length,interpret=True)
old=qm.build_layer(c,length,qm.native_policy())
batched=build_layer(c,length,batch_size=2,interpret=True)
unfused=build_layer(c,length,fused_sites=(),interpret=True)
for i in range(2):
    a=new(x[i],w); b=old(x[i],w)
    record('single_new_vs_old_'+str(i),a,b)
    r,n=composed.prefix(c,length)(x[i],w)
    y=composed.suffix(c)(qm.dot(composed.activation(c)(qm.dot(n,w['gateup'])),w['down']),r,w)
    record('single_new_vs_composed_'+str(i),a,y)
    record('single_unfused_vs_new_'+str(i),unfused(x[i],w),a)
record('batch_vs_singles',batched(x,w),jnp.stack([new(v,w) for v in x]))
from strassen_mm.tuner_joint_v001 import arm
policy={site:dict(arm((32,512,512),0,'outputs',2,'bfloat16'),architecture='v6e',implementation='cubic') for site in ('q','k','v','o','gateup','down')}
custom=build_layer(c,length,policy,batch_size=2,interpret=True)
plain=build_layer(c,length,policy,batch_size=2,fused_sites=(),interpret=True)
record('cubic_fused_vs_native',custom(x,w),batched(x,w))
record('cubic_fused_vs_unfused',custom(x,w),plain(x,w))
for key in ('q','k','v','o','gateup','down'):
    k,n=w[key].shape
    sparse=jnp.zeros((k,n),jnp.bfloat16).at[jnp.arange(n)%k,jnp.arange(n)].set(.125)
    w[key]=sparse
record('sparse_cubic_fused_vs_native',custom(x,w),batched(x,w))
record('sparse_cubic_fused_vs_unfused',custom(x,w),plain(x,w))
record('sparse_native_vs_legacy',new(x[0],w),old(x[0],w))
(out/'diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')
