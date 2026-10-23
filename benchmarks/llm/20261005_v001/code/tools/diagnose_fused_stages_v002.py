"""Locate first differing model operation using independent old/new traces."""
import json, os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm import model_n9_v001 as qm
from strassen_mm.fusion_v002 import reference,Epilogue
from strassen_mm.kernels_v001 import _dot

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
rng=np.random.default_rng(1002003);s,h,d=7,512,128
def rand(shape,scale=1):return jnp.asarray(rng.normal(size=shape)*scale,jnp.bfloat16)
w={name:rand(shape,1/np.sqrt(shape[0])) for name,shape in dict(q=(h,2*d),k=(h,d),v=(h,d),o=(2*d,h),gateup=(h,768),down=(384,h)).items()}
w.update(norm1=rand((h,),.1)+1,norm2=rand((h,),.1)+1,qnorm=rand((d,),.1)+1,knorm=rand((d,),.1)+1)
x=rand((s,h))
def trace(x,w,new):
    result={}
    norm=qm.rms(x,w['norm1'],1e-6); result['inputnorm']=norm
    inv=1.0/(10000.**(jnp.arange(0,d,2,dtype=jnp.float32)/d))
    freq=jnp.arange(s,dtype=jnp.float32)[:,None]*inv[None,:]
    pos=dict(cos=jnp.cos(freq).astype(jnp.bfloat16),sin=jnp.sin(freq).astype(jnp.bfloat16))
    qkv=[]
    for name,heads in [('q',2),('k',1),('v',1)]:
        p=_dot(norm,w[name]);result[name+'_projected']=p.astype(jnp.bfloat16)
        if name!='v':
            old=qm.rope(qm.rms(p.astype(jnp.bfloat16).reshape(s,heads,d),w[name+'norm'],1e-6),10000.)
            newvalue=reference(p,Epilogue('qk_norm_rope'),jnp.bfloat16,**pos,scale=w[name+'norm']).reshape(s,heads,d)
            value=newvalue if new else old
        else:value=p.astype(jnp.bfloat16).reshape(s,heads,d)
        result[name+'_final']=value;qkv.append(value)
    q,k,v=qkv;k=jnp.repeat(k,2,1);v=jnp.repeat(v,2,1)
    if new:
        scores=jnp.einsum('bthd,bshd->bhts',q[None],k[None],preferred_element_type=jnp.float32)[0].astype(jnp.bfloat16)
    else:scores=jnp.einsum('thd,shd->hts',q,k,preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    result['scores_unscaled']=scores
    scores=(scores*(d**-0.5)).astype(jnp.bfloat16);result['scores']=scores
    scores=jnp.where((jnp.arange(s)[:,None]>=jnp.arange(s)[None,:])[None],scores,jnp.finfo(jnp.bfloat16).min)
    prob=jax.nn.softmax(scores.astype(jnp.float32),-1).astype(jnp.bfloat16);result['prob']=prob
    value=jnp.einsum('hts,shd->thd',prob,v,preferred_element_type=jnp.float32).astype(jnp.bfloat16).reshape(s,2*d);result['value']=value
    p=_dot(value,w['o']);result['o_projected']=p.astype(jnp.bfloat16)
    residual=reference(p,Epilogue('residual_add'),jnp.bfloat16,residual=x) if new else (x+p.astype(jnp.bfloat16)).astype(jnp.bfloat16)
    result['residual']=residual
    n=qm.rms(residual,w['norm2'],1e-6);result['mlpnorm']=n
    p=_dot(n,w['gateup']);result['gateup']=p.astype(jnp.bfloat16)
    if new:middle=reference(p,Epilogue('swiglu'),jnp.bfloat16)
    else:
        g,u=jnp.split(p.astype(jnp.bfloat16),2,-1)
        middle=(jax.nn.silu(g.astype(jnp.float32)).astype(jnp.bfloat16)*u).astype(jnp.bfloat16)
    result['middle']=middle
    p=_dot(middle,w['down']);result['down']=p.astype(jnp.bfloat16)
    result['end']=reference(p,Epilogue('residual_add'),jnp.bfloat16,residual=residual) if new else (residual+p.astype(jnp.bfloat16)).astype(jnp.bfloat16)
    return result
rows=[]
for compiled in (False,True):
    fn=jax.jit(trace,static_argnums=2) if compiled else trace
    a,b=fn(x,w,True),fn(x,w,False)
    for key in a:
        av,bv=np.asarray(a[key],dtype=np.float32),np.asarray(b[key],dtype=np.float32)
        row=dict(compiled=compiled,stage=key,max_abs=float(abs(av-bv).max()),different=int(np.count_nonzero(av!=bv)))
        rows.append(row);print(json.dumps(row),flush=True)
(out/'stages.json').write_text(json.dumps(rows,indent=2)+'\n')
