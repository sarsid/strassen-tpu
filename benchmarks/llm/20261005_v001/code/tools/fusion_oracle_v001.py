"""Independent host NumPy oracle with materialized BF16 rounding boundaries."""
import numpy as np
import ml_dtypes


def reference(projected,spec,dtype,**aux):
    def bf(x):return np.asarray(x,dtype=ml_dtypes.bfloat16).astype(np.float32)
    x=np.asarray(projected,dtype=np.float32)
    aux={k:np.asarray(v,dtype=np.float32) for k,v in aux.items()}
    if spec.rounding=='model' and spec.kind!='none':x=bf(x)
    def norm(x,scale):
        inv=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+np.float32(spec.eps))
        y=x*inv
        if spec.norm=='gemma':y=y*(1+scale)
        elif spec.rounding=='model':y=bf(y)*bf(scale)
        else:y=y*scale
        return bf(y) if spec.rounding=='model' else y
    if spec.gated:
        first,second=np.split(x,2,-1)
        if spec.kind=='swiglu':value=first/(1+np.exp(-first))
        else:value=.5*first*(1+np.tanh(np.sqrt(2/np.pi)*(first+.044715*first**3)))
        if spec.rounding=='model':value=bf(value)
        x=value*second
    elif spec.kind=='residual_add':x=x+aux['residual']
    elif spec.kind=='bias_add':x=x+aux['bias']
    elif spec.kind=='norm_residual_add':x=norm(x,aux['scale'])+aux['residual']
    elif spec.rotary:
        shape=x.shape;x=x.reshape(shape[0],-1,spec.head_dim)
        if spec.kind=='qk_norm_rope':x=norm(x,aux['scale'])
        first,second=np.split(x,2,-1)
        cos,sin=aux['cos'][:,None],aux['sin'][:,None]
        if spec.rounding=='model':
            cos,sin=bf(cos),bf(sin)
            left,right=bf(first*cos)-bf(second*sin),bf(second*cos)+bf(first*sin)
        else:left,right=first*cos-second*sin,second*cos+first*sin
        x=np.concatenate((left,right),-1).reshape(shape)
    return x.astype(ml_dtypes.bfloat16 if str(dtype)=='bfloat16' else np.float32)
