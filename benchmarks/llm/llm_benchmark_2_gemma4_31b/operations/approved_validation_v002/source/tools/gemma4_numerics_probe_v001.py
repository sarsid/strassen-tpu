"""Matched-input diagnostics; no tuning, timing claims, or accuracy-gate changes."""
import gc
import json
import math
import numpy as np
import torch
from transformers.models.gemma4 import modeling_gemma4 as official
import gemma4_parent_model_v002 as model


def host(value):
    if isinstance(value,torch.Tensor):return value.detach().float().numpy().copy()
    return np.asarray(model.jax.device_get(value),dtype=np.float32)


def metrics(reference,candidate):
    r,v=host(reference),host(candidate)
    if r.shape!=v.shape:raise ValueError((r.shape,v.shape))
    d=v.astype(np.float64)-r.astype(np.float64)
    return dict(finite=bool(np.isfinite(v).all()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(r.astype(np.float64).ravel()),1e-30)),
        max_abs=float(np.max(np.abs(d))),different_fraction=float(np.mean(v!=r)))


def capture(block,hidden,position,mask):
    """Capture actual official operations, including their exact input tensors."""
    records={};hooks=[]
    names=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm',
        'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
        'self_attn.q_norm','self_attn.k_norm','self_attn.v_norm',
        'mlp.gate_proj','mlp.up_proj','mlp.act_fn','mlp.down_proj']
    for name in names:
        module=block
        for part in name.split('.'):module=getattr(module,part)
        if module is None:continue
        def hook(mod,args,out,key=name):
            records[key]=dict(input=args[0].detach().clone(),output=out.detach().clone(),
                weight=getattr(mod,'weight',None))
        hooks.append(module.register_forward_hook(hook))
    original=official.eager_attention_forward
    def attention(mod,q,k,v,attention_mask,**kw):
        out,probs=original(mod,q,k,v,attention_mask,**kw)
        records['attention']=dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone(),output=out.detach().clone(),probability=probs.detach().clone())
        return out,probs
    official.eager_attention_forward=attention
    try:
        with torch.no_grad():out=block(hidden,position_embeddings=position,attention_mask=mask)
    finally:
        official.eager_attention_forward=original
        for h in hooks:h.remove()
    records['output']=out.detach().clone();records['input']=hidden.detach().clone()
    records['cos'],records['sin']=position
    return records


class Probe:
    def __init__(self,backends=('cpu','tpu')):
        self.devices={b:model.jax.devices(b)[0] for b in backends};self.compiled={}
        self.jax,self.jnp=model.jax,model.jnp

    def array(self,value,backend):
        # Explicit dtype preserves official BF16 boundaries, including booleans.
        a=host(value) if isinstance(value,torch.Tensor) else value
        dtype=self.jnp.bool_ if np.asarray(a).dtype==np.bool_ else self.jnp.bfloat16
        with self.jax.default_device(self.devices[backend]):return self.jnp.asarray(a,dtype)

    def call(self,name,fn,args,backend):
        values=tuple(self.array(x,backend) for x in args)
        key=(backend,name,tuple((x.shape,str(x.dtype)) for x in values))
        if key not in self.compiled:
            with self.jax.default_device(self.devices[backend]):self.compiled[key]=self.jax.jit(fn).lower(*values).compile()
        return self.compiled[key](*values)

    def layer(self,records,index,emit):
        jax,jnp=self.jax,self.jnp;eps=model.config['rms_norm_eps'];rows=[]
        for name,rec in records.items():
            if not isinstance(rec,dict) or name=='attention':continue
            x=rec['input'];weight=rec['weight']
            if name.endswith('_proj'):
                args=(x,weight.T);fn=model.native.dot
            elif name.endswith('norm') or name.endswith('layernorm'):
                if weight is None:args=(x,);fn=lambda x:model.native.rms(x,None,eps)
                else:args=(x,weight);fn=lambda x,w:model.native.rms(x,w,eps)
            elif name=='mlp.act_fn':
                args=(x,);fn=lambda x:model.native.rounded(jax.nn.gelu(x.astype(jnp.float32),approximate=True))
            else:raise ValueError(name)
            outputs={b:self.call(name,fn,args,b) for b in self.devices}
            row=dict(kind='matched_operation',layer=index,operation=name,
                backends={b:metrics(rec['output'],v) for b,v in outputs.items()})
            if 'cpu' in outputs and 'tpu' in outputs:row['cpu_vs_tpu']=metrics(outputs['cpu'],outputs['tpu'])
            if name.endswith('_proj'):
                row['float64_spot_checks']=self.dot_spot_check(rec,outputs)
            emit(row);rows.append(row)
        a=records['attention'];q,k,v=(a[n].transpose(1,2) for n in ('q','k','v'))
        allowed=model.native.attention_mask(q.shape[1],model.geometry(model.config,index)['sliding_window'])==0
        allowed=np.asarray(model.jax.device_get(allowed),bool)
        cases=[('attention',(q,k,v,allowed),model.attention_bf16_scores,a['output'])]
        for site in ('q','k'):
            cases.append((site+'_rotary',(records['self_attn.'+site+'_norm']['output'],records['cos'],records['sin']),model.native.apply_rotary,q if site=='q' else k))
        cases.append(('gate_product',(records['mlp.act_fn']['output'],records['mlp.up_proj']['output']),lambda x,y:model.native.rounded(x.astype(jnp.float32)*y.astype(jnp.float32)),records['mlp.down_proj']['input']))
        cases.append(('residual1',(records['input'],records['post_attention_layernorm']['output']),lambda x,y:model.native.rounded(x.astype(jnp.float32)+y.astype(jnp.float32)),records['pre_feedforward_layernorm']['input']))
        for name,args,fn,reference in cases:
            outputs={b:self.call(name,fn,args,b) for b in self.devices}
            row=dict(kind='matched_operation',layer=index,operation=name,backends={b:metrics(reference,v) for b,v in outputs.items()})
            if 'cpu' in outputs and 'tpu' in outputs:row['cpu_vs_tpu']=metrics(outputs['cpu'],outputs['tpu'])
            emit(row);rows.append(row)
        # Tables are generated independently, rather than feeding Torch tables.
        for backend,device in self.devices.items():
            with jax.default_device(device):tables=model.native.rotary_tables(model.config,index,jnp.arange(q.shape[1])[None])
            for name,table in zip(('cos','sin'),tables):
                row=dict(kind='matched_operation',layer=index,operation='rotary_table_'+name,backends={backend:metrics(records[name],table)})
                emit(row);rows.append(row)
        return rows

    def dot_spot_check(self,rec,outputs):
        """FP64 dot on up to eight differing entries per backend, no new weights."""
        r=host(rec['output']);x=host(rec['input']);w=host(rec['weight']);result={}
        for backend,out in outputs.items():
            y=host(out);locations=np.argwhere(r!=y)[:8];samples=[]
            for loc in locations:
                pos=tuple(int(v) for v in loc);value=float(np.dot(x[pos[:-1]].astype(np.float64),w[pos[-1]].astype(np.float64)))
                rounded=float(torch.tensor(value,dtype=torch.float64).to(torch.bfloat16).float())
                samples.append(dict(index=pos,fp64=value,nearest_bf16=rounded,torch=float(r[pos]),candidate=float(y[pos]),torch_matches_nearest=bool(r[pos]==rounded),candidate_matches_nearest=bool(y[pos]==rounded)))
            result[backend]=samples
        return result


def official_layer(raw,index,cfg):
    def tensor(x):
        if x.dtype==np.float32:return torch.from_numpy(x.copy())
        return torch.from_numpy(x.view(np.uint16).copy()).view(torch.bfloat16)
    with torch.device('meta'):block=official.Gemma4TextDecoderLayer(cfg,index)
    state={}
    for site in ('q','k','v','o'):
        if site in raw:state['self_attn.'+site+'_proj.weight']=tensor(raw[site].T)
    gate,up=np.split(raw['gateup'],2,axis=1)
    state.update({'mlp.gate_proj.weight':tensor(gate.T),'mlp.up_proj.weight':tensor(up.T),'mlp.down_proj.weight':tensor(raw['down'].T),'self_attn.q_norm.weight':tensor(raw['qnorm']),'self_attn.k_norm.weight':tensor(raw['knorm']),'layer_scalar':tensor(raw['layer_scalar'])})
    for key,name in [('norm1','input_layernorm'),('norm2','post_attention_layernorm'),('norm3','pre_feedforward_layernorm'),('norm4','post_feedforward_layernorm')]:state[name+'.weight']=tensor(raw[key])
    block.load_state_dict(state,assign=True,strict=True);block.eval();return block
