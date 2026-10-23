"""Official Transformers tiny-model semantic gate on all shared adapters.

Synthetic weights here only qualify model semantics, never real-model accuracy.
The independent oracle is the actual pinned Transformers forward implementation.
"""
import argparse, importlib.metadata as md, json, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch, transformers, jax
    import jax.numpy as jnp
    import numpy as np
    from strassen_mm.model_fused_v004 import build_layer
    assert transformers.__version__=='4.56.2' and torch.__version__.split('+')[0]=='2.8.0'
    torch.set_num_threads(1);torch.manual_seed(20261003)
    results=[]
    for family,Config,Model in [('qwen3',transformers.Qwen3Config,transformers.Qwen3ForCausalLM),('mistral',transformers.MistralConfig,transformers.MistralForCausalLM),('gemma3_text',transformers.Gemma3TextConfig,transformers.Gemma3ForCausalLM)]:
        try:
            c=Config(vocab_size=97,hidden_size=96,intermediate_size=192,num_hidden_layers=6,
                num_attention_heads=2,num_key_value_heads=1,head_dim=32,sliding_window=16 if family=='gemma3_text' else None,
                rms_norm_eps=1e-6,rope_theta=1000000.,query_pre_attn_scalar=32,
                **({'rope_scaling':{'rope_type':'linear','factor':8.0}} if family=='gemma3_text' else {}))
            c._attn_implementation='eager';model=Model(c).to(torch.bfloat16).eval()
            with torch.no_grad():
                for name,value in model.named_parameters():
                    if 'norm' in name:value.copy_(torch.linspace(-.2,.3,value.numel()).reshape(value.shape)+(0 if family=='gemma3_text' else 1))
            captured={};handles=[]
            for i,layer in enumerate(model.model.layers):
                def pre(mod,args,kwargs,i=i):
                    x=args[0] if args else kwargs['hidden_states'];captured[(i,'input')]=x.detach().float().numpy()
                def post(mod,args,value,i=i):
                    captured[(i,'output')]=(value[0] if isinstance(value,tuple) else value).detach().float().numpy()
                handles += [layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post)]
            ids=torch.tensor(np.random.default_rng(20261003).integers(3,97,(2,40)),dtype=torch.long)
            with torch.inference_mode():model(ids,use_cache=False)
            for handle in handles:handle.remove()
            state=model.state_dict();cfg=c.to_dict();cfg['torch_dtype']='bfloat16'
            for i in range(6):
                prefix=f'model.layers.{i}.'
                def tensor(name):return jnp.asarray(state[prefix+name].float().numpy(),jnp.bfloat16)
                w={s:tensor('self_attn.'+s+'_proj.weight').T for s in ('q','k','v','o')}
                w.update(gateup=jnp.concatenate([tensor('mlp.gate_proj.weight').T,tensor('mlp.up_proj.weight').T],-1),down=tensor('mlp.down_proj.weight').T,
                    norm1=tensor('input_layernorm.weight'),norm2=tensor('post_attention_layernorm.weight'))
                if family!='mistral':w.update(qnorm=tensor('self_attn.q_norm.weight'),knorm=tensor('self_attn.k_norm.weight'))
                if family=='gemma3_text':w.update(norm3=tensor('pre_feedforward_layernorm.weight'),norm4=tensor('post_feedforward_layernorm.weight'))
                x=jnp.asarray(captured[i,'input'],jnp.bfloat16);ref=captured[i,'output']
                for dtype in ('bfloat16','float32'):
                    fn=build_layer(cfg,40,batch_size=2,layer_index=i,output_dtype=dtype)
                    y=fn(x,w).block_until_ready();actual=np.asarray(y,dtype=np.float32)
                    err=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30))
                    row=dict(family=family,layer=i,output_dtype=dtype,relative_l2=err,max_abs=float(abs(actual-ref).max()),finite=bool(np.isfinite(actual).all()),status='passed' if np.isfinite(actual).all() and err<=.02 else 'failed')
                    results.append(row);print(json.dumps(row),flush=True)
                jax.clear_caches()
        except Exception as e:
            results.append(dict(family=family,status='failed',error_type=type(e).__name__,error=str(e)[-2000:]));print(json.dumps(results[-1]),flush=True)
        (a.output/'cases.json').write_text(json.dumps(results,indent=2)+'\n')
    summary=dict(scope='Official tiny-model semantic qualification; synthetic weights, no real-model quality or timing claim',passed=sum(r['status']=='passed' for r in results),failed=sum(r['status']=='failed' for r in results),versions={n:md.version(n) for n in ('jax','jaxlib','torch','transformers')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    return int(summary['failed']>0)
if __name__=='__main__':raise SystemExit(main())
