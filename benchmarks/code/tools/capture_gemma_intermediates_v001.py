"""Capture exact official Gemma intermediates without changing its operations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
from transformers.models.gemma3 import modeling_gemma3 as hf


def capture(model, ids, out):
    backbone = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    selected = {0, min(5, len(backbone.layers)-1), len(backbone.layers)-1}
    arrays = {}; handles = []
    def host(x):
        return x.detach().float().cpu().numpy().copy()
    def hook(name, module):
        def before(mod, args): arrays[name+'_input'] = host(args[0])
        def after(mod, args, value): arrays[name+'_output'] = host(value[0] if isinstance(value, tuple) else value)
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    hook('embedding', backbone.embed_tokens); hook('final_norm', backbone.norm)
    for i, layer in enumerate(backbone.layers):
        hook(f'layer_{i:03}', layer)
        if i in selected:
            for name, module in dict(norm1=layer.input_layernorm, norm2=layer.post_attention_layernorm,
                norm3=layer.pre_feedforward_layernorm, norm4=layer.post_feedforward_layernorm,
                qnorm=layer.self_attn.q_norm, knorm=layer.self_attn.k_norm,
                q=layer.self_attn.q_proj, k=layer.self_attn.k_proj, v=layer.self_attn.v_proj,
                o=layer.self_attn.o_proj, gate=layer.mlp.gate_proj, up=layer.mlp.up_proj,
                gelu=layer.mlp.act_fn, down=layer.mlp.down_proj).items():
                hook(f'op_{i:03}_{name}', module)
    original_rope, original_attn = hf.apply_rotary_pos_emb, hf.eager_attention_forward
    counts = dict(rope=0, attn=0)
    def rope(*args, **kwargs):
        i=counts['rope']; counts['rope']+=1; value=original_rope(*args, **kwargs)
        if i in selected:
            for name,x in zip(['q','k','cos','sin'],args[:4]): arrays[f'rope_{i:03}_{name}_input']=host(x)
            for name,x in zip(['q','k'],value): arrays[f'rope_{i:03}_{name}_output']=host(x)
        return value
    def attn(module,q,k,v,mask,*args,**kwargs):
        i=counts['attn'];counts['attn']+=1;value=original_attn(module,q,k,v,mask,*args,**kwargs)
        if i in selected:
            for name,x in [('q',q),('k',k),('v',v),('mask',mask),('output',value[0]),('prob',value[1])]:
                if x is not None:arrays[f'attn_{i:03}_{name}']=host(x)
        return value
    hf.apply_rotary_pos_emb, hf.eager_attention_forward = rope, attn
    try:
        with torch.inference_mode():
            result=model(torch.as_tensor(ids[None],dtype=torch.long),use_cache=False,output_hidden_states=True)
        arrays['logits']=host(result.logits);arrays['final_hidden']=host(result.hidden_states[-1])
    finally:
        hf.apply_rotary_pos_emb, hf.eager_attention_forward = original_rope, original_attn
        for handle in handles:handle.remove()
    assert counts==dict(rope=len(backbone.layers),attn=len(backbone.layers)),counts
    np.savez_compressed(out/'official.npz',**arrays)
    meta=dict(selected_layers=sorted(selected),counts=counts,config=backbone.config.to_dict(),
        model_class=type(model).__name__,embedding_scale=float(backbone.embed_tokens.embed_scale),
        embedding_scale_dtype=str(backbone.embed_tokens.embed_scale.dtype),
        global_inv_freq_dtype=str(backbone.rotary_emb.inv_freq.dtype),
        local_inv_freq_dtype=str(backbone.rotary_emb_local.inv_freq.dtype),keys=list(arrays))
    (out/'official.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k not in ('config','keys')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    bundle=json.loads(a.model_input.read_text());manifest=json.loads(Path(bundle['model_manifest']).read_text())
    tm=Path(bundle['tokens_manifest']);tokens=json.loads(tm.read_text());ids=np.load(tm.parent/tokens['token_array_path']).reshape(-1)[:64]
    torch.set_num_threads(1)
    cls=Gemma3ForConditionalGeneration if manifest['config']['model_type']=='gemma3' else Gemma3ForCausalLM
    model=cls.from_pretrained(manifest['cache_dir'],local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).eval()
    capture(model,ids,a.output_dir)

if __name__=='__main__':main()
