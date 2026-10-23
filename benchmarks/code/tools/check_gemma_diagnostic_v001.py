"""CPU integration check of official hooks and large-Gemma adapter semantics."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import Gemma3ForCausalLM,Gemma3TextConfig
from capture_gemma_intermediates_v001 import capture
from strassen_mm import model_gemma_v002 as gm,model_composed_v002 as composed
from strassen_mm.kernels_v002 import make_matmul

def main():
    torch.set_num_threads(1);torch.manual_seed(923)
    cfg=Gemma3TextConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=6,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,sliding_window=32,
        rope_scaling={'rope_type':'linear','factor':8.0})
    cfg._attn_implementation='eager';model=Gemma3ForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'norm' in name:p.copy_(torch.linspace(-.2,.3,p.numel()).reshape(p.shape).to(torch.bfloat16))
    ids=np.random.default_rng(12).integers(3,97,64,dtype=np.int32)
    output=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';output.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);model.save_pretrained(root,safe_serialization=True)
        manifest=dict(model_id='local/tiny-linear-gemma3',revision='0'*40,cache_dir=str(root),config=json.loads((root/'config.json').read_text()),
            files=[dict(path=p.name,bytes=p.stat().st_size,sha256=gm.sha256(p)) for p in root.iterdir() if p.is_file()])
        path=root/'manifest.json';path.write_text(json.dumps(manifest));cp=gm.Checkpoint(path)
        capture(model,ids,output);official=np.load(output/'official.npz');state=jax.device_put(cp.embeddings(ids));checks=[]
        for i in range(6):
            w=jax.device_put(cp.layer(i));length=len(ids);h=64;inner=128
            fn=composed.Layer(composed.prefix(cp.config,length,i),jax.jit(make_matmul('native',(length,h,2*inner))),
                composed.activation(cp.config),jax.jit(make_matmul('native',(length,inner,h))),composed.suffix(cp.config))
            value=fn(jnp.asarray(official[f'layer_{i:03}_input'][0],jnp.bfloat16),w)
            ref=official[f'layer_{i:03}_output'][0];rel=float(np.linalg.norm(np.asarray(value,dtype=float)-ref)/np.linalg.norm(ref))
            checks.append(dict(layer=i,relative_l2=rel));assert rel<.02,checks[-1]
            state=fn(state,w)
        ref=official['final_hidden'][0];got=gm.rms(state,jnp.asarray(cp.tensor('model.norm.weight')),cp.config['rms_norm_eps'])
        rel=float(np.linalg.norm(np.asarray(got,dtype=float)-ref)/np.linalg.norm(ref));assert rel<.03,rel
    report=dict(passed=True,checks=checks,final_hidden_relative_l2=rel,scope='Synthetic CPU integration only; not real-checkpoint qualification')
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
