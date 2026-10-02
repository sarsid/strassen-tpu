"""Independent official-model checks for new Gemma12B text loading and linear RoPE."""
import json,os,tempfile
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp
from safetensors.torch import load_file,save_file
from transformers import Gemma3TextConfig,Gemma3ForCausalLM
from strassen_mm import model_gemma_v002 as gm,model_composed_v002 as composed
from strassen_mm.kernels_v002 import make_matmul
import qualify_gemma_v001 as q

def main():
    checks=[];torch.set_num_threads(1);torch.manual_seed(922)
    c=Gemma3TextConfig(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,sliding_window=8,
        layer_types=['sliding_attention','full_attention'],rope_scaling={'rope_type':'linear','factor':8.0},
        rope_theta=10000.,rope_local_base_freq=100.,query_pre_attn_scalar=16,
        pad_token_id=0,bos_token_id=1,eos_token_id=2,tie_word_embeddings=True)
    c._attn_implementation='eager'
    model=Gemma3ForCausalLM(c).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'norm' in n:p.copy_(torch.linspace(-.2,.3,p.numel()).reshape(p.shape).to(torch.bfloat16))
    ids=np.random.default_rng(922).integers(3,97,33,dtype=np.int64)
    ref=q.capture_official(model,ids)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);model.save_pretrained(root,safe_serialization=True)
        old=load_file(root/'model.safetensors')
        save_file({'language_model.'+k:v for k,v in old.items()},root/'renamed.safetensors')
        (root/'model.safetensors').unlink();(root/'renamed.safetensors').rename(root/'model.safetensors')
        config={'model_type':'gemma3','text_config':c.to_dict(),'vision_config':{'model_type':'siglip_vision_model'},'torch_dtype':'bfloat16'}
        (root/'config.json').write_text(json.dumps(config))
        manifest=dict(model_id='local/official-tiny-gemma3-multimodal-layout',revision='0'*40,cache_dir=str(root),config=config,
            files=[dict(path=f.name,bytes=f.stat().st_size,sha256=gm.sha256(f)) for f in root.iterdir() if f.is_file()])
        (root/'manifest.json').write_text(json.dumps(manifest))
        cp=gm.Checkpoint(root/'manifest.json');m,_=q.compare_adapter(cp,ref,ids,gm)
        checks.extend(dict(name=k,**v) for k,v in m.items())
        for index in range(2):
            weights=jax.tree_util.tree_map(jnp.asarray,cp.layer(index))
            fn=composed.Layer(composed.prefix(cp.config,len(ids),index),jax.jit(make_matmul('native',(33,32,128))),
                composed.activation(cp.config),jax.jit(make_matmul('native',(33,64,32))),composed.suffix(cp.config))
            got=fn(jnp.asarray(ref[f'layer_{index:03d}_input'],dtype=jnp.bfloat16),weights)
            checks.append(dict(name=f'composed_official_layer{index}',**q.metrics(ref[f'layer_{index:03d}_output'],got,layer=True)))
        # An incorrect factor must be observably distinguishable from the official global RoPE.
        x=jnp.asarray(np.random.default_rng(1).normal(size=(33,4,16)),dtype=jnp.bfloat16)
        assert np.linalg.norm(np.asarray(gm.rope(x,10000,8),np.float32)-np.asarray(gm.rope(x,10000,1),np.float32))>1
    source=Path(__file__).resolve().parents[1]
    from strassen_mm import benchmark_mlsys_shapes_v001 as grid
    cfg=json.loads((source/'configs/large_real_models_v001/campaign.json').read_text())
    arms=grid.candidates(cfg);assert len(arms)==52
    assert all(sum(a['family']==f for a in arms)==16 for f in ['cubic','one_level','two_level'])
    assert cfg['token_shape']==[32,2048] and cfg['quality_windows']==16 and cfg['capture_windows']==32
    shapes=json.loads((source/'configs/large_real_models_v001/shapes.json').read_text())['shapes'];assert len(shapes)==18
    for key,model in [('qwen','Qwen--Qwen3-8B'),('mistral','mistralai--Mistral-7B-v0.3'),('gemma','google--gemma-3-12b-pt')]:
        c=json.loads((source/'plans/large_real_models_v001'/(model+'.json')).read_text())['config'];c=c.get('text_config',c)
        for m in [2048,8192,16384]:
            s=next(s for s in shapes if s['id']==f'{key}_gateup_m{m}')
            assert (s['m'],s['k'],s['n'])==(m,c['hidden_size'],2*c['intermediate_size'])
            assert 4*m<=32*2048
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,registered_candidates=len(arms),
        scope='CPU correctness: independent official tiny Gemma with linear global and unscaled local RoPE, actual multimodal tensor naming, composed stages. No TPU performance.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    assert report['passed']
if __name__=='__main__':main()
