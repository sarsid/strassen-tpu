"""Independent Gemma4 semantics and parent Pallas interpreter validation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import types
import numpy as np
import jax
import jax.numpy as jnp
import torch
from transformers import Gemma4TextConfig,Gemma4ForCausalLM
from transformers.models.gemma4 import modeling_gemma4 as official
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import gemma4_parent_model_v002 as model
import gemma4_batch_stage_v002 as stage
model.setup();stage.jax,stage.jnp,stage.np=jax,jnp,np
REAL=copy.deepcopy(model.config)


def array(x):return jnp.asarray(x.detach().float().numpy(),jnp.bfloat16)


def raw(block):
    p={s:array(getattr(block.self_attn,s+'_proj').weight.T) for s in ('q','k','v','o') if getattr(block.self_attn,s+'_proj') is not None}
    p.update(gateup=array(torch.cat((block.mlp.gate_proj.weight,block.mlp.up_proj.weight)).T),down=array(block.mlp.down_proj.weight.T),qnorm=array(block.self_attn.q_norm.weight),knorm=array(block.self_attn.k_norm.weight),layer_scalar=array(block.layer_scalar))
    for key,name in [('norm1','input_layernorm'),('norm2','post_attention_layernorm'),('norm3','pre_feedforward_layernorm'),('norm4','post_feedforward_layernorm')]:p[key]=array(getattr(block,name).weight)
    return p


class GemmaParent(unittest.TestCase):
    def tearDown(self):model.config=copy.deepcopy(REAL);jax.clear_caches()

    def test_attention_rounding_against_official_real_head_widths(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(109)
        for dim,kv,window in ((256,16,8),(512,4,None)):
            shape=(2,32,32,dim);kvshape=(2,32,kv,dim)
            q=torch.tensor(rng.normal(0,.25,shape),dtype=torch.bfloat16)
            k=torch.tensor(rng.normal(0,.5,kvshape),dtype=torch.bfloat16)
            v=torch.tensor(rng.normal(0,1,kvshape),dtype=torch.bfloat16)
            mask=model.native.attention_mask(32,window)
            torchmask=torch.tensor(np.asarray(mask,dtype=np.float32),dtype=torch.bfloat16)[None,None]
            with torch.no_grad():
                reference,_=official.eager_attention_forward(types.SimpleNamespace(num_key_value_groups=32//kv,training=False),q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),torchmask,scaling=1.)
            jq,jk,jv=array(q),array(k),array(v);allowed=mask==0
            corrected=jax.jit(model.attention_bf16_scores)(jq,jk,jv,allowed)
            previous=jax.jit(lambda q,k,v:jax.nn.dot_product_attention(q,k,v,mask=allowed[None,None],scale=1.,implementation='xla'))(jq,jk,jv)
            good=stage.error(array(reference),corrected);old=stage.error(array(reference),previous)
            print(json.dumps(dict(test='attention_rounding',head_dim=dim,kv_heads=kv,window=window,corrected=good,previous=old)),flush=True)
            self.assertLess(good['relative_l2'],.001)
            self.assertLess(good['relative_l2'],old['relative_l2']/3)

    def test_all_registered_candidates_are_legal(self):
        for b in (1,4,8,16,32,64):
            for layer in (0,5):
                g=model.geometry(REAL,layer,b,1024)
                for arm,sites in [('cubic_tuned',('gateup','o','down')),('strassen_tuned',('q','k','gateup','o','down'))]:
                    for site in sites:
                        if layer==5 and site=='k':continue
                        menu=stage.candidates(site,arm,layer,b,1024);self.assertTrue(menu)
                        for entry in menu:
                            shape=g['projection_shapes_mkn'][site];bm,bn,bk=entry['tile']
                            self.assertEqual((shape[0]%bm,shape[1]%bk,shape[2]%bn),(0,0,0))
                            if entry['fused'] and site in ('o','down'):self.assertEqual(bn,shape[2])
                            if site in ('q','k'):self.assertEqual(bn%(2*g['head_dim']),0)

    def test_native_against_official_two_types(self):
        torch.set_num_threads(1);torch.manual_seed(29)
        cfg=Gemma4TextConfig(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,global_head_dim=32,num_global_key_value_heads=1,attention_k_eq_v=True,hidden_size_per_layer_input=0,layer_types=['sliding_attention','full_attention'],sliding_window=8,max_position_embeddings=8192,final_logit_softcapping=3.,dtype='bfloat16')
        cfg._attn_implementation='eager';net=Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval();net.model.rotary_emb=official.Gemma4TextRotaryEmbedding(cfg)
        model.config=cfg.to_dict();batch,sequence=2,17
        tokens=torch.randint(1,97,(batch,sequence));captured=[]
        for layer in net.model.layers:layer.register_forward_hook(lambda m,args,out:captured.append((args[0].detach().clone(),out.detach().clone())))
        with torch.no_grad():reference=net(tokens,use_cache=False).logits
        hidden=model.native.scaled_embeddings(array(net.model.embed_tokens.weight[tokens]),32).reshape(-1,32)
        for index,block in enumerate(net.model.layers):
            p=raw(block);p['cos'],p['sin']=model.native.rotary_tables(model.config,index,jnp.arange(sequence)[None])
            profile=model.default_profile(batch,sequence)
            fn=jax.jit(model.build_layer('native',profile,index,batch,sequence))
            isolated=fn(array(captured[index][0]).reshape(-1,32),p)
            self.assertLess(stage.error(array(captured[index][1]).reshape(-1,32),isolated)['relative_l2'],.02)
            hidden=fn(hidden,p)
        output=jax.jit(lambda h,n,w:model.native.head_logits(h,n,w,model.config))(hidden,array(net.model.norm.weight),array(net.lm_head.weight.T))
        self.assertLess(stage.error(array(reference).reshape(-1,97),output)['relative_l2'],.03)

    def test_parent_fused_projection_semantics(self):
        # Aligned interpreter fixture with actual local/global head widths.
        rng=np.random.default_rng(19);h,inner,m=512,1024,16
        cfg=copy.deepcopy(REAL);cfg.update(hidden_size=h,intermediate_size=inner,num_attention_heads=4,num_key_value_heads=2,num_global_key_value_heads=1)
        model.config=cfg
        def rand(shape,scale=.03):return jnp.asarray(rng.normal(0,scale,shape),jnp.bfloat16)
        for index in (0,5):
            g=model.geometry(cfg,index,1,m);p={s:rand(tuple(shape[1:])) for s,shape in g['projection_shapes_mkn'].items()}
            p.update({k:jnp.ones((h,),jnp.bfloat16) for k in ('norm1','norm2','norm3','norm4')})
            p.update(qnorm=jnp.ones((g['head_dim'],),jnp.bfloat16),knorm=jnp.ones((g['head_dim'],),jnp.bfloat16),layer_scalar=jnp.ones((1,),jnp.bfloat16))
            profile=model.default_profile(1,m)
            for site in ('q','k'):profile['strassen'][site]={'tile':[16,2*g['head_dim'],256],'fused':True}
            profile['strassen']['gateup']={'tile':[16,512,256],'fused':True}
            for site in ('o','down'):
                profile['strassen'][site]={'tile':[16,h,256],'fused':True};profile['cubic'][site]={'tile':[16,256,256]}
            profile['cubic']['gateup']={'tile':[16,512,256]}
            pp=model.prepare_weights(p,profile,index,1,m)
            for site in ('q','k','gateup','o','down'):
                shape=g['projection_shapes_mkn'][site];x=rand((m,shape[1]),.3);residual=rand((m,h),.1) if site in ('o','down') else None
                reference=jax.jit(model.projection(site,'native',profile,index,1,m))(x,pp,residual)
                for arm in ('cubic_tuned','strassen_tuned'):
                    output=jax.jit(model.projection(site,arm,profile,index,1,m,interpret=True))(x,pp,residual)
                    metric=stage.error(reference,output)
                    self.assertTrue(metric['finite']);self.assertLess(metric['relative_l2'],.03,(index,site,arm,metric))


if __name__=='__main__':unittest.main(verbosity=2)
