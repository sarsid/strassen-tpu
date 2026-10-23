import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import gemma4_fp32_semantics_v001 as control


class FP32Semantics(unittest.TestCase):
    def test_local_global_layers_and_boundary_restoration(self):
        model=control.load_model(Path(__file__).resolve().parents[1]);saved=copy.deepcopy(model.config)
        rounded,bf32=model.native.rounded,model.native.bf32
        cfg=Gemma4TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,global_head_dim=32,num_global_key_value_heads=1,attention_k_eq_v=True,hidden_size_per_layer_input=0,layer_types=['sliding_attention','full_attention'],sliding_window=8,max_position_embeddings=8192)
        cfg._attn_implementation='eager';model.config=cfg.to_dict();torch.manual_seed(2026);torch.set_num_threads(1)
        try:
            for index in (0,1):
                block=official.Gemma4TextDecoderLayer(cfg,index).float().eval()
                array=lambda x:x.detach().numpy().copy()
                raw={s:array(getattr(block.self_attn,s+'_proj').weight.T) for s in ('q','k','v','o') if getattr(block.self_attn,s+'_proj') is not None}
                raw.update(gateup=array(torch.cat((block.mlp.gate_proj.weight,block.mlp.up_proj.weight)).T),down=array(block.mlp.down_proj.weight.T),qnorm=array(block.self_attn.q_norm.weight),knorm=array(block.self_attn.k_norm.weight),layer_scalar=array(block.layer_scalar))
                for key,name in [('norm1','input_layernorm'),('norm2','post_attention_layernorm'),('norm3','pre_feedforward_layernorm'),('norm4','post_feedforward_layernorm')]:raw[key]=array(getattr(block,name).weight)
                result=control.compare(raw,index,np.random.default_rng(index).normal(size=(2,17,64)).astype(np.float32))
                print(result,flush=True);self.assertTrue(result['finite']);self.assertLess(result['relative_l2'],1e-5)
                self.assertIs(model.native.rounded,rounded);self.assertIs(model.native.bf32,bf32)
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with control.fp32_boundaries():raise RuntimeError('fixture')
            self.assertIs(model.native.rounded,rounded);self.assertIs(model.native.bf32,bf32)
        finally:model.config=saved;model.jax.clear_caches()


if __name__=='__main__':unittest.main(verbosity=2)
