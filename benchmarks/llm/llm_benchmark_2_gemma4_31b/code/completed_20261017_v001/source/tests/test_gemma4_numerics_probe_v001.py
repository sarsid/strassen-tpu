import copy
import json
from pathlib import Path
import sys
import unittest
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import gemma4_numerics_probe_v001 as probe


class MatchedProbe(unittest.TestCase):
    def test_independent_operations_and_hook_cleanup(self):
        probe.model.setup();saved=copy.deepcopy(probe.model.config)
        torch.set_num_threads(1);torch.manual_seed(117)
        cfg=Gemma4TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,global_head_dim=32,num_global_key_value_heads=1,attention_k_eq_v=True,hidden_size_per_layer_input=0,layer_types=['sliding_attention','full_attention'],sliding_window=8,max_position_embeddings=8192,dtype='bfloat16')
        cfg._attn_implementation='eager';probe.model.config=cfg.to_dict()
        original=official.eager_attention_forward
        try:
            for index in (0,1):
                layer=official.Gemma4TextDecoderLayer(cfg,index).to(torch.bfloat16).eval()
                x=torch.randn(1,16,64,dtype=torch.bfloat16);pos=official.Gemma4TextRotaryEmbedding(cfg)(x,torch.arange(16)[None],cfg.layer_types[index])
                mask=probe.model.native.attention_mask(16,8 if index==0 else None)
                mask=torch.tensor(probe.host(mask),dtype=torch.bfloat16)[None,None]
                with torch.no_grad():expected=layer(x,position_embeddings=pos,attention_mask=mask)
                capture=probe.capture(layer,x,pos,mask)
                self.assertTrue(torch.equal(expected,capture['output']))
                self.assertIs(official.eager_attention_forward,original)
                self.assertTrue(all(not m._forward_hooks for m in layer.modules()))
                rows=probe.Probe(('cpu',)).layer(capture,index,lambda r:print(json.dumps(r),flush=True))
                self.assertGreaterEqual(len(rows),20)
                self.assertTrue(all(r['backends']['cpu']['finite'] for r in rows))
                self.assertLess(max(r['backends']['cpu']['relative_l2'] for r in rows),.01)
        finally:probe.model.config=saved


if __name__=='__main__':unittest.main()
