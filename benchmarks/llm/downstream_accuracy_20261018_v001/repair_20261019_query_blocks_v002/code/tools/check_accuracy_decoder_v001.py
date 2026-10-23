"""Reduced Qwen fixtures test cache alignment, masking and official semantics."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from accuracy_decoder_v001 import make_layer


def fixture():
    c=dict(model_type='qwen3',hidden_size=128,head_dim=32,num_attention_heads=4,num_key_value_heads=2,
           intermediate_size=256,rms_norm_eps=1e-6,rope_theta=1000000.,num_hidden_layers=1)
    rng=np.random.default_rng(82)
    shapes=dict(q=(128,128),k=(128,64),v=(128,64),o=(128,128),gate=(128,256),up=(128,256),down=(256,128),
                norm1=(128,),norm2=(128,),qnorm=(32,),knorm=(32,))
    p={k:jnp.asarray(rng.normal(size=s)*(.05 if len(s)==2 else .1)+(0 if len(s)==2 else 1),jnp.bfloat16) for k,s in shapes.items()}
    x=jnp.asarray(rng.normal(size=(2,5,128)),jnp.bfloat16)
    return c,p,x


class Tests(unittest.TestCase):
    def test_cached_decode_matches_full_prefix(self):
        c,p,x=fixture();fn=jax.jit(make_layer(c,0));cache=jnp.zeros((2,8,2,32),jnp.bfloat16)
        whole=fn(x,p,cache,cache,jnp.tile(jnp.arange(5),(2,1)),jnp.array([5,5]))[0]
        _,k,v=fn(x[:,:4],p,cache,cache,jnp.tile(jnp.arange(4),(2,1)),jnp.array([4,4]))
        last=fn(x[:,4:],p,k,v,jnp.full((2,1),4),jnp.array([5,5]))[0]
        np.testing.assert_array_equal(np.asarray(whole[:,4:]),np.asarray(last))

    def test_padding_does_not_change_valid_prefix(self):
        c,p,x=fixture();fn=jax.jit(make_layer(c,0));cache=jnp.zeros((2,8,2,32),jnp.bfloat16)
        a=fn(x,p,cache,cache,jnp.tile(jnp.arange(5),(2,1)),jnp.array([3,5]))[0]
        y=x.at[0,3:].set(42)
        b=fn(y,p,cache,cache,jnp.tile(jnp.arange(5),(2,1)),jnp.array([3,5]))[0]
        np.testing.assert_array_equal(np.asarray(a[0,:3]),np.asarray(b[0,:3]))

    def test_official_qwen_layer_bf16(self):
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
        c,p,x=fixture();cfg=Qwen3Config(**{k:v for k,v in c.items() if k!='model_type'})
        cfg._attn_implementation='eager';model=Qwen3DecoderLayer(cfg,0).to(torch.bfloat16).eval()
        names={'q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','o':'self_attn.o_proj',
               'gate':'mlp.gate_proj','up':'mlp.up_proj','down':'mlp.down_proj','norm1':'input_layernorm',
               'norm2':'post_attention_layernorm','qnorm':'self_attn.q_norm','knorm':'self_attn.k_norm'}
        with torch.no_grad():
            for name,dest in names.items():
                val=np.asarray(p[name],dtype=np.float32)
                if val.ndim==2:val=val.T
                model.get_submodule(dest).weight.copy_(torch.tensor(val.copy(),dtype=torch.bfloat16))
            tx=torch.tensor(np.asarray(x,dtype=np.float32),dtype=torch.bfloat16)
            positions=torch.arange(5)[None].expand(2,-1)
            cos,sin=Qwen3RotaryEmbedding(cfg)(tx,positions)
            mask=torch.triu(torch.full((5,5),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16),diagonal=1)[None,None]
            expected=model(tx,attention_mask=mask,position_embeddings=(cos,sin)).float().numpy()
        cache=jnp.zeros((2,8,2,32),jnp.bfloat16)
        actual=np.asarray(jax.jit(make_layer(c,0))(x,p,cache,cache,jnp.tile(jnp.arange(5),(2,1)),jnp.array([5,5]))[0],dtype=np.float32)
        relative=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        print('official_fixture_relative_l2',float(relative))
        self.assertLess(relative,0.01)


if __name__=='__main__':unittest.main(verbosity=2)
