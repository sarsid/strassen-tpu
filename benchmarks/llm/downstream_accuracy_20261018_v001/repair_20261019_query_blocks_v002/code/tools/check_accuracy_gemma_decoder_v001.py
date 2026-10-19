"""Gemma local/global cache checks against the previously qualified native path."""
import json
from pathlib import Path
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from accuracy_decoder_v001 import make_layer,geometry
from strassen_mm.model_gemma4_v001 import build_layer


def fixture(index):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'plans/llm_accuracy_20261018_v001/metadata/gemma4_31b.config.json').read_text())['text_config']
    c.update(hidden_size=128,intermediate_size=256,num_attention_heads=4,num_key_value_heads=2,
             num_global_key_value_heads=1,head_dim=32,global_head_dim=64,num_hidden_layers=6,
             layer_types=['sliding_attention']*5+['full_attention'],sliding_window=4)
    g=geometry(c,index);rng=np.random.default_rng(32+index)
    shapes={k:tuple(v[1:]) for k,v in g['projection_shapes_mkn'].items() if k!='gateup'}
    shapes.update(gate=(128,256),up=(128,256),norm1=(128,),norm2=(128,),norm3=(128,),norm4=(128,),
                  qnorm=(g['head_dim'],),knorm=(g['head_dim'],),layer_scalar=(1,))
    p={k:jnp.asarray(rng.normal(size=s)*(.05 if len(s)==2 else .1)+(0 if len(s)==2 else 1),jnp.bfloat16) for k,s in shapes.items()}
    x=jnp.asarray(rng.normal(size=(2,5,128)),jnp.bfloat16)
    return c,p,x


class Tests(unittest.TestCase):
    def test_local_and_global_match_native_and_cache(self):
        for index in (0,5):
            c,p,x=fixture(index);g=geometry(c,index)
            cache=jnp.zeros((2,8,g['kv_heads'],g['head_dim']),jnp.bfloat16)
            fn=jax.jit(make_layer(c,index))
            actual=fn(x,p,cache,cache,jnp.tile(jnp.arange(5),(2,1)),jnp.array([5,5]))[0]
            reference_p={k:v for k,v in p.items() if k not in ('gate','up')}
            reference_p['gateup']=jnp.concatenate((p['gate'],p['up']),axis=1)
            expected=build_layer(c,5,batch_size=2,layer_index=index)(x,reference_p)
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
            _,k,v=fn(x[:,:4],p,cache,cache,jnp.tile(jnp.arange(4),(2,1)),jnp.array([4,4]))
            last=fn(x[:,4:],p,k,v,jnp.full((2,1),4),jnp.array([5,5]))[0]
            np.testing.assert_array_equal(np.asarray(last),np.asarray(actual[:,4:]))


if __name__=='__main__':unittest.main(verbosity=2)
