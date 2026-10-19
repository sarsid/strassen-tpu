"""Verify query-block engine masking and independent cached histories on CPU."""
import unittest
import numpy as np
import jax.numpy as jnp
from accuracy_engine_v004 import Engine
from accuracy_decoder_v001 import make_layer,geometry
from check_accuracy_decoder_v001 import fixture as qwen_fixture
from check_accuracy_gemma_decoder_v001 import fixture as gemma_fixture

ARMS=('native','strassen')

class Tests(unittest.TestCase):
    def engine(self,kind='qwen',index=0):
        c,p,_=qwen_fixture() if kind=='qwen' else gemma_fixture(index)
        e=object.__new__(Engine);e.config=dict(c);e.qwen=kind=='qwen';e.emit=lambda _:None
        if not e.qwen:
            e.config['layer_types']=[c['layer_types'][index]]*c['num_hidden_layers']
            e.config['layer_types'][-1]='full_attention'
        rng=np.random.default_rng(124)
        e.embedding=np.asarray(jnp.asarray(rng.normal(size=(256,128)),jnp.bfloat16))
        e.weights=[p,p];e.executables={}
        layer_kind='qwen' if e.qwen else c['layer_types'][index]
        e.functions={(arm,layer_kind):make_layer(c,index) for arm in ARMS}
        e.device_norm=jnp.ones(128,jnp.bfloat16);e.device_head=jnp.asarray(rng.normal(size=(128,256))*.1,jnp.bfloat16)
        if not e.qwen:
            # The test returns hidden states, avoiding unrelated head configuration.
            pass
        return e

    def compare(self,kind='qwen',index=0):
        e=self.engine(kind,index);e.reset(2,64)
        rng=np.random.default_rng(492)
        tokens={arm:rng.integers(0,256,size=(2,32),dtype=np.int32) for arm in ARMS}
        lengths={arm:np.array([17,19],np.int32) for arm in ARMS}
        positions={arm:np.tile(np.arange(32,dtype=np.int32),(2,1)) for arm in ARMS}
        full,_,_=e.forward(tokens,positions,lengths,last_indices={a:lengths[a]-1 for a in ARMS},return_hidden=True)
        cached=self.engine(kind,index);cached.reset(2,64)
        prefix={a:tokens[a].copy() for a in ARMS}
        for a in ARMS:
            for row,end in enumerate(lengths[a]-1):prefix[a][row,end:]=0
        cached.forward(prefix,positions,{a:lengths[a]-1 for a in ARMS},return_hidden=True)
        one={a:tokens[a][np.arange(2),lengths[a]-1,None] for a in ARMS}
        actual,_,_=cached.forward(one,{a:(lengths[a]-1)[:,None] for a in ARMS},lengths,last_indices={a:np.zeros(2,np.int32) for a in ARMS},return_hidden=True)
        for a in ARMS:np.testing.assert_array_equal(np.asarray(actual[a]),np.asarray(full[a]))
        self.assertFalse(np.array_equal(actual['native'],actual['strassen']))
        self.assertTrue(all(key[2]==(2,16,128) for key in cached.executables))
        altered={a:tokens[a].copy() for a in ARMS}
        for a in ARMS:
            for row,end in enumerate(lengths[a]):altered[a][row,end:]=255
        e.reset(2,64)
        changed,_,_=e.forward(altered,positions,lengths,last_indices={a:lengths[a]-1 for a in ARMS},return_hidden=True,retain_cache=False)
        for a in ARMS:np.testing.assert_array_equal(np.asarray(changed[a]),np.asarray(full[a]))
        self.assertTrue(all(not np.any(np.asarray(k)) for states in e.caches.values() for k,v in states))

    def test_qwen_variable_lengths_and_block_boundary(self):self.compare()
    def test_gemma_local_variable_lengths_and_block_boundary(self):self.compare('gemma',0)
    def test_gemma_global_variable_lengths_and_block_boundary(self):self.compare('gemma',5)

if __name__=='__main__':unittest.main(verbosity=2)
