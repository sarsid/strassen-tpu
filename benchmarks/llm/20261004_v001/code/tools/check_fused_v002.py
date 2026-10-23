"""Bounded CPU-interpret qualification, never a TPU speed measurement."""
import dataclasses
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

environment_before = dict(os.environ)
from strassen_mm.fusion_v001 import Epilogue, reference, unpack_rotary
from strassen_mm.kernels_fused_v002 import make_projection
from strassen_mm.tuner_joint_v001 import arm
from strassen_mm.tuner_fused_v001 import registry
from strassen_mm.model_fused_v002 import build_layer, native_policy, validate_config
from strassen_mm import model_n9_v001 as qm, model_gemma_v002 as gm, model_composed_v002 as composed
from strassen_mm.kernels_joint_v001 import make_matmul as preserved_mm

out = Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
out.mkdir()
checks = []
rng = np.random.default_rng(1002001)
assert dict(os.environ) == environment_before, 'Import modified process environment'


def passed(name,**fields):
    row = dict(test=name,**fields); checks.append(row)
    print(json.dumps(row),flush=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')


def choice(impl='current', depth=1, mode='products', dtype='bfloat16', tile=(32,512,512)):
    a = dict(arm(tile,depth,mode,2,dtype),architecture='v6e',implementation=impl)
    if impl != 'current':
        a['depth'] = 0
    if impl == 'native':
        a['tile'] = None
    return a


def close(actual, expected, *, atol=2e-5, rtol=2e-5):
    a,b = np.asarray(actual,dtype=np.float32),np.asarray(expected,dtype=np.float32)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
    return float(np.max(np.abs(a-b)))


def auxiliary(shape,spec):
    m,k,n = shape
    all_values = dict(residual=jnp.asarray(rng.normal(size=(m,n)),jnp.bfloat16),
        bias=jnp.asarray(rng.normal(size=(n,)),jnp.bfloat16),
        scale=jnp.asarray(rng.normal(size=spec.head_dim if spec.rotary else n)*.1+(.0 if spec.norm=='gemma' else 1),jnp.bfloat16))
    theta = 10000.0
    frequency = jnp.arange(m,dtype=jnp.float32)[:,None]/theta**(jnp.arange(0,spec.head_dim,2,dtype=jnp.float32)/spec.head_dim)[None,:]
    all_values.update(cos=jnp.cos(frequency).astype(jnp.bfloat16),sin=jnp.sin(frequency).astype(jnp.bfloat16))
    return {key:all_values[key] for key in spec.auxiliaries}


# Sparse integer oracles make every recursive pre-add exact, isolating
# padding/layout/panel-order/epilogue errors from Strassen approximation.
implementations = [('native',0,'products'),('cubic',0,'outputs'),('cubic_full',0,'outputs')]
implementations += [('current',d,mode) for d in (1,2) for mode in ('products','outputs')]
for dtype in ('float32','bfloat16'):
    for impl,depth,mode in implementations:
        for spec in (Epilogue('swiglu'),Epilogue('residual_add'),Epilogue('qk_norm_rope')):
            shape = (33,513,768 if spec.rotary else 770)
            aa = rng.integers(-1,2,size=shape[:2]).astype(np.float32)
            aa[:,np.arange(shape[1])%64!=0] = 0
            bb = rng.integers(-1,2,size=shape[1:]).astype(np.float32)
            a,b = jnp.asarray(aa,jnp.bfloat16),jnp.asarray(bb,jnp.bfloat16)
            aux = auxiliary(shape,spec)
            exact = jnp.asarray(aa.astype(np.float64)@bb.astype(np.float64),jnp.float32)
            expected = jax.jit(lambda x,aux:reference(x,spec,jnp.dtype(dtype),**aux))(exact,aux)
            fn = make_projection(choice(impl,depth,mode,dtype),shape,spec,interpret=True)
            actual = jax.jit(fn)(a,b,**aux)
            error = close(actual,expected,atol=2e-3 if dtype=='bfloat16' else 3e-5,rtol=2e-3 if dtype=='bfloat16' else 3e-5)
            assert actual.dtype == jnp.dtype(dtype)
            prepared = jax.jit(fn.prepared)(a,fn.prepare_weights(b),**aux)
            close(prepared,actual,atol=0,rtol=0)
            passed('exact_integer_multitile_multipanel',dtype=dtype,impl=impl,depth=depth,mode=mode,epilogue=spec.kind,max_abs=error)
            jax.clear_caches()

# Less common epilogues, both numerical contracts, Gemma offset norm and
# full-row normalization with padding. Early finalization is separately named.
specs = [Epilogue('none'),Epilogue('geglu'),Epilogue('bias_add'),Epilogue('rope',head_dim=256),
         Epilogue('qk_norm_rope',norm='gemma',head_dim=256),Epilogue('norm_residual_add',norm='gemma'),
         Epilogue('norm_residual_add',norm='qwen'),Epilogue('swiglu',early=True)]
for base_spec in specs:
    for rounding in ('model','accumulator'):
        spec = dataclasses.replace(base_spec,rounding=rounding)
        shape = (17,513,768 if spec.rotary else 258)
        aa = rng.integers(-1,2,size=shape[:2]).astype(np.float32); aa[:,np.arange(shape[1])%64!=0]=0
        bb = rng.integers(-1,2,size=shape[1:]).astype(np.float32)
        a,b = jnp.asarray(aa,jnp.bfloat16),jnp.asarray(bb,jnp.bfloat16)
        exact = jnp.asarray(aa.astype(np.float64)@bb.astype(np.float64),jnp.float32)
        aux = auxiliary(shape,spec)
        for dtype in ('float32','bfloat16'):
            fn = make_projection(choice(mode='outputs',dtype=dtype),shape,spec,interpret=True)
            actual = jax.jit(fn)(a,b,**aux)
            expected = jax.jit(lambda x,aux:reference(x,spec,jnp.dtype(dtype),**aux))(exact,aux)
            error = close(actual,expected,atol=2e-3 if dtype=='bfloat16' else 3e-5,rtol=2e-3 if dtype=='bfloat16' else 3e-5)
            passed('additional_epilogue_contract',epilogue=spec.kind,rounding=rounding,norm=spec.norm,dtype=dtype,early=spec.early,max_abs=error)
            jax.clear_caches()

# Gaussian replay against PRESERVED S1/S2 arithmetic, not an algebraically
# equivalent custom implementation. Packing is accounted for in the oracle.
for depth in (1,2):
    for mode in ('products','outputs'):
        shape = (17,513,770); tile=(32,512,512)
        a=jnp.asarray(rng.normal(size=shape[:2])/np.sqrt(shape[1]),jnp.bfloat16)
        b=jnp.asarray(rng.normal(size=shape[1:]),jnp.bfloat16)
        spec=Epilogue('swiglu')
        fn=make_projection(choice(depth=depth,mode=mode,dtype='float32'),shape,spec,interpret=True)
        packed=fn.prepare_weights(b); padded=tuple(fn.metadata['padded_shape_mkn'])
        pa=jnp.pad(a,((0,padded[0]-shape[0]),(0,padded[1]-shape[1])))
        old=preserved_mm(padded,tile,depth=depth,accumulator=mode,interpret=True)
        projection=jax.jit(old)(pa,packed)[:shape[0]]
        halves=projection.reshape(shape[0],-1,2,tile[1]//2).transpose(0,2,1,3).reshape(shape[0],2,-1)
        natural=jnp.concatenate((halves[:,0,:shape[2]//2],halves[:,1,:shape[2]//2]),-1)
        expected=reference(natural,spec,jnp.float32)
        actual=jax.jit(fn.prepared)(a,packed)
        error=close(actual,expected,atol=3e-5,rtol=3e-5)
        bf=make_projection(choice(depth=depth,mode=mode),shape,spec,interpret=True)
        close(jax.jit(bf.prepared)(a,bf.prepare_weights(b)),actual.astype(jnp.bfloat16),atol=0,rtol=0)
        passed('preserved_gaussian_arithmetic_and_final_cast',depth=depth,mode=mode,max_abs=error)
        jax.clear_caches()

# Whole-layer Native agrees with the existing model adapter, including Gemma
# local/global attention; batched execution agrees with independent sequences.
for model in ('qwen3','mistral','gemma3_text'):
    c=dict(model_type=model,hidden_size=512,intermediate_size=384,num_hidden_layers=2,
           num_attention_heads=2,num_key_value_heads=1,head_dim=128,rms_norm_eps=1e-6,
           rope_theta=10000.,torch_dtype='bfloat16',sliding_window=None)
    if model=='gemma3_text':
        c.update(head_dim=256,sliding_window=3,layer_types=['sliding_attention','full_attention'],
                 rope_scaling={'rope_type':'linear','factor':8.0})
    c=validate_config(c); length=7; h=c['hidden_size']; d=c['head_dim']
    def random(shape,scale=1):
        return jnp.asarray(rng.normal(size=shape)*scale,jnp.bfloat16)
    w={name:random(shape,1/np.sqrt(shape[0])) for name,shape in dict(q=(h,2*d),k=(h,d),v=(h,d),o=(2*d,h),gateup=(h,768),down=(384,h)).items()}
    w.update(norm1=random((h,),.1)+(0 if model=='gemma3_text' else 1),norm2=random((h,),.1)+(0 if model=='gemma3_text' else 1))
    if model!='mistral':
        w.update(qnorm=random((d,),.1)+(0 if model=='gemma3_text' else 1),knorm=random((d,),.1)+(0 if model=='gemma3_text' else 1))
    if model=='gemma3_text':
        w.update(norm3=random((h,),.1),norm4=random((h,),.1))
    x=random((2,length,h))
    for index in (range(2) if model=='gemma3_text' else [0]):
        # Explicit preserved composition supplies an independent semantic baseline.
        prefix=composed.prefix(c,length,index); activate=composed.activation(c); suffix=composed.suffix(c)
        expected=[]
        for item in x:
            residual,norm=prefix(item,w)
            middle=activate(qm.dot(norm,w['gateup']))
            expected.append(suffix(qm.dot(middle,w['down']),residual,w))
        expected=jnp.stack(expected)
        fn=build_layer(c,length,batch_size=2,layer_index=index,interpret=True)
        actual=fn.prepared(x,fn.prepare_weights(w))
        error=close(actual,expected,atol=.02,rtol=.01)
        passed('native_layer_preserved_semantics',model=model,layer_index=index,max_abs=error)
        # Equal Native contract with epilogue expressed outside projection.
        unfused=build_layer(c,length,batch_size=2,layer_index=index,fused_sites=(),interpret=True)
        close(unfused.prepared(x,unfused.prepare_weights(w)),actual,atol=.02,rtol=.01)
        if index==0:
            # All six sites custom, including partial-head and gate padding.
            cubic={site:choice('cubic',0,'outputs') for site in native_policy()}
            custom=build_layer(c,length,cubic,batch_size=2,layer_index=index,interpret=True)
            result=custom.prepared(x,custom.prepare_weights(w))
            error=close(result,actual,atol=.025,rtol=.015)
            passed('cubic_whole_layer_integration',model=model,max_abs=error)
        jax.clear_caches()

cfg=dict(search=dict(output_tiles=[[32,512],[128,1024]],native_mib=[None,48],estimate_prune_mib=112))
for spec,n in ((Epilogue('swiglu'),1536),(Epilogue('qk_norm_rope',head_dim=256),768),(Epilogue('norm_residual_add',norm='gemma'),2304)):
    s=dict(id='qualification',m=32,k=1024,n=n,output_dtype='bfloat16')
    candidates,trace=registry(s,cfg,spec,include_early=True)
    assert len({a['arm_id'] for a in candidates})==len(candidates)
    assert {a['implementation'] for a in candidates}>={'native','current','cubic','cubic_full'}
    assert {a['depth'] for a in candidates}=={0,1,2}
    assert all(row['reasons'] for row in trace['candidates'] if row['disposition']=='pruned')
    if spec.kind=='norm_residual_add':
        assert all(a['tile'][1]>=n for a in candidates if a['tile'])
    (out/(spec.kind+'-registry.json')).write_text(json.dumps(trace,indent=2)+'\n')
    passed('fused_registry',epilogue=spec.kind,offered=len(candidates),decisions=len(trace['candidates']))

invalid=[(choice(depth=2,mode='outputs'),(32,512,512),Epilogue('swiglu',early=True)),
         (choice(),(32,512,1024),Epilogue('norm_residual_add')),
         (choice(),(32,512,513),Epilogue('swiglu')),
         (choice(),(32,512,129),Epilogue('qk_norm_rope'))]
for a,shape,spec in invalid:
    try: make_projection(a,shape,spec,interpret=True)
    except ValueError: pass
    else: raise AssertionError('Invalid candidate was accepted')
passed('invalid_candidate_guards',cases=len(invalid))
(out/'summary.json').write_text(json.dumps(dict(completed=True,jax_version=jax.__version__,backend=jax.default_backend(),checks=len(checks),scope='CPU Pallas interpret correctness; no TPU timing or prediction-quality claim'),indent=2)+'\n')
