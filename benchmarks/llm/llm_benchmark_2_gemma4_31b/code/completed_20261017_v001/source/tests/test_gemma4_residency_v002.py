"""Bit-preserving host parking and unchanged timed computation contracts."""
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
import unittest
import weakref
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARMS=('native','native_tuned','cubic_tuned','strassen_tuned')


def load(version):
    module=ast.parse((ROOT/f'tools/gemma4_batch_stage_{version}.py').read_text())
    nodes=[n for n in module.body if isinstance(n,ast.FunctionDef)]
    ns=dict(jax=jax,jnp=jnp,np=np,gc=gc,hashlib=hashlib,math=math,statistics=statistics,
            time=time,Path=Path,json=json,ARMS=ARMS,REPS=(0,5))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),version,'exec'),ns)
    return ns


def digest(v):return hashlib.sha256(np.ascontiguousarray(jax.device_get(v)).view(np.uint8)).hexdigest()


def fixture(ns, output):
    model=types.SimpleNamespace(config={'hidden_size':8},Checkpoint=lambda *args:None)
    model.geometry=lambda config,i:dict(layer_type='full_attention' if i%6==5 else 'sliding_attention')
    model.load_layer=lambda checkpoint,i,*args:{'scale':jnp.asarray((i+1)/1024,jnp.bfloat16)}
    model.initial_hidden=lambda checkpoint,ids:jnp.asarray(np.arange(ids.size*8).reshape(-1,8)/64,jnp.bfloat16)
    model.arm_options=lambda *args:{}
    model.projection=lambda site,arm,*args:lambda x,p,residual:((x+p['scale']) if residual is None else (x+residual+p['scale'])).astype(jnp.bfloat16)
    ns['model']=model
    ns['tokens']=lambda batch,sequence,emit,natural=False,segments=1:np.arange(batch*sequence*(segments if natural else 1),dtype=np.int32).reshape((segments,batch,sequence) if natural else (batch,sequence))
    def block_exes(*args):
        return {arm:jax.jit(lambda x,p,offset=i:(x+p['scale']+offset/512).astype(jnp.bfloat16)) for i,arm in enumerate(ARMS)}
    ns['block_exes']=block_exes
    ns['errors']=lambda outputs:dict(finite=True,hashes={k:digest(v) for k,v in outputs.items()})
    ns['bootstrap']=lambda values:{'sum_ms':sum(values)}
    ns['task_metrics']=lambda hidden,*args:{arm:dict(passes=True,hashes=[digest(v) for v in values]) for arm,values in hidden.items()}
    ns['compile_fn']=lambda fn,*args,**kwargs:jax.jit(fn)
    ns['captures']=lambda x,*args:({'q':x,'down':(x+1).astype(jnp.bfloat16)},{'down':x},x)
    args=types.SimpleNamespace(batch=2,sequence=4,cache=Path('unused'),output=output)
    plan=dict(layer_warmups=1,layer_runs=2,block_warmups=1,block_runs=2,
              projection_warmups=1,projection_runs=2,quality_segments=3,
              repetitive_quality_gates={},natural_quality_gates={})
    profiles={'sliding_attention':{},'full_attention':{}}
    return args,plan,profiles


class Checks(unittest.TestCase):
    def test_scientific_and_timing_contracts_unchanged(self):
        def functions(v):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/f'tools/gemma4_batch_stage_{v}.py').read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=functions('v004'),functions('v006')
        for name in old:
            if name not in ('projection_confirm','confirm','quality'):self.assertEqual(old[name],new[name],name)

    def test_confirmation_states_round_trip_exactly(self):
        summaries=[]
        for version in ('v004','v006'):
            with tempfile.TemporaryDirectory() as tmp:
                ns=load(version);args,plan,profiles=fixture(ns,Path(tmp))
                ns['projection_confirm']=lambda *args:None
                ns['confirm'](args,plan,profiles,lambda row:None)
                summaries.append(json.loads((Path(tmp)/'summary.json').read_text())['repetitive_quality'])
        self.assertEqual(summaries[0],summaries[1])

    def test_natural_quality_states_round_trip_exactly(self):
        summaries=[]
        for version in ('v004','v006'):
            with tempfile.TemporaryDirectory() as tmp:
                ns=load(version);args,plan,profiles=fixture(ns,Path(tmp))
                ns['quality'](args,plan,profiles,lambda row:None)
                summaries.append(json.loads((Path(tmp)/'summary.json').read_text()))
        self.assertEqual(summaries[0],summaries[1])

    def test_projection_inputs_and_final_outputs_unchanged(self):
        outputs=[]
        for version in ('v004','v006'):
            ns=load(version);args,plan,_=fixture(ns,Path('unused'));events=[]
            x=np.arange(64,dtype=np.float32).reshape(8,8).astype(np.float32)
            ns['projection_confirm'](x,{'scale':jnp.asarray(.125,jnp.bfloat16)},{},5,args,plan,events.append)
            outputs.append([(r['site'],r['hashes']) for r in events if r['kind']=='projection_confirmation'])
        self.assertEqual(outputs[0],outputs[1])

    def test_projection_retains_no_device_outputs_between_calls(self):
        old=load('v004');new=load('v006');old_events=[];events=[];live=[]
        class Buffer:
            def __init__(self,value):self.value=value
        class FakeJax:
            @staticmethod
            def block_until_ready(v):return v
            @staticmethod
            def device_get(v):return np.array(v.value,copy=True)
        def execute(value):
            self.assertFalse(any(ref() is not None for ref in live))
            result=Buffer(value);live.append(weakref.ref(result));return result
        args={arm:() for arm in ARMS}
        old['round_robin']({arm:(lambda value=i:jnp.asarray(value)) for i,arm in enumerate(ARMS)},args,2,3,old_events.append)
        new['jax']=FakeJax
        samples,outputs=new['projection_round_robin']({arm:(lambda value=i:execute(value)) for i,arm in enumerate(ARMS)},args,2,3,events.append)
        self.assertFalse(any(ref() is not None for ref in live))
        self.assertEqual([(e['arm'],e['round']) for e in old_events],[(e['arm'],e['round']) for e in events])
        self.assertTrue(all(len(v)==3 for v in samples.values()))
        self.assertEqual([int(outputs[arm]) for arm in ARMS],list(range(4)))


if __name__=='__main__':unittest.main()
