"""Real Gemma BF16 rounding-boundary ablation; no speed claims."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import benchmark_v001 as base


def main():
    p=argparse.ArgumentParser()
    for name in ('campaign','output-dir','expected-identity','model-input','private-root'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=3600);a=p.parse_args()
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False);journal=base.Journal(out,a.phase)
    error=None;result=None;deadline=time.monotonic()+a.max_wall_seconds
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Gemma diagnosis wall budget exhausted')
    try:
        source=Path(__file__).resolve().parents[2];sys.path.insert(0,str(source/'tools'))
        from .benchmark_large_prepare_v003 import host_identity
        expected=json.loads(a.expected_identity.read_text());expected=expected.get('identity',expected)
        host=host_identity(a.allocation_id,expected)
        for key in host:
            if host[key]!=expected[key]:raise ValueError('CPU capture identity mismatch: '+key)
        from install_mlsys_model_tools_v001 import install_environment
        cmd=[str(a.private_root/'cpu-tools-v002/bin/python'),str(source/'tools/capture_gemma_intermediates_v001.py'),
             '--model-input',str(a.model_input),'--output-dir',str(out/'official')]
        base.exclusive_json(out/'capture-command.json',dict(argv=cmd))
        journal.emit('preparation_step',step='capture_official_gemma_intermediates')
        with (out/'capture.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=install_environment(),check=True,timeout=min(900,a.max_wall_seconds-60))
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from . import model_gemma_v002 as gm, model_composed_v002 as composed, model_n9_v001 as qm
        from . import benchmark_llm_tradeoff_v001 as runner, benchmark_large_models_v001 as prior
        from .benchmark_n9_v001 import metrics,eligible,QUALIFICATION
        from .kernels_v002 import make_matmul
        original = {name:getattr(gm,name) for name in ('rope','rms','dot','gelu')}
        def rounded_rope(x,theta,factor=1.):
            length,_,dim=x.shape
            inverse=1./(float(theta)**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))/float(factor)
            frequency=jnp.arange(length,dtype=jnp.float32)[:,None]*inverse[None,:]
            angle=jnp.concatenate((frequency,frequency),-1)[:,None,:]
            co=jax.lax.optimization_barrier(jnp.cos(angle).astype(x.dtype))
            si=jax.lax.optimization_barrier(jnp.sin(angle).astype(x.dtype))
            def rounded(v):
                return jax.lax.optimization_barrier(jax.lax.reduce_precision(v,8,7).astype(x.dtype))
            left=rounded(x.astype(jnp.float32)*co.astype(jnp.float32))
            right=rounded(gm.rotate_half(x).astype(jnp.float32)*si.astype(jnp.float32))
            return jax.lax.optimization_barrier((left+right).astype(x.dtype))
        def bounded(fn):
            def call(x,*rest):
                return jax.lax.optimization_barrier(fn(jax.lax.optimization_barrier(x),*rest))
            return call
        def configure(name):
            for key,fn in original.items():setattr(gm,key,fn)
            if name!='original':gm.rope=rounded_rope
            if name=='all_boundaries':
                for key in ('rms','dot','gelu'):setattr(gm,key,bounded(original[key]))
        for module in (base,runner,prior):module.jax,module.jnp,module.np,module.ml_dtypes=jax,jnp,np,ml_dtypes
        for module in (runner,prior):module.gm,module.qm,module.composed=gm,qm,composed
        cfg=json.loads(a.campaign.read_text());runner.verify_environment(a,cfg)
        bundle=json.loads(a.model_input.read_text())
        class NativeStudy(runner.Study):
            def mm(self,arm,shape):return jax.jit(make_matmul('native',shape))
        study=NativeStudy(a,cfg,journal);cp,ids,ref=study.load(bundle)
        official=np.load(out/'official/official.npz',allow_pickle=False)
        meta=json.loads((out/'official/official.json').read_text());rows=[]
        def metric(r,v):return prior.vector_metrics(r,np.asarray(jax.device_get(v),dtype=np.float32))
        def compare(name,r,fn,*values):
            check()
            with jax.disable_jit(True):eager=fn(*values);jax.block_until_ready(eager)
            with jax.disable_jit(False):
                executable=jax.jit(fn).lower(*values).compile()
                compiled=executable(*values);jax.block_until_ready(compiled)
            stem=name.replace('/','_')
            (out/(stem+'.hlo.txt')).write_text(executable.as_text())
            np.savez_compressed(out/(stem+'.npz'),reference=np.asarray(r,dtype=np.float32),
                eager=np.asarray(jax.device_get(eager),dtype=np.float32),
                jit=np.asarray(jax.device_get(compiled),dtype=np.float32))
            row=dict(name=name,eager=metric(r,eager),jit=metric(r,compiled))
            rows.append(row);journal.emit('primitive_diagnostic',**row)
        def array(x):return jnp.asarray(x,dtype=jnp.bfloat16)
        policy={k:dict(algorithm='native',candidate_id='native_default',compiler_options={},family='native',tile=None,variant='plain') for k in ('gateup','down')}
        embedding=jax.device_put(cp.embeddings(ids[0,:64]));embedding_error=metric(official['embedding_output'][0],embedding)
        states={name:embedding for name in ('original','rope_boundaries','all_boundaries')}
        functions={}
        journal.emit('embedding_diagnostic',**embedding_error,official_scale=meta['embedding_scale'])
        layers=[]
        for i in range(cp.config['num_hidden_layers']):
            check();weights=jax.device_put(cp.layer(i));jax.block_until_ready(weights)
            refin=array(official[f'layer_{i:03}_input'][0]);refout=official[f'layer_{i:03}_output'][0]
            row=dict(layer=i,attention_type=cp.config['layer_types'][i],common={},propagated={})
            for name in states:
                configure(name)
                key=(name,cp.config['layer_types'][i])
                if key not in functions:
                    h,inner=cp.config['hidden_size'],cp.config['intermediate_size']
                    functions[key]=composed.Layer(composed.prefix(cp.config,64,i),
                        jax.jit(make_matmul('native',(64,h,2*inner))),composed.activation(cp.config),
                        jax.jit(make_matmul('native',(64,inner,h))),composed.suffix(cp.config))
                fun=functions[key]
                local=fun(refin,weights);jax.block_until_ready(local)
                value=fun(states[name],weights);jax.block_until_ready(value)
                row['common'][name]=metric(refout,local);row['propagated'][name]=metric(refout,value);states[name]=value
                del local
            layers.append(row);journal.emit('layer_diagnostic',**row)
            if i in meta['selected_layers']:
                local=cp.config['layer_types'][i]=='sliding_attention'
                theta=cp.config['rope_local_base_freq'] if local else cp.config['rope_theta'];factor=1. if local else cp.config['rope_scaling']['factor']
                for op in ('q','k'):
                    x=array(official[f'rope_{i:03}_{op}_input'][0].transpose(1,0,2));r=official[f'rope_{i:03}_{op}_output'][0].transpose(1,0,2)
                    compare(f'{i}/original_rope_{op}',r,lambda x:original['rope'](x,theta,factor),x)
                    compare(f'{i}/corrected_rope_{op}',r,lambda x:rounded_rope(x,theta,factor),x)
            del weights,refin;gc.collect()
        configure('original')
        norm=jax.device_put(np.array(cp.tensor('model.norm.weight'),copy=True));head=jax.device_put(cp.head())
        final={}
        for name,state in states.items():
            with jax.disable_jit(False):
                hidden=gm.rms(state,norm,cp.config['rms_norm_eps']);logits=gm.head_logits(state,norm,head,cp.config);jax.block_until_ready((hidden,logits))
            raw={k:v.item() for k,v in jax.device_get(qm.compare_logits(jax.device_put(official['logits'][0,:63]),logits[:63],jax.device_put(ids[0,1:64]))).items()}
            row=metrics(raw);row['hidden']=metric(official['final_hidden'][0],hidden)
            row['passed']=bool(eligible(row,QUALIFICATION) and row['hidden']['relative_l2']<=.02)
            final[name]=row;journal.emit('final_diagnostic',variant=name,**row)
        result=dict(embedding=embedding_error,layers=layers,primitives=rows,final=final,official_metadata=meta,
                    qualification_gates=QUALIFICATION,hidden_l2_limit=.02,scope='Real 12B checkpoint rounding ablation; no timings or Strassen quality claim')
        base.exclusive_json(out/'diagnosis.json',base.json_safe(result))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('diagnostic_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None,error=error,diagnosis_available=result is not None)
    base.exclusive_json(out/'summary.json',summary);journal.emit('run_complete',**summary);journal.close()
    return 0 if error is None else 1

if __name__=='__main__':raise SystemExit(main())
