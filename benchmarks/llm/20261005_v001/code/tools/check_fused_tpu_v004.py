"""Bounded hardware compilation/correctness qualification; no speed claims."""
import argparse, dataclasses, importlib.metadata, json, os, time, traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.fusion_v003 import Epilogue
from strassen_mm.fusion_v003 import reference
from strassen_mm.kernels_fused_v004 import make_projection
from strassen_mm.tuner_joint_v001 import arm
from fusion_oracle_v001 import reference as ideal_reference
from fusion_oracle_v002 import reference as host_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','libtpu')}
    identity=dict(versions=versions,devices=[dict(kind=d.device_kind,id=d.id) for d in jax.devices()],backend=jax.default_backend())
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    assert versions==dict(jax='0.11.2',jaxlib='0.11.2',libtpu='0.0.48'),identity
    assert identity['backend']=='tpu' and len(jax.devices())==1 and jax.devices()[0].device_kind=='TPU v6 lite'
    rng=np.random.default_rng(1002004);results=[]
    cases=[]
    for dtype in ('float32','bfloat16'):
        for impl,depth,mode in [('native',0,'products'),('cubic',0,'outputs'),('cubic_full',0,'outputs'),('current',1,'products'),('current',1,'outputs'),('current',2,'products'),('current',2,'outputs')]:
            for kind in ('swiglu','qk_norm_rope','residual_add'):
                cases.append((impl,depth,mode,dtype,2,Epilogue(kind)))
        for spec in (Epilogue('geglu'),Epilogue('norm_residual_add',norm='gemma'),
                     Epilogue('rope',head_dim=256),Epilogue('qk_norm_rope',norm='gemma',head_dim=256),
                     Epilogue('swiglu',early=True),Epilogue('swiglu',rounding='accumulator')):
            cases.append(('current',1,'outputs',dtype,1,spec))
    for index,(impl,depth,mode,dtype,buffers,spec) in enumerate(cases):
        started=time.monotonic()
        shape=(33,513,768 if spec.rotary else 258 if spec.kind=='norm_residual_add' else 770)
        m,k,n=shape
        a=dict(arm((32,512,512),depth,mode,buffers,dtype),architecture='v6e',implementation=impl)
        row=dict(index=index,arm=a,epilogue=dataclasses.asdict(spec),shape_mkn=list(shape))
        try:
            aa=rng.integers(-1,2,size=(m,k)).astype(np.float32);aa[:,np.arange(k)%64!=0]=0
            bb=rng.integers(-1,2,size=(k,n)).astype(np.float32)
            ad,bd=jnp.asarray(aa,jnp.bfloat16),jnp.asarray(bb,jnp.bfloat16)
            freqs=jnp.arange(m,dtype=jnp.float32)[:,None]/10000.**(jnp.arange(0,spec.head_dim,2,dtype=jnp.float32)/spec.head_dim)[None,:]
            operands=dict(residual=jnp.asarray(rng.normal(size=(m,n)),jnp.bfloat16),
                          scale=jnp.asarray(rng.normal(size=spec.head_dim if spec.rotary else n)*.1+(0 if spec.norm=='gemma' else 1),jnp.bfloat16),
                          cos=jnp.cos(freqs).astype(jnp.bfloat16),sin=jnp.sin(freqs).astype(jnp.bfloat16))
            aux={key:operands[key] for key in spec.auxiliaries}
            exact=jnp.asarray(aa.astype(np.float64)@bb.astype(np.float64),jnp.float32)
            expected=host_reference(np.asarray(exact),spec,dtype,default_segmented=impl!='native' and spec.kind=='qk_norm_rope',**aux)
            ideal=ideal_reference(np.asarray(exact),spec,dtype,**aux)
            fn=make_projection(a,shape,spec);row['metadata']=fn.metadata
            packed=fn.prepare_weights(bd)
            result=jax.jit(fn.prepared)(ad,packed,**aux).block_until_ready()
            actual,ref=np.asarray(result,dtype=np.float32),np.asarray(expected,dtype=np.float32)
            row['ideal_fp32_norm_relative_l2']=float(np.linalg.norm(actual-np.asarray(ideal,dtype=np.float32))/max(np.linalg.norm(np.asarray(ideal,dtype=np.float32)),1e-30))
            row.update(finite=bool(np.isfinite(actual).all()),max_abs=float(abs(actual-ref).max()),
                       relative_l2=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)))
            np.testing.assert_allclose(actual,ref,atol=.004 if dtype=='bfloat16' else 1e-4,rtol=.004 if dtype=='bfloat16' else 1e-4)
            assert row['finite'] and result.dtype==jnp.dtype(dtype)
            row['status']='passed'
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        row['compile_and_check_seconds']=time.monotonic()-started
        results.append(row)
        (out/'cases.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('metadata','traceback','arm')})[:2000],flush=True)
        jax.clear_caches()
    summary=dict(completed=True,cases=len(results),passed=sum(r['status']=='passed' for r in results),
                 failed=sum(r['status']!='passed' for r in results),identity=identity,
                 scope='v6e DEFAULT arithmetic-contract qualification against independent BF16 dot-operand oracle; ideal-FP32 error retained separately; no timing or real-LLM quality')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return int(summary['failed']>0)

if __name__=='__main__':raise SystemExit(main())
