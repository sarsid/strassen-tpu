"""Verify new Pallas algebra, sequential-K accumulation and padding on CPU."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_two_level_v001 import make_matmul


def main():
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    rng=np.random.default_rng(739);records=[]
    for shape,distribution in [((32,512,512),'integer'),((35,520,516),'integer'),((32,512,512),'gaussian')]:
        m,k,n=shape
        if distribution=='integer':
            a=rng.integers(-1,2,(m,k)).astype(np.float32);b=rng.integers(-1,2,(k,n)).astype(np.float32)
        else:
            a=rng.standard_normal((m,k),dtype=np.float32)/np.sqrt(k);b=rng.standard_normal((k,n),dtype=np.float32)
        a=jnp.asarray(a,dtype=jnp.bfloat16);b=jnp.asarray(b,dtype=jnp.bfloat16)
        fn=make_matmul(shape,(32,512,512),interpret=True)
        got=np.asarray(jax.jit(fn)(a,b));ref=np.asarray(a,dtype=np.float64)@np.asarray(b,dtype=np.float64)
        assert got.dtype==np.float32 and got.shape==(m,n) and np.isfinite(got).all()
        rel=float(np.linalg.norm(got-ref)/np.linalg.norm(ref))
        if distribution=='integer':np.testing.assert_array_equal(got,ref)
        else:assert rel<.02,rel
        records.append({'shape_mkn':shape,'distribution':distribution,'relative_l2':rel,'passed':True})
        print(json.dumps(records[-1]),flush=True)
        jax.clear_caches()
    summary={'passed':True,'tests':records,'backend':jax.default_backend(),'scope':'CPU Pallas interpretation, not performance'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
