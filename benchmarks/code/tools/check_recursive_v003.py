"""CPU algebra checks; compiled TPU separately measures BF16 rounding.

Preserved diagnostic recursive-oracle-diagnostic-v001 showed that the CPU
Pallas interpreter removes BF16 intermediate rounding: its relative error
against FP64 was 8e-8, versus 0.0044 against explicit BF16 arithmetic. CPU
interpretation therefore checks algebra, padding and K accumulation only.
"""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from strassen_mm.kernels_recursive_v001 import make_matmul


def reference(a, b, depth):
    if depth == 0:
        return a.astype(np.float32) @ b.astype(np.float32)
    def split(x):
        r,c = np.array_split(x,2,0)
        return (*np.array_split(r,2,1),*np.array_split(c,2,1))
    left,right = split(a),split(b)
    a,b,c,d = left
    e,f,g,h = right
    def q(x):return x.astype(ml_dtypes.bfloat16)
    p1=reference(q(a+d),q(e+h),depth-1)
    p2=reference(q(c+d),e,depth-1)
    p3=reference(a,q(f-h),depth-1)
    p4=reference(d,q(g-e),depth-1)
    p5=reference(q(a+b),h,depth-1)
    p6=reference(q(c-a),q(e+f),depth-1)
    p7=reference(q(b-d),q(g+h),depth-1)
    return np.block([[p1+p4-p5+p7,p3+p5],[p2+p4,p1-p2+p3+p6]])


def main():
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    rng=np.random.default_rng(923);records=[]
    for depth in (1,2,3,4):
        for distribution in ('integer','gaussian'):
            tile=(32,32,32);shape=(35,67,37) if distribution=='integer' else (32,32,32)
            m,k,n=shape
            a=(rng.integers(-1,2,(m,k)) if distribution=='integer' else rng.normal(size=(m,k))).astype(ml_dtypes.bfloat16)
            b=(rng.integers(-1,2,(k,n)) if distribution=='integer' else rng.normal(size=(k,n))).astype(ml_dtypes.bfloat16)
            fn=make_matmul(shape,tile,depth,interpret=True,validate_tpu_alignment=False)
            actual=np.asarray(jax.jit(fn)(jnp.asarray(a),jnp.asarray(b)))
            exact=a.astype(np.float64)@b.astype(np.float64)
            assert actual.shape==(m,n) and actual.dtype==np.float32 and np.isfinite(actual).all()
            if distribution=='integer':np.testing.assert_array_equal(actual,exact)
            else:
                assert np.linalg.norm(actual-exact)/np.linalg.norm(exact)<1e-4
                assert np.abs(actual-exact).max()<1e-4*np.abs(exact).max()+1e-5
            row=dict(depth=depth,distribution=distribution,shape_mkn=shape,
                     relative_l2=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact)),passed=True)
            records.append(row);print(json.dumps(row),flush=True);jax.clear_caches()
    (out/'summary.json').write_text(json.dumps(dict(passed=True,tests=records,
        scope='Small CPU interpreted tiles; exact integer algebra with padding and multiple K panels, FP64 floating algebra. CPU interpretation does not preserve BF16 intermediate rounding. TPU alignment and compiled arithmetic checked separately.'),indent=2)+'\n')


if __name__=='__main__':main()
