"""CPU exact algebra checks for outer-deferred depth two; TPU checks remain required."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_v6e_v002 import make_matmul


def main():
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    rng=np.random.default_rng(92371);rows=[]
    for order in ('standard','interleaved'):
        for shape in ((16,32,16),(19,67,21)):
            m,k,n=shape;a=rng.integers(-1,2,(m,k)).astype(np.float32);b=rng.integers(-1,2,(k,n)).astype(np.float32)
            fn=make_matmul(shape,(16,16,32),order=order,interpret=True,validate_tpu_alignment=False)
            actual=np.asarray(jax.jit(fn)(jnp.asarray(a,dtype=jnp.bfloat16),jnp.asarray(b,dtype=jnp.bfloat16)))
            np.testing.assert_array_equal(actual,a.astype(np.float64)@b.astype(np.float64))
            assert actual.dtype==np.float32
            row=dict(order=order,shape=shape,passed=True);rows.append(row);print(json.dumps(row),flush=True);jax.clear_caches()
    (out/'summary.json').write_text(json.dumps(dict(passed=True,tests=rows,scope='CPU integer algebra only.'),indent=2)+'\n')


if __name__=='__main__':main()
