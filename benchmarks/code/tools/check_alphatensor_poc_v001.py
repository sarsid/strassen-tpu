"""CPU correctness checks only; timings are intentionally not collected."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.alphatensor_adapter_v001 import load, make_matmul


def main():
    output=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';output.mkdir()
    records=[]
    rng=np.random.default_rng(29)
    for mode in [False,True]:
        factors,_=load(mode)
        assert factors.shape==(3,16,49)
        for shape in [(8,12,16),(12,8,20)]:
            m,k,n=shape
            a=rng.integers(-2,3,(m,k)).astype(np.float32)
            b=rng.integers(-2,3,(k,n)).astype(np.float32)
            fn=jax.jit(make_matmul(shape,fp32_recombine=mode))
            out=fn(jnp.asarray(a,dtype=jnp.float32),jnp.asarray(b,dtype=jnp.float32))
            np.testing.assert_array_equal(np.asarray(out),a@b)
            bf=fn(jnp.asarray(a,dtype=jnp.bfloat16),jnp.asarray(b,dtype=jnp.bfloat16))
            assert bf.dtype==(jnp.float32 if mode else jnp.bfloat16),bf.dtype
            # Small integer operands are exactly representable throughout this fixture.
            np.testing.assert_array_equal(np.asarray(bf,dtype=np.float32),a@b)
            records.append({'shape_mkn':shape,'fp32_recombine':mode,'integer_correct':True,'bf16_output_dtype':str(bf.dtype)})
    root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
    campaign=json.loads((root/'configs/alphatensor_poc_v001/campaign.json').read_text())
    shapes=json.loads((root/'configs/alphatensor_poc_v001/shapes.json').read_text())['shapes']
    assert len(shapes)==4 and len(campaign['arms'])==6
    assert all(all(s[d]%4==0 for d in ('m','k','n')) for s in shapes)
    summary={'passed':True,'tensor_identity_exact':True,'rectangular_fixtures':records,'planned_shapes':4,'planned_arms':6,
             'jax':jax.__version__,'backend':jax.default_backend(),'scope':'CPU algebra and adapter checks only; not TPU speed'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
