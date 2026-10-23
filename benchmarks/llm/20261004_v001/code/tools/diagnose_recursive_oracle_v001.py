"""Locate CPU interpreter versus independent BF16 oracle discrepancy."""
import os,json
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np,ml_dtypes
from check_recursive_v002 import reference
from strassen_mm.kernels_recursive_v001 import make_matmul
from strassen_mm.kernels_two_level_v001 import quarters,operands,ORDER,TARGETS

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
rng=np.random.default_rng(44)
a=rng.normal(size=(32,32)).astype(ml_dtypes.bfloat16)
b=rng.normal(size=(32,32)).astype(ml_dtypes.bfloat16)
ja,jb=jnp.asarray(a),jnp.asarray(b)
direct=np.zeros((32,32),np.float32)
for p in ORDER:
 left,right=operands(quarters(ja),quarters(jb),p)
 product=np.asarray(left,dtype=np.float32)@np.asarray(right,dtype=np.float32)
 for q,sign in TARGETS[p]:
  row,col=divmod(q,2);direct[row*16:(row+1)*16,col*16:(col+1)*16]+=sign*product
fn=make_matmul((32,32,32),(32,32,32),1,interpret=True,validate_tpu_alignment=False)
actual=np.asarray(jax.jit(fn)(ja,jb));oracle=reference(a,b,1);exact=a.astype(np.float64)@b.astype(np.float64)
result={}
for label,x,y in [('actual_oracle',actual,oracle),('direct_oracle',direct,oracle),('actual_direct',actual,direct),('actual_exact',actual,exact),('oracle_exact',oracle,exact)]:
 result[label]=dict(max_abs=float(np.abs(x-y).max()),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)))
print(json.dumps(result));(out/'summary.json').write_text(json.dumps(result,indent=2))
