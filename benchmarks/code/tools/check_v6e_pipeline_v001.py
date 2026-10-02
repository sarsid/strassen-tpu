"""Exact CPU algebra across panel boundaries, traversal, padding and depths."""
import os,json
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from strassen_mm.kernels_v6e_v003 import make_matmul
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
rows=[]
for depth,mode in [(1,'panel'),(1,'deferred'),(2,'panel'),(2,'hybrid')]:
 for traversal in ('mn','nm'):
  shape=(48,96,48);tile=(32,32,32);rng=np.random.default_rng(338)
  a=rng.integers(-1,2,shape[:2]).astype(np.float32);b=rng.integers(-1,2,shape[1:]).astype(np.float32)
  fn=make_matmul(shape,tile,depth,mode=mode,traversal=traversal,interpret=True,validate_tpu_alignment=False)
  y=np.asarray(jax.jit(fn)(jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16)))
  err=float(np.max(np.abs(y-a@b)));assert err==0,(depth,mode,traversal,err)
  rows.append(dict(depth=depth,mode=mode,traversal=traversal,max_abs=err))
(out/'summary.json').write_text(json.dumps(dict(completed=True,checks=rows),indent=2)+'\n');print('8 exact checks passed')
