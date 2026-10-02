"""Exact full-output checks for parent adapters and full/multiple K panels."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_fullk_v001 import make_matmul,parent_modules
from strassen_mm.benchmark_fullk_v001 import groups

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
shape=(32,512,256);rng=np.random.default_rng(92451)
a=rng.integers(-1,2,shape[:2]).astype(np.float32);a[:,np.arange(512)%16!=0]=0
b=rng.integers(-1,2,shape[1:]).astype(np.float32);ref=a.astype(np.float64)@b.astype(np.float64)
checks=[]
for dtype in ['float32','bfloat16']:
    specs=[('native',0),('parent_s1',1),('parent_cubic',0)]
    if dtype=='float32':specs += [('current',1),('current',2)]
    for impl,depth in specs:
        for bk in ([None] if impl=='native' else [256,512]):
            if depth==2 and bk==256:continue  # S2 requires BK a multiple of 512.
            tile=None if bk is None else (32,256 if depth!=2 else 512,bk)
            local_shape=shape if depth!=2 else (32,512,512)
            bb=b if depth!=2 else np.concatenate([b,b],axis=1)
            fn=make_matmul(local_shape,tile,impl,depth=depth,output_dtype=dtype,interpret=True)
            y=np.asarray(jax.jit(fn)(jnp.asarray(a,jnp.bfloat16),jnp.asarray(bb,jnp.bfloat16)))
            np.testing.assert_array_equal(y.astype(np.float32),a.astype(np.float64)@bb.astype(np.float64))
            assert y.dtype==np.dtype(jnp.dtype(dtype))
            checks.append(dict(implementation=impl,depth=depth,bk=bk,output_dtype=dtype,exact=True))
            print(json.dumps(checks[-1]),flush=True);jax.clear_caches()
try:make_matmul((33,512,256),(32,256,512),'parent_s1')
except ValueError:pass
else:raise AssertionError('Padding must be rejected')
cfg=json.loads(Path('configs/fullk_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/fullk_v001/shapes.json').read_text())['shapes']
planned=groups(cfg,shapes)
for group in planned:
    s=group['shape'];seen=set()
    for arm in group['arms']:
        assert arm['arm_id'] not in seen;seen.add(arm['arm_id'])
        if arm['tile']:
            bm,bn,bk=arm['tile'];assert s['m']%bm==s['n']%bn==s['k']%bk==0
            if arm['depth']==2:assert bm%32==bn%512==bk%512==0
    assert any(a['tile'] and a['tile'][2]==s['k'] for a in group['arms'])
assert len(planned)==5 and sum(len(g['arms']) for g in planned)==146
assert len(groups(cfg,shapes,True))==2
assert not parent_modules()[0].MANAGES_PROCESS_CEILING
(out/'summary.json').write_text(json.dumps(dict(completed=True,exact_checks=checks,groups=5,arms=146,measurement_cases=438),indent=2)+'\n')
