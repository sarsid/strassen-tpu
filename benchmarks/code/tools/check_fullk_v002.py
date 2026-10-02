"""Exact algebra, FP32 regression, BF16 rounding, and frozen-plan validation."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_fullk_v002 import make_matmul
from strassen_mm.kernels_v6e_v003 import make_matmul as old
from strassen_mm.benchmark_fullk_v002 import groups, select

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
shape=(32,1024,512);rng=np.random.default_rng(924800)
a=rng.integers(-1,2,shape[:2]).astype(np.float32);a[:,np.arange(shape[1])%16!=0]=0
b=rng.integers(-1,2,shape[1:]).astype(np.float32);ref=a.astype(np.float64)@b.astype(np.float64)
ad,bd=jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16)
checks=[]
for dtype in ['float32','bfloat16']:
    for impl,depth in [('native',0),('cubic',0),('current',1),('current',2)]:
        for bk in ([None] if impl=='native' else [512,1024]):
            tile=None if bk is None else (32,512,bk)
            fn=make_matmul(shape,tile,impl,depth=depth,output_dtype=dtype,interpret=True)
            y=np.asarray(jax.jit(fn)(ad,bd))
            np.testing.assert_array_equal(y.astype(np.float32),ref)
            assert y.dtype==np.dtype(jnp.dtype(dtype))
            checks.append(dict(test='exact_integer',dtype=dtype,implementation=impl,depth=depth,bk=bk))
            print(checks[-1],flush=True);jax.clear_caches()
ad=jnp.asarray(rng.normal(size=shape[:2])/np.sqrt(shape[1]),jnp.bfloat16)
bd=jnp.asarray(rng.normal(size=shape[1:]),jnp.bfloat16)
for depth in (1,2):
    for bk in (512,1024):
        tile=(32,512,bk)
        baseline=jax.jit(old(shape,tile,depth,mode='deferred' if depth==1 else 'hybrid',interpret=True))(ad,bd)
        for dtype in ('float32','bfloat16'):
            fn=make_matmul(shape,tile,'current',depth=depth,output_dtype=dtype,interpret=True)
            result=jax.jit(fn)(ad,bd)
            np.testing.assert_array_equal(np.asarray(result),np.asarray(baseline.astype(dtype)))
            checks.append(dict(test='bitwise_old_or_final_cast',dtype=dtype,depth=depth,bk=bk))
            print(checks[-1],flush=True)
        jax.clear_caches()
cfg=json.loads(Path('configs/fullk_v002/campaign.json').read_text())
shapes=json.loads(Path('configs/fullk_v002/shapes.json').read_text())['shapes']
planned=groups(cfg,shapes,'screen');fake=[]
for g in planned:
    s=g['shape'];seen=set()
    for i,a in enumerate(g['arms']):
        assert a['arm_id'] not in seen;seen.add(a['arm_id'])
        if a['tile']:
            bm,bn,bk=a['tile'];assert s['m']%bm==s['n']%bn==s['k']%bk==0
            assert bm%32==bn%512==bk%512==0
        fake.append(dict(shape_id=s['id'],arm_id=a['arm_id'],eligible_for_speedup_claim=True,status='ok',
                         timing=dict(sample_count=9,mean_ms=100-i)))
chosen=select(fake,cfg,[g['shape'] for g in planned])
confirmed=groups(cfg,shapes,'confirm',chosen)
for g in confirmed:
    roles={role:a for a in g['arms'] for role in a['headline_roles']}
    for depth in (1,2):
        assert roles[f'full_s{depth}']['tile'][2]==g['shape']['k']
        assert roles[f'short_s{depth}']['tile'][2]<g['shape']['k']
    assert set(x['seed'] for x in g['inputs']).isdisjoint({cfg['screen_seed']+g['shape']['seed_offset']})
failed=[{**r,'status':'oom','eligible_for_speedup_claim':False} for r in fake]
assert all(v['status']=='no_eligible_candidate' for d in select(failed,cfg,[g['shape'] for g in planned]).values() for v in d.values())
assert len(planned)==len(confirmed)==24
try:make_matmul((33,1024,512),(32,512,512),'current',depth=1)
except ValueError:pass
else:raise AssertionError('Padding must be rejected')
summary=dict(completed=True,checks=checks,geometries=12,precision_groups=24,
    screen_cases=sum(len(g['arms']) for g in planned),smoke_cases=sum(len(g['arms']) for g in groups(cfg,shapes,'smoke')),
    confirmation_cases='data-dependent frozen selections; at most 864')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary,flush=True)
