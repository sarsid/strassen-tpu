"""Independent full-output algebra checks for split/cross terms and edge cases."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_padded_v001 import make_matmul
from strassen_mm.benchmark_padding_v001 import groups

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
checks=[];rng=np.random.default_rng(92426)
for depth in (1,2):
    for policy in ('edge_tiles','native_fringe'):
        # All-axes below/above, individual axes, exact core and no-full-core fallback.
        for shape in [(16,32,16),(15,31,15),(17,33,17),(17,32,16),(16,33,16),(16,32,17),(3,5,7)]:
            m,k,n=shape
            a=rng.integers(-1,2,(m,k)).astype(np.float32)
            b=rng.integers(-1,2,(k,n)).astype(np.float32)
            fn=make_matmul(shape,(16,16,16),depth,policy=policy,interpret=True,validate_tpu_alignment=False)
            y=np.asarray(jax.jit(fn)(jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16)))
            np.testing.assert_array_equal(y,a.astype(np.float64)@b.astype(np.float64))
            assert y.shape==(m,n) and y.dtype==np.float32
            volumes=fn.metadata['strassen_useful_volume_fraction']+fn.metadata['native_useful_volume_fraction']
            assert abs(volumes-1)<1e-12
            checks.append(dict(depth=depth,policy=policy,shape=shape,exact=True))
            print(json.dumps(checks[-1]),flush=True)
            jax.clear_caches()
cfg=json.loads(Path('configs/padding_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/padding_v001/shapes.json').read_text())['shapes']
planned=groups(cfg,shapes)
assert len(planned)==13 and len({(s['m'],s['k'],s['n']) for s in shapes})==13
assert sum(len(g['arms'])*len(g['inputs']) for g in planned)==273
for g in planned:
    for depth in (1,2):
        arms=[a for a in g['arms'] if a.get('depth')==depth]
        assert len(arms)==3 and len({tuple(a['tile']) for a in arms})==1
        assert len({a['mode'] for a in arms})==1
(out/'summary.json').write_text(json.dumps(dict(completed=True,exact_checks=checks,shapes=13,measurement_cases=273),indent=2)+'\n')
