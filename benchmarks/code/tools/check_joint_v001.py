"""Validate new accumulator arithmetic, coverage, selection and fallback rules."""
import copy
import json
import os
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm.kernels_joint_v001 import make_matmul
from strassen_mm.kernels_fullk_v002 import make_matmul as old
from strassen_mm.kernels_fullk_v001 import make_matmul as parent
from strassen_mm.tuner_joint_v001 import groups, registry, select, expanded
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_joint_v001 import recommendation


out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
checks=[]
def passed(name,**values):
    record=dict(test=name,**values);checks.append(record);print(record,flush=True)
shape=(32,1536,512);rng=np.random.default_rng(9260000)
a=rng.integers(-1,2,shape[:2]).astype(np.float32);a[:,np.arange(shape[1])%16!=0]=0
b=rng.integers(-1,2,shape[1:]).astype(np.float32)
ad,bd=jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16)
ref=a.astype(np.float64)@b.astype(np.float64)
for dtype in ('float32','bfloat16'):
    for depth in (1,2):
        for mode in ('products','outputs'):
            for bk in (512,1536):
                fn=make_matmul(shape,(32,512,bk),depth=depth,accumulator=mode,output_dtype=dtype,interpret=True)
                y=np.asarray(jax.jit(fn)(ad,bd))
                np.testing.assert_array_equal(y.astype(np.float32),ref)
                assert y.dtype==np.dtype(jnp.dtype(dtype))
                passed('exact_integer',dtype=dtype,depth=depth,mode=mode,bk=bk)
                jax.clear_caches()
ad=jnp.asarray(rng.normal(size=shape[:2])/np.sqrt(shape[1]),jnp.bfloat16)
bd=jnp.asarray(rng.normal(size=shape[1:]),jnp.bfloat16)
for depth in (1,2):
    for bk in (512,1536):
        tile=(32,512,bk)
        for mode in ('products','outputs'):
            fp=jax.jit(make_matmul(shape,tile,depth=depth,accumulator=mode,interpret=True))(ad,bd)
            bf=jax.jit(make_matmul(shape,tile,depth=depth,accumulator=mode,output_dtype='bfloat16',interpret=True))(ad,bd)
            np.testing.assert_array_equal(np.asarray(bf),np.asarray(fp.astype(jnp.bfloat16)))
            passed('one_final_bf16_round',depth=depth,mode=mode,bk=bk)
            if mode=='products':
                baseline=jax.jit(old(shape,tile,'current',depth=depth,interpret=True))(ad,bd)
                np.testing.assert_array_equal(np.asarray(fp),np.asarray(baseline))
                passed('unchanged_products_regression',depth=depth,bk=bk)
            elif depth==1:
                for dtype,actual in [('float32',fp),('bfloat16',bf)]:
                    reference=jax.jit(parent(shape,tile,'parent_s1',output_dtype=dtype,interpret=True))(ad,bd)
                    np.testing.assert_array_equal(np.asarray(actual),np.asarray(reference))
                    passed('parent_s1_bitwise',dtype=dtype,bk=bk)
        jax.clear_caches()
cfg=json.loads(Path('configs/joint_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/joint_v001/shapes.json').read_text())['shapes']
planned=groups(cfg,shapes,'screen');fake=[]
for batch,g in enumerate(planned):
    scale=1+batch%7
    for a in g['arms']:
        # Stable ratios despite different absolute speed in each batch.
        ratio=1.0 if not a['depth'] else .9+.03*a['depth']+(.01 if a['accumulator']=='products' else 0)
        fake.append(dict(shape_id=g['shape']['id'],group_id=g['group_id'],arm_id=a['arm_id'],status='ok',
            eligible_for_speedup_claim=True,timing=dict(sample_count=cfg['timing']['screen']['repeats'],mean_ms=ratio*scale),
            correctness=dict(relative_l2=.004,finite=True,pass_gate=True)))
selected=select(fake,cfg,expanded(shapes))
for s in expanded(shapes):
    offered,trace=registry(s,cfg);ids={a['arm_id'] for a in offered}
    assert len(ids)==len(offered)
    assert selected[s['id']]['choices']['overall']['candidate']['depth']==1
    assert selected[s['id']]['choices']['overall']['candidate']['accumulator']=='outputs'
    assert abs(selected[s['id']]['choices']['overall']['score']-.93)<1e-12
    for c in s['historical_controls'][s['output_dtype']]:
        assert any(a['depth']==c['depth'] and a['tile']==c['tile'] and a['buffers']==c['buffers'] and a['accumulator']=='products' for a in offered)
    for a in offered:
        if a['tile']:
            bm,bn,bk=a['tile'];assert s['m']%bm==s['n']%bn==s['k']%bk==0
    assert all(d['reasons'] for d in trace['candidates'] if d['disposition']=='pruned')
passed('registry_and_anchor_normalization',precision_groups=len(selected),distinct_candidates=sum(len(registry(s,cfg)[0]) for s in expanded(shapes)))
confirmed=groups(cfg,shapes,'confirm',selected)
assert len(confirmed)==12
for g in confirmed:
    assert {i['seed'] for i in g['inputs']}.isdisjoint({cfg['screen_seed']+g['shape']['seed_offset']})
    roles={role for a in g['arms'] for role in a['headline_roles']}
    assert {'s1_other_accumulator','s2_other_accumulator','s1_other_buffers','s2_other_buffers','overall','native'}<=roles
passed('fresh_confirmation_and_counterfactuals')
invalid=[{**r,'status':'numerical_failure','eligible_for_speedup_claim':False} if r['arm_id'].startswith('S') else r for r in fake]
assert all(v['choices']['overall']['candidate']['depth']==0 for v in select(invalid,cfg,expanded(shapes)).values())
passed('numerically_failing_custom_candidates_excluded')
native=dict(eligible=True,candidate=dict(depth=0,arm_id='Native'))
custom=dict(eligible=True,candidate=dict(depth=1,arm_id='S1'),comparisons=dict(native=dict(speedup=1.1,ci95=[1.08,1.12])))
assert recommendation(dict(native=native,overall=custom),cfg['deployment_rule'])['status']=='custom'
for modification,reason in [({'eligible':False},'errors'),({'comparisons':{'native':{'speedup':1.005,'ci95':[1.001,1.009]}}},'margin'),({'comparisons':{'native':{'speedup':1.02,'ci95':[.99,1.04]}}},'uncertainty')]:
    assert recommendation(dict(native=native,overall={**custom,**modification}),cfg['deployment_rule'])['status']=='native'
    passed('confirmation_fallback',reason=reason)
summary=dict(completed=True,checks=checks,screen_outcomes_including_anchors=sum(len(g['arms']) for g in planned),
    smoke_cases=sum(len(g['arms']) for g in groups(cfg,shapes,'smoke')))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(completed=True,checks=len(checks))),flush=True)
