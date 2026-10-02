"""Qualify adapter arithmetic, architecture separation and the full registry."""
import json,os,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from strassen_mm.kernels_arch_study_v001 import make_matmul
from strassen_mm.tuner_arch_study_v001 import groups,registry,expanded,select
from strassen_mm.tuner_joint_v001 import registry as pilot_registry

out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
cfg=json.loads(Path('configs/arch_v6e_168_v001/campaign.json').read_text())
shapes=json.loads(Path('configs/arch_v6e_168_v001/shapes.json').read_text())['shapes']
checks=[]
def passed(name,**kw):
    checks.append(dict(check=name,**kw));print(checks[-1],flush=True)

# Exact, independently computed integer references cover two K panels, edge
# padding/crop, both accumulator strategies, both levels and output contracts.
rng=np.random.default_rng(92700)
for s in [dict(id='test',m=33,k=513,n=513,seed_offset=0)]:
    a=rng.integers(-1,2,(s['m'],s['k'])).astype(np.float32);a[:,np.arange(s['k'])%16!=0]=0
    b=rng.integers(-1,2,(s['k'],s['n'])).astype(np.float32)
    ref=a.astype(np.float64)@b.astype(np.float64)
    for g in groups(cfg,[],'smoke'):
        if g['shape']['m']!=33:continue
        for arm in g['arms']:
            if arm.get('buffers')==1:continue # identical interpret arithmetic; hardware smoke tests both
            fn=make_matmul(arm,(s['m'],s['k'],s['n']),interpret=True)
            y=jax.jit(fn)(jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16))
            assert str(y.dtype)==arm['output_dtype']
            np.testing.assert_array_equal(np.asarray(y).astype(np.float32),ref)
            jax.clear_caches();passed('exact_boundary',arm=arm['arm_id'],output=arm['output_dtype'])

# Native numerical rounding and v5e wrapper are exactly final-round-only.
v5=json.loads(Path('configs/mlsys_shapes_v001/campaign.json').read_text())
from strassen_mm.benchmark_mlsys_shapes_v001 import candidates
for original in candidates(v5):
    if original['family']!='native' and original['tile'] not in ([16,128,128],[16,256,256],[32,512,512]):continue
    arm={**original,'architecture':'v5e','output_dtype':'float32'}
    shape=(17,129,131);a=jnp.asarray(rng.normal(size=shape[:2]),jnp.bfloat16);b=jnp.asarray(rng.normal(size=shape[1:]),jnp.bfloat16)
    fp=jax.jit(make_matmul(arm,shape,interpret=True))(a,b)
    bf=jax.jit(make_matmul({**arm,'output_dtype':'bfloat16'},shape,interpret=True))(a,b)
    np.testing.assert_array_equal(np.asarray(bf),np.asarray(fp.astype(jnp.bfloat16)))
    jax.clear_caches();passed('v5e_final_round_only',arm=arm['arm_id'])

pcfg=json.loads(Path('configs/joint_v001/campaign.json').read_text())
pshapes=json.loads(Path('configs/joint_v001/shapes.json').read_text())['shapes']
for s in expanded(pshapes):
    before,_=pilot_registry(s,pcfg);after,_=registry(s,cfg);keys={(a['arm_id'],tuple(a['tile'] or [])) for a in after}
    assert all((a['arm_id'],tuple(a['tile'] or [])) in keys for a in before)
passed('all_aligned_pilot_candidates_preserved',groups=len(expanded(pshapes)))

assert len(shapes)==168 and len({tuple(s[d] for d in ('m','k','n')) for s in shapes})==168
counts={};sample_shapes=[shapes[i] for i in (0,5,36,76,130,167)];fake=[]
for s in expanded(shapes):
    arms,trace=registry(s,cfg);assert {'native','cubic','s1','s2'}<={a['family'] for a in arms}
    assert all(d['reasons'] for d in trace['candidates'] if d['disposition']=='pruned')
    assert {a['depth'] for a in arms}<={0,1,2}
    counts[s['id']]=len(arms)
for i,g in enumerate(groups(cfg,sample_shapes,'screen')):
    for a in g['arms']:
        scale=1+(i%5);ratio=0.85 if a['family']=='cubic' else 1
        fake.append(dict(shape_id=g['shape']['id'],group_id=g['group_id'],arm_id=a['arm_id'],status='ok',eligible_for_speedup_claim=True,timing=dict(sample_count=7,mean_ms=ratio*scale)))
sel=select(fake,cfg,expanded(sample_shapes))
assert all(s['choices']['overall']['candidate']['family']=='cubic' for s in sel.values())
for g in groups(cfg,sample_shapes,'confirm',sel):
    assert {v['seed'] for v in g['inputs']}.isdisjoint({cfg['screen_seed']+g['shape']['seed_offset']})
    assert {'native','native_default','cubic','s1','s2'}<={r for a in g['arms'] for r in a['headline_roles']}
passed('coverage_independent_cubic_selection_and_fresh_confirmation',shapes=168,contracts=336)
(out/'summary.json').write_text(json.dumps(dict(completed=True,checks=checks,candidate_counts=counts,screen_outcomes=sum(len(g['arms']) for g in groups(cfg,shapes,'screen'))),indent=2)+'\n')
