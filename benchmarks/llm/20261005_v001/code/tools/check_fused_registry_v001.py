"""Affected planner/guard checks after adding full-row full-cubic candidates."""
import json, os
from pathlib import Path
from strassen_mm.fusion_v001 import Epilogue
from strassen_mm.kernels_fused_v003 import make_projection
from strassen_mm.tuner_joint_v001 import arm
from strassen_mm.tuner_fused_v002 import registry
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
checks=[]
def passed(name,**values):
    row=dict(test=name,**values);checks.append(row);print(json.dumps(row),flush=True)
def choice(impl='current',depth=1,mode='products',dtype='bfloat16',tile=(32,512,512)):
    return dict(arm(tile,depth,mode,2,dtype),architecture='v6e',implementation=impl)
cfg=dict(search=dict(output_tiles=[[32,512],[128,1024]],native_mib=[None,48],estimate_prune_mib=112))
for spec,n in ((Epilogue('swiglu'),1536),(Epilogue('qk_norm_rope',head_dim=256),768),(Epilogue('norm_residual_add',norm='gemma'),2304)):
    s=dict(id='qualification',m=32,k=1024,n=n,output_dtype='bfloat16')
    candidates,trace=registry(s,cfg,spec,include_early=True)
    assert len({a['arm_id'] for a in candidates})==len(candidates)
    assert {a['implementation'] for a in candidates}>={'native','current','cubic','cubic_full'}
    assert {a['depth'] for a in candidates}=={0,1,2}
    assert all(row['reasons'] for row in trace['candidates'] if row['disposition']=='pruned')
    if spec.kind=='norm_residual_add':
        assert all(a['tile'][1]>=n for a in candidates if a['tile'])
    (out/(spec.kind+'-registry.json')).write_text(json.dumps(trace,indent=2)+'\n')
    passed('fused_registry',epilogue=spec.kind,offered=len(candidates),decisions=len(trace['candidates']))

invalid=[(choice(depth=2,mode='outputs'),(32,512,512),Epilogue('swiglu',early=True)),
         (choice(),(32,512,1024),Epilogue('norm_residual_add')),
         (choice(),(32,512,513),Epilogue('swiglu')),
         (choice(),(32,512,129),Epilogue('qk_norm_rope'))]
for a,shape,spec in invalid:
    try: make_projection(a,shape,spec,interpret=True)
    except ValueError: pass
    else: raise AssertionError('Invalid candidate was accepted')
passed('invalid_candidate_guards',cases=len(invalid))
(out/'summary.json').write_text(json.dumps(dict(completed=True,checks=checks),indent=2)+'\n')
