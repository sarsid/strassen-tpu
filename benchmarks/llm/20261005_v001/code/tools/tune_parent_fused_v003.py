"""Bound the measured search cost without dropping a family or dtype.

Use parent-sized budgets for full K, retaining one/two buffers there. For short
K use two input buffers, which permit pipelining across K panels. The preceding
v002 pilot remains the explicit 112-vs-parent-budget control. Model computation,
sample counts, accuracy gates and independent confirmation remain unchanged.
"""
from strassen_mm.fusion_v001 import plan
from strassen_mm.parent_fused_adapter_v001 import specification
from strassen_mm.tuner_joint_v001 import arm
import tune_parent_fused_v001 as base

original_menu=base.menu

def menu(site,shape,dtype):
    for candidate,metadata,reason in original_menu(site,shape,dtype):
        full=candidate['tile'][2]==shape[1]
        if not full and candidate['buffers']==1:continue
        mib=(124 if candidate['family']=='cubic' else 120) if full else 112
        a=dict(candidate,vmem_limit_bytes=mib*1024**2,arm_id=candidate['arm_id']+f'__vmem{mib}')
        a['candidate_id']=a['variant']=a['arm_id']
        meta=plan(a,shape,specification(site,a['early']))
        yield a,meta,'estimated_vmem_above_kernel_limit' if meta['estimated_vmem_bytes']>a['vmem_limit_bytes'] else None
    if site=='gateup':
        for implementation in ('cubic','cubic_full'):
            a=dict(arm((2048,2048,512),0,'outputs',2,dtype),architecture='v6e',implementation=implementation,
                   family='cubic',algorithm='cubic',early=False,vmem_limit_bytes=124*1024**2,
                   arm_id=f'{implementation}_parenttile_b2_{dtype}')
            a['candidate_id']=a['variant']=a['arm_id']
            yield a,plan(a,shape,specification(site)),None

base.menu=menu

if __name__=='__main__':base.main()
