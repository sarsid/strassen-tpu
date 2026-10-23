"""v001 protocol plus the parent's larger per-kernel v6e VMEM allowance.

The original 112 MiB candidates remain controls. Full-contraction candidates
also receive 120 MiB (Strassen) or 124 MiB (cubic), independently of the tuned
whole-layer Native limit. Include the parent's gate/up cubic tile explicitly.
"""
from strassen_mm.fusion_v001 import plan
from strassen_mm.parent_fused_adapter_v001 import specification
from strassen_mm.tuner_joint_v001 import arm
import tune_parent_fused_v001 as base

original_menu=base.menu

def menu(site,shape,dtype):
    for candidate,metadata,reason in original_menu(site,shape,dtype):
        yield candidate,metadata,reason
        if candidate['tile'][2]==shape[1]:
            mib=124 if candidate['family']=='cubic' else 120
            a=dict(candidate,vmem_limit_bytes=mib*1024**2,arm_id=candidate['arm_id']+f'__vmem{mib}')
            a['candidate_id']=a['variant']=a['arm_id']
            meta=plan(a,shape,specification(site,a['early']))
            yield a,meta,'estimated_vmem_above_kernel_limit' if meta['estimated_vmem_bytes']>a['vmem_limit_bytes'] else None
    if site=='gateup':
        for implementation in ('cubic','cubic_full'):
            for buffers in (1,2):
                a=dict(arm((2048,2048,512),0,'outputs',buffers,dtype),architecture='v6e',
                       implementation=implementation,family='cubic',algorithm='cubic',early=False,vmem_limit_bytes=124*1024**2,
                       arm_id=f'{implementation}_parenttile_b{buffers}_{dtype}')
                a['candidate_id']=a['variant']=a['arm_id']
                meta=plan(a,shape,specification(site))
                yield a,meta,None

base.menu=menu

if __name__=='__main__':base.main()
