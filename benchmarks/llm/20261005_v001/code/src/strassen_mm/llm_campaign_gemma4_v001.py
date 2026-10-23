"""Gemma4 per-attention-type v6e candidate planning; no queue changes.

This module plans experiments. It never marks a planned entry as measured and
never selects a kernel without execution evidence.
"""
import hashlib, json, math
from .model_gemma4_fused_v001 import geometry as layer_geometry
from .fusion_v004 import plan
from .tuner_joint_v001 import arm

VERSION = "llm_campaign_gemma4_v001"

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

def geometry(config,batch,sequence,layer_index=0):
    c,g,specs=layer_geometry(config,batch,sequence,layer_index,rounding='accumulator')
    return c,g['projection_shapes_mkn'],specs

def candidates(shape,spec,dtype,family):
    """Every offered/pruned choice carries an explicit reason and architecture."""
    if dtype != "bfloat16":raise ValueError("Gemma4 candidate search supports BF16 output only")
    m,k,n=shape
    if family in ('native_default','native_tuned'):
        for mib in ([None] if family=='native_default' else [None,32,48,64,96,112]):
            a=dict(arm(dtype=dtype,mib=mib),architecture='v6e',family=family)
            yield dict(candidate_id=identity(a),arm=a,disposition='offered',reasons=[],options_scope='whole_layer')
        return
    if family not in ('cubic_tuned','s1','s2'):raise ValueError('Unknown family')
    depth={'cubic_tuned':0,'s1':1,'s2':2}[family]
    divisor=2**depth if depth else 2
    bms=[v for v in (256,512,1024,2048) if v<=max(256,m)]
    bns=sorted({math.ceil(n/(128*divisor))*128*divisor}) if spec.kind=='norm_residual_add' else [512,1024,2048,2560]
    bks=sorted({512,1024,2048,math.ceil(k/(128*divisor))*128*divisor})
    for bm in bms:
        for bn in bns:
            for bk in bks:
                for mode in (['blocked','full'] if depth==0 else ['products','outputs']):
                    for buffers in (1,2):
                        a=dict(arm((bm,bn,bk),depth,'outputs' if mode in ('blocked','full') else mode,buffers,dtype),architecture='v6e',family=family)
                        if depth==0:a['implementation']='cubic_full' if mode=='full' else 'cubic'
                        record=dict(candidate_id=identity(a),arm=a,reasons=[],disposition='offered')
                        try:
                            meta=plan(a,shape,spec);record['metadata']=dict(meta,kernel_version='kernels_fused_v005',dot_precision='DEFAULT',norm_reduction_precision='DEFAULT' if spec.kind=='qk_norm_rope' else 'FP32')
                            actual_divisor=1 if mode=='full' else divisor
                            record['leaf_mkn']=[bm//actual_divisor,bk//actual_divisor,bn//actual_divisor]
                            record['preferred_v6e_256_leaf']=bk//actual_divisor%256==0 and bn//actual_divisor%256==0
                            record['padded_work_ratio']=math.prod(meta['padded_shape_mkn'])/math.prod(shape)
                            if record['padded_work_ratio']>2:record['reasons'].append('padding_work_above_2x')
                            if meta['estimated_vmem_bytes']>112*1024**2:record['reasons'].append('rough_vmem_estimate_above_112MiB_not_proof_of_infeasibility')
                        except ValueError as e:record['reasons'].append(str(e))
                        if record['reasons']:record['disposition']='pruned'
                        yield record
