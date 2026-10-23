"""Deterministic workload identities and inspectable v6e candidate decisions.

This module plans experiments. It never marks a planned entry as measured and
never selects a kernel without execution evidence.
"""
import hashlib, json, math
from .model_fused_v004 import validate_config,SITES
from .fusion_v003 import Epilogue,plan
from .tuner_joint_v001 import arm

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

def geometry(config,batch,sequence):
    c=validate_config(config);m=batch*sequence;h=c['hidden_size'];i=c['intermediate_size'];d=c['head_dim']
    q=c['num_attention_heads']*d;kv=c['num_key_value_heads']*d;gemma=c['model_type']=='gemma3_text'
    shapes=dict(q=(m,h,q),k=(m,h,kv),v=(m,h,kv),o=(m,q,h),gateup=(m,h,2*i),down=(m,i,h))
    kinds=dict(q='qk_norm_rope' if c['model_type']!='mistral' else 'rope',k='qk_norm_rope' if c['model_type']!='mistral' else 'rope',v='none',o='norm_residual_add' if gemma else 'residual_add',gateup='geglu' if gemma else 'swiglu',down='norm_residual_add' if gemma else 'residual_add')
    return c,shapes,{s:Epilogue(kinds[s],rounding='accumulator',norm='gemma' if gemma else 'qwen',head_dim=d,eps=c['rms_norm_eps']) for s in SITES}

def candidates(shape,spec,dtype,family):
    """Every offered/pruned choice carries an explicit reason and architecture."""
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
                            meta=plan(a,shape,spec);record['metadata']=dict(meta,kernel_version='kernels_fused_v004',dot_precision='DEFAULT',norm_reduction_precision='DEFAULT' if spec.kind=='qk_norm_rope' else 'FP32')
                            record['leaf_mkn']=[bm//divisor,bk//divisor,bn//divisor]
                            record['preferred_v6e_256_leaf']=bk//divisor%256==0 and bn//divisor%256==0
                            record['padded_work_ratio']=math.prod(meta['padded_shape_mkn'])/math.prod(shape)
                            if record['padded_work_ratio']>2:record['reasons'].append('padding_work_above_2x')
                            if meta['estimated_vmem_bytes']>112*1024**2:record['reasons'].append('rough_vmem_estimate_above_112MiB_not_proof_of_infeasibility')
                        except ValueError as e:record['reasons'].append(str(e))
                        if record['reasons']:record['disposition']='pruned'
                        yield record

def entries(protocol,models):
    indexed={r['id']:r for r in models}
    for item in protocol['models']:
        record=indexed[item['id']]
        if record['status']!='metadata_ready' or record['revision']!=item['revision']:raise ValueError('Pinned model metadata mismatch')
        for workload in protocol['prefill']:
            c,shapes,specs=geometry(record['config'],**workload)
            for family in protocol['algorithms']:
                for dtype in protocol['output_dtypes']:
                    key=dict(model=item['id'],revision=item['revision'],batch=workload['batch'],sequence=workload['sequence'],algorithm=family,output_dtype=dtype,architecture=protocol['architecture'])
                    yield dict(entry_id=identity(key),**key,projection_shapes_mkn=shapes,model_type=c['model_type'],status='planned',requires=['official_checkpoint_qualification','frozen_profile','independent_confirmation'],measurement=None)
