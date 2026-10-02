"""v6e joint-search extension to 168 shapes, including independent cubic.

Every aligned pilot candidate is retained. Boundary handling uses existing
tile-multiple padding, not the discarded experimental padding algorithms.
"""
import math
import random
import statistics
from .tuner_joint_v001 import expanded, arm


def registry(s,cfg):
    m,k,n=(s[x] for x in ('m','k','n'));dtype=s['output_dtype']
    geometries=set(map(tuple,cfg['search']['output_tiles']))
    controls=s.get('historical_controls',{}).get(dtype,[])
    protected={(tuple(c['tile']),c['depth'],c.get('accumulator','products'),c['buffers']) for c in controls}
    geometries.update(tuple(c['tile'][:2]) for c in controls)
    if n%2560==0:geometries.add((1024,2560))
    # Tiny/skinny shapes need feasible small tiles; keep the three geometries
    # minimizing padded output area, with larger useful tiles breaking ties.
    if min(m,n)<512:
        geometries.update({(32,512),(128,512),(256,512),(512,512)})
        geometries=set(sorted(geometries,key=lambda t:(math.ceil(m/t[0])*t[0]*math.ceil(n/t[1])*t[1],-min(m,t[0])*min(n,t[1]),t))[:3])
        geometries.update(tuple(c['tile'][:2]) for c in controls)
    divisors=[v for v in range(512,k+1,512) if k%v==0]
    panels={math.ceil(k/512)*512,512,*[c['tile'][2] for c in controls]}
    for cap in (1024,k//4,k//2):
        pool=[v for v in divisors if v<=cap]
        if pool:panels.add(max(pool))
    offered=[dict(arm(dtype=dtype,mib=mib),architecture='v6e') for mib in cfg['search']['native_mib']]
    decisions=[]
    for bm,bn in sorted(geometries):
        for bk in sorted(panels):
            for depth in (0,1,2):
                for mode in (('products',) if depth==0 else ('products','outputs')):
                    for buffers in (1,2):
                        a=arm((bm,bn,bk),depth,mode,buffers,dtype)
                        if depth==0:
                            name=f'Cubic_{bm}_{bn}_{bk}_b{buffers}'
                            a.update(arm_id=name,candidate_id=name,variant=name,family='cubic',algorithm='cubic',implementation='cubic')
                        a['architecture']='v6e'
                        scratch=(4 if depth==0 else 7 if mode=='products' else 4 if dtype=='bfloat16' else 0)*bm*bn
                        estimate=buffers*2*(bm*bk+bk*bn)+scratch+2*(2 if dtype=='bfloat16' else 4)*bm*bn
                        keep=(tuple(a['tile']),depth,mode,buffers) in protected
                        reasons=[]
                        alignment=max(2,2**depth) # parent blocked cubic splits once
                        if bm%(8*alignment) or bn%(128*alignment) or bk%(128*alignment):reasons.append('leaf_alignment')
                        if estimate>cfg['search']['estimate_prune_mib']*1024**2 and not keep:reasons.append('rough_memory_estimate_exceeds_search_cap; not_proven_infeasible')
                        padded=[math.ceil(x/t)*t for x,t in zip((m,k,n),(bm,bk,bn))]
                        a.update(padded_shape_mkn=padded,padded_volume_ratio=math.prod(padded)/(m*k*n))
                        decisions.append(dict(candidate=a,disposition='pruned' if reasons else 'offered',reasons=reasons,
                            estimated_vmem_bytes=estimate,historical_control=keep,contraction='full' if bk==padded[1] else 'short'))
                        if not reasons:offered.append(a)
                        if depth==0 and buffers==2:
                            full=dict(a,arm_id=a['arm_id'].replace('Cubic_','Full_cubic_'),candidate_id=a['arm_id'].replace('Cubic_','Full_cubic_'),implementation='cubic_full')
                            # Full-tile cubic uses the qualified compiler-managed
                            # pipeline. Its buffer count is not explicitly tuned.
                            full['buffers']=None
                            full_estimate=4*bm*bn+4*(bm*bk+bk*bn)
                            full_reasons=['rough_memory_estimate_exceeds_search_cap; not_proven_infeasible'] if full_estimate>cfg['search']['estimate_prune_mib']*1024**2 else []
                            decisions.append(dict(candidate=full,disposition='pruned' if full_reasons else 'offered',reasons=full_reasons,estimated_vmem_bytes=full_estimate,historical_control=False,contraction='full' if bk==padded[1] else 'short'))
                            if not full_reasons:offered.append(full)
    assert len({a['arm_id'] for a in offered})==len(offered)
    return offered,dict(shape=s,output_geometries=[list(t) for t in sorted(geometries)],all_aligned_k_divisors=divisors,
        offered_k_panels=sorted(panels),omitted_k_divisors=[v for v in divisors if v not in panels],candidates=decisions)


def select(cases,cfg,shapes):
    selected={}
    for s in shapes:
        offered,trace=registry(s,cfg);rows=[r for r in cases if r['shape_id']==s['id']]
        anchors={r['group_id']:r for r in rows if r['arm_id']=='Native_default'}
        def eligible(r):
            t=r.get('timing') or {}
            return r['status']=='ok' and r['eligible_for_speedup_claim'] and t.get('sample_count')==cfg['timing']['screen']['repeats'] and math.isfinite(t.get('mean_ms',float('nan'))) and t['mean_ms']>0
        scores={}
        for a in offered:
            observations=[r for r in rows if r['arm_id']==a['arm_id']]
            valid=[r for r in observations if eligible(r) and eligible(anchors[r['group_id']])]
            scores[a['arm_id']]=dict(candidate=a,status='eligible' if valid and len(valid)==len(observations) else 'excluded',
                score=statistics.median(r['timing']['mean_ms']/anchors[r['group_id']]['timing']['mean_ms'] for r in valid) if valid else None,
                observations=[dict(group_id=r['group_id'],status=r['status'],eligible=eligible(r),timing=r.get('timing'),correctness=r.get('correctness')) for r in observations])
        filters={'native':lambda a:a['family']=='native','cubic':lambda a:a['family']=='cubic','overall':lambda a:True}
        for d in (1,2):
            filters[f's{d}']=lambda a,d=d:a['depth']==d
            for mode in ('products','outputs'):
                for contraction in ('short','full'):
                    filters[f's{d}_{mode}_{contraction}']=lambda a,d=d,mode=mode,c=contraction:(a['depth']==d and a['accumulator']==mode and (a['tile'][2]>=s['k'])==(c=='full'))
        choices={}
        for role,accept in filters.items():
            pool=[x for x in scores.values() if x['status']=='eligible' and accept(x['candidate'])]
            if pool:
                best=min(pool,key=lambda x:(x['score'],x['candidate']['arm_id']))
                choices[role]={key:best[key] for key in ('candidate','status','score')}
                choices[role]['reason']='minimum_eligible_same_batch_native_normalized_latency; stable_ID_tiebreak'
            else:choices[role]=dict(status='no_eligible_candidate')
        if 'candidate' not in choices['native']:raise ValueError('No eligible Native: '+s['id'])
        selected[s['id']]=dict(choices=choices,candidate_scores=scores,registry=trace)
    return selected


def groups(cfg,shapes,stage,selection=None,start=0,count=None):
    if stage=='smoke':
        shapes=[dict(id='qual_'+str(i),m=m,k=k,n=n,seed_offset=i) for i,(m,k,n) in enumerate([(32,512,512),(33,513,513),(3,3,3)])]
        result=[]
        for s in expanded(shapes):
            arms=[dict(arm(dtype=s['output_dtype']),architecture='v6e')]
            arms.append(dict(arm((32,512,512),dtype=s['output_dtype']),architecture='v6e',arm_id='Full_cubic',candidate_id='Full_cubic',family='cubic',implementation='cubic_full',buffers=None))
            for depth in (0,1,2):
                for mode in (('products',) if depth==0 else ('products','outputs')):
                    for buffers in (1,2):
                        a=arm((32,512,512),depth,mode,buffers,s['output_dtype']);a['architecture']='v6e'
                        if depth==0:a.update(arm_id='Cubic_b'+str(buffers),candidate_id='Cubic_b'+str(buffers),family='cubic',implementation='cubic',algorithm='cubic')
                        arms.append(a)
            result.append(dict(group_id=s['id']+'__smoke',shape=s,tile=None,arms=arms,scopes=['call'],timing='smoke',inputs=[dict(distribution='integer',seed=9270000)]))
        return result
    result=[]
    for s in expanded(shapes[start:None if count is None else start+count]):
        offered,_=registry(s,cfg);default=next(a for a in offered if a['arm_id']=='Native_default')
        if stage=='screen':
            native=[a for a in offered if a['family']=='native'];custom=[a for a in offered if a['family']!='native']
            random.Random(cfg['order_seed']+s['seed_offset']).shuffle(custom)
            size=cfg['search']['batch_size'];batches=[native]+[[default,*custom[i:i+size]] for i in range(0,len(custom),size)]
        elif stage=='confirm':
            by_id={}
            def add(a,role):
                entry=by_id.setdefault(a['arm_id'],dict(a,headline_roles=[]))
                if role not in entry['headline_roles']:entry['headline_roles'].append(role)
            for role,c in selection[s['id']]['choices'].items():
                if 'candidate' in c:
                    add(c['candidate'],role)
                    if role in ('s1','s2'):add(c['candidate'],{'s1':'one_level','s2':'two_level'}[role])
            add(default,'native_default')
            for d in (1,2):
                a=selection[s['id']]['choices'][f's{d}'].get('candidate')
                if a:
                    for label,mode,buf in [('other_accumulator','outputs' if a['accumulator']=='products' else 'products',a['buffers']),('other_buffers',a['accumulator'],3-a['buffers'])]:
                        b=dict(arm(a['tile'],d,mode,buf,s['output_dtype']),architecture='v6e');add(b,f's{d}_{label}')
                    b=dict(a,arm_id=f'Matched_cubic_s{d}',candidate_id=f'Matched_cubic_s{d}',family='cubic',implementation='cubic',algorithm='cubic',depth=0);add(b,f'matched_s{d}')
            batches=[list(by_id.values())]
        else:raise ValueError(stage)
        for i,arms in enumerate(batches):
            inputs=[dict(distribution='gaussian',seed=v+s['seed_offset']) for v in cfg['confirm_seeds']] if stage=='confirm' else [dict(distribution='gaussian',seed=cfg['screen_seed']+s['seed_offset'])]
            result.append(dict(group_id=s['id']+'__'+stage+'_'+str(i),shape=s,tile=None,arms=arms,scopes=['call'],timing=stage,inputs=inputs))
    return result
