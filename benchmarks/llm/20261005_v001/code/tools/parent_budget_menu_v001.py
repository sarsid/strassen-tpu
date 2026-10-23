"""Only audit the down tile whose old 112 MiB cap excluded plausible v6e arms."""
from copy import deepcopy


def shortlist(events,site,dtype,profile):
    if site!='down':raise ValueError('Budget audit is limited to down')
    candidates={}
    for row in events:
        a=row.get('arm',{})
        if (row.get('kind') not in ('candidate','candidate_failure','candidate_pruned') or row.get('site')!='down'
                or a.get('output_dtype')!=dtype or a.get('tile')!=[2048,2560,1024] or a.get('buffers')!=2):continue
        b=deepcopy(a);mib=124 if b['family']=='cubic' else 120
        for field in ('arm_id','candidate_id','variant'):b[field]=b[field].replace('__vmem112','__vmem'+str(mib))
        b['vmem_limit_bytes']=mib*1024**2;candidates[b['arm_id']]=b
    for family in ('cubic','s1','s2'):
        old=profile['profiles'][dtype][family]['down']
        if old['implementation']!='native':candidates[old['arm_id']]=old
    return [candidates[k] for k in sorted(candidates)]
