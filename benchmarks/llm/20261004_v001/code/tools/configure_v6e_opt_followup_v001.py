"""Freeze first-round selections as controls for a bounded hybrid follow-up."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.cohort/'opt-screen-finished.json').read_text())
    if receipt['status']!='completed':raise ValueError('First screen must be complete')
    run=Path(receipt['run']);source=a.cohort/'source/configs/v6e_opt_v001'
    if hashlib.sha256((run/'completion.json').read_bytes()).hexdigest()!=receipt['completion_sha256']:
        raise ValueError('Receipt mismatch')
    path=run/'artifacts/selections.json';selection=json.loads(path.read_text())
    cfg=json.loads((source/'campaign.json').read_text())
    if hashlib.sha256((source/'campaign.json').read_bytes()).hexdigest()!=selection['campaign_sha256']:
        raise ValueError('Prior config mismatch')
    cfg.update(campaign_id='v6e_opt_v002',screen_seed=2026092371,confirm_seeds=[2026092372,2026092373,2026092374],order_seed=92371)
    focus=set(cfg['profile_shapes'])
    families={'native':cfg['candidate_families']['native']}
    for family in ['original1','original2','optimized1','optimized2']:
        families[family]=[]
        for shape,choices in selection['selected'].items():
            if shape not in focus:continue
            if choices[family]['status']!='eligible':raise ValueError('Unqualified first-round incumbent')
            c=copy.deepcopy(choices[family]['candidate'])
            c.update(candidate_id=c['candidate_id']+'__'+shape,only_shape=shape,profile=False)
            c.pop('arm_id',None);families[family].append(c)
    families['hybrid2']=[]
    tiles=[[2048,1024,1024],[1024,2048,2048],[2048,2048,2048],
           [1024,2048,4096],[2048,1024,2048],[2048,1024,4096]]
    for tile in tiles:
        families['hybrid2'].append(dict(candidate_id='hybrid2_'+'_'.join(map(str,tile)),algorithm='strassen',
            variant='outer_deferred_interleaved',implementation='hybrid',depth=2,mode='outer_deferred',order='interleaved',
            tile=tile,compiler_options={},vmem_limit_bytes=112*1024**2,profile=False))
    cfg['candidate_families']=families;cfg['arms']=list(families)
    cfg['followup_provenance']=dict(prior_cohort=str(a.cohort),prior_selection=str(path),
        prior_selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rationale='49 leaf accumulators exceeded VMEM on large tiles; retain 7 outer accumulators, reconstruct inner level per K panel. New allocation requires remeasurement of all controls; no cross-allocation pooling.')
    (a.output_dir/'campaign.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shape_data=json.loads((source/'shapes.json').read_text())
    shape_data['shapes']=[s for s in shape_data['shapes'] if s['id'] in focus]
    (a.output_dir/'shapes.json').write_text(json.dumps(shape_data,indent=2)+'\n')
    for name in ['distributions.json','v5e_evidence.json']:
        (a.output_dir/name).write_bytes((source/name).read_bytes())
    print(json.dumps(dict(config=str(a.output_dir),screen_attempts=30,shape_count=2)))


if __name__=='__main__':main()
