"""All 168 shapes, dimension-only tile coverage and frozen paired confirmation."""
import argparse,gc,json,math,os,sys,traceback
from pathlib import Path
from . import benchmark_v6e_diagnostic_v001 as routing
from . import benchmark_v6e_opt_v001 as prior
from . import benchmark_v001 as base

def chosen_tiles(cfg,shape):
    m,k,n=[shape[d] for d in ('m','k','n')]
    def rank(tile):
        bm,bn,bk=tile
        volume=math.ceil(m/bm)*bm*math.ceil(k/bk)*bk*math.ceil(n/bn)*bn
        return (volume,-bm*bn*bk,tuple(tile))
    return sorted(cfg['tile_pool'],key=rank)[:4 if min(m,k,n)<1024 else 6]

def shape_candidates(cfg,shape):
    tiles=chosen_tiles(cfg,shape);result=[]
    for c in prior.candidates(cfg):
        if c['family']=='native' or c['tile'] in tiles[:2 if c.get('implementation')=='original' else len(tiles)]:
            result.append(c)
    return result

def plan_groups(cfg,shapes,stage,selected=None,start=0,count=None):
    if stage=='smoke':shapes=cfg['qualification_shapes']
    else:shapes=shapes[start:start+count if count is not None else None]
    groups=[]
    for i,s in enumerate(shapes,start):
        candidates=shape_candidates(cfg,s)
        if stage=='screen':arms=candidates
        elif stage=='smoke':
            tile=chosen_tiles(cfg,s)[0]
            arms=[c for c in candidates if c['candidate_id']=='native_default' or c['tile']==tile]
        elif stage=='confirm':
            arms_by_id={}
            def add(c,role):
                arm=arms_by_id.setdefault(c['arm_id'],dict(c,headline_roles=[]))
                if role not in arm['headline_roles']:arm['headline_roles'].append(role)
            for family,item in selected[s['id']].items():
                if 'candidate' in item:add(item['candidate'],family)
            add(next(c for c in candidates if c['candidate_id']=='native_default'),'native_default')
            for family in ('one_level','two_level'):
                item=selected[s['id']][family]
                if 'candidate' in item:
                    tile=item['candidate']['tile']
                    add(next(c for c in candidates if c['family']=='cubic' and c['tile']==tile),'matched_'+family)
            arms=list(arms_by_id.values())
        else:raise ValueError(stage)
        inputs=[dict(distribution='gaussian',seed=x+i) for x in cfg['confirm_seeds']] if stage=='confirm' else [dict(distribution='integer' if stage=='smoke' else 'gaussian',seed=cfg['screen_seed']+i)]
        groups.append(dict(group_id=s['id']+'__'+stage,shape=s,tile=None,arms=arms,scopes=['call'],timing=stage,inputs=inputs,
            offered_tiles=chosen_tiles(cfg,s),candidate_policy=cfg['candidate_policy']))
    return groups

class Journal(prior.Journal):
    def emit(self,event,**fields):
        if event=='case_result':
            fields['headline_roles']=(fields.get('kernel_metadata') or {}).get('headline_roles',[])
        super().emit(event,**fields)

def main():
    p=argparse.ArgumentParser()
    for name in ('campaign','output-dir','expected-identity'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=10800);p.add_argument('--selection',type=Path)
    p.add_argument('--shape-start',type=int,default=0);p.add_argument('--shape-count',type=int)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.campaign.read_text());stage=args.phase.rsplit('-',1)[-1]
    journal=Journal(out,args.phase);error=None;done=[];planned=[]
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        base.snapshot_sources(out,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        selected=None
        if stage=='confirm':
            selection=json.loads(args.selection.read_text())
            if selection['allocation_id']!=args.allocation_id or selection['campaign_sha256']!=base.digest_file(args.campaign):raise ValueError('Selection identity/config mismatch')
            selected=selection['selected'];base.exclusive_json(out/'selection_used.json',selection)
        planned=plan_groups(cfg,shapes,stage,selected,args.shape_start,args.shape_count)
        base.exclusive_json(out/'planned_cases.json',planned)
        if 'jax' in sys.modules:raise RuntimeError('Fresh TPU process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes;prior.jax=jax
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat())
        name=env['identity']['device_kind'].lower().replace(' ','')
        if not ('v6e' in name or 'v6lite' in name) or env['device_count']!=1 or env['local_device_count']!=1 or env['process_count']!=1 or env['backend']!='tpu':raise RuntimeError('Exactly one v6e required')
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
            if expected[key]!=env['identity'][key]:raise RuntimeError('Identity mismatch: '+key)
        base.exclusive_json(out/'environment.json',env);journal.emit('identity_check',status='matched')
        from .benchmark_mlsys_shapes_v001 import install_reference_policy
        cache=install_reference_policy(cfg)
        if stage=='smoke':
            def integer_inputs(shape,distribution,seed):
                rng=np.random.default_rng(seed);m,k,n=shape
                return rng.integers(-1,2,(m,k),dtype=np.int8).astype(ml_dtypes.bfloat16),rng.integers(-1,2,(k,n),dtype=np.int8).astype(ml_dtypes.bfloat16)
            base.generate_inputs=integer_inputs
        runner=prior.Runner(args,cfg,journal)
        for i,group in enumerate(planned):
            journal.emit('group_start',group_id=group['group_id'],index=i+1,total=len(planned))
            runner.run_group(group);cache.clear();gc.collect();done.append(group['group_id'])
            journal.emit('group_complete',group_id=group['group_id'],index=i+1,total=len(planned))
            if stage=='smoke' and any(r['status']!='ok' or r['correctness']['max_abs_error']!=0 for r in journal.cases):raise RuntimeError('Exact qualification failed; broad screening blocked')
        if len(journal.cases)!=sum(len(g['arms'])*len(g['inputs'])for g in planned):raise RuntimeError('Incomplete terminal case coverage')
        if stage=='screen':
            subset=[g['shape']for g in planned]
            chosen=prior.choose(journal.cases,cfg,subset)
            base.exclusive_json(out/'selections.json',dict(allocation_id=args.allocation_id,campaign_sha256=base.digest_file(args.campaign),selected=chosen,rule=cfg['selection_policy']))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None and len(done)==len(planned),stage=stage,error=error,completed_groups=done,planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='168 fixed development shapes. Dimension-only candidate policy, paired screening, frozen choices, three fresh confirmation inputs. BF16 inputs/preadds, FP32 output. Synthetic matrices; no prediction-accuracy claim.')
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1

if __name__=='__main__':raise SystemExit(main())
