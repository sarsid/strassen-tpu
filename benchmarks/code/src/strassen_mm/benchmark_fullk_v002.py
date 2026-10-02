"""Separate FP32/BF16 full-K tuning, frozen choices, fresh paired confirmation."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from . import benchmark_v001 as base
from . import benchmark_v6e_opt_v001 as prior
from .benchmark_fullk_v001 import Runner
from .benchmark_two_level_tune_v001 import Journal


def candidates(s, smoke=False):
    arms = []; dtype = s['output_dtype']; k = s['k']
    def add(name, impl, tile=None, depth=0, buffers=2, **extra):
        arms.append(dict(arm_id=name, candidate_id=name, family=impl, algorithm=impl,
            implementation=impl, variant=name, tile=tile, depth=depth, buffers=buffers,
            output_dtype=dtype, compiler_options={}, vmem_limit_bytes=112*1024**2, **extra))
    for mib in ([None] if smoke else [None,32,48,64,96,112]):
        add('Native_default' if mib is None else f'Native_{mib}MiB', 'native')
        if mib is not None: arms[-1]['compiler_options']={'xla_tpu_scoped_vmem_limit_kib':mib*1024}
    for depth in (1,2):
        if not smoke or dtype == 'float32':
            add(f'Existing_S{depth}', 'incumbent', s['incumbents'][str(depth)]['tile'], depth,
                role='existing_control')
    for bm,bn in s['output_tiles']:
        for bk in s['panels']:
            for buffers in ([1,2] if bk == k or smoke else [2]):
                for depth in (1,2):
                    add(f'S{depth}_{bm}_{bn}_{bk}_b{buffers}', 'current', [bm,bn,bk], depth,
                        buffers, role='panel_curve')
                if smoke:
                    add(f'Cubic_{bm}_{bn}_{bk}_b{buffers}', 'cubic', [bm,bn,bk], 0, buffers)
    return arms


def expanded(shapes):
    return [{**s, 'base_shape_id':s['id'], 'id':s['id']+'__'+dtype, 'output_dtype':dtype}
            for s in shapes for dtype in ('float32','bfloat16')]


def select(cases, cfg, shapes):
    selected = {}
    for s in shapes:
        registered = {a['arm_id']:a for a in candidates(s)}
        rows = [r for r in cases if r['shape_id']==s['id'] and r['eligible_for_speedup_claim']
                and r['status']=='ok' and (r.get('timing') or {}).get('sample_count')==cfg['timing']['screen']['repeats']]
        choices = {}
        for role in ('native','short_s1','full_s1','short_s2','full_s2'):
            def matches(r):
                a=registered[r['arm_id']]
                if role=='native': return a['implementation']=='native'
                depth=int(role[-1]); full=role.startswith('full')
                return a['implementation'] in ('current','incumbent') and a['depth']==depth and (a['tile'][2]==s['k'])==full
            valid=[r for r in rows if matches(r)]
            if valid:
                best=min(valid,key=lambda r:(r['timing']['mean_ms'],r['arm_id']))
                choices[role]=dict(status='eligible',candidate=registered[best['arm_id']],
                                   screen_mean_ms=best['timing']['mean_ms'])
            else: choices[role]=dict(status='no_eligible_candidate')
        selected[s['id']]=choices
    return selected


def groups(cfg, shapes, stage, selected=None, start=0, count=None):
    if stage=='smoke':
        shapes=[dict(id='integer_nonpower',m=256,k=1536,n=1024,seed_offset=0,
                     output_tiles=[[128,512]],panels=[512,1536],
                     incumbents={str(d):dict(tile=[128,512,512]) for d in (1,2)})]
    else: shapes=shapes[start:None if count is None else start+count]
    result=[]
    for s in expanded(shapes):
        offered=candidates(s,stage=='smoke')
        if stage=='confirm':
            by_id={}
            def add(a,role):
                item=by_id.setdefault(a['arm_id'],dict(a,headline_roles=[]))
                if role not in item['headline_roles']:item['headline_roles'].append(role)
            for role,v in selected[s['id']].items():
                if 'candidate' in v:add(v['candidate'],role)
            for name,role in [('Native_default','native_default'),('Existing_S1','existing_s1'),('Existing_S2','existing_s2')]:
                add(next(a for a in offered if a['arm_id']==name),role)
            for role in ('short_s1','full_s1','short_s2','full_s2'):
                c=selected[s['id']][role].get('candidate')
                if c:
                    a=dict(c);bm,bn,bk=a['tile'];name=f'Cubic_{bm}_{bn}_{bk}_b{a["buffers"]}'
                    a.update(arm_id=name,candidate_id=name,variant=name,implementation='cubic',family='cubic',algorithm='cubic',depth=0,role='matched_cubic')
                    add(a,'matched_'+role)
            arms=list(by_id.values())
        else: arms=offered
        inputs=([dict(distribution='gaussian',seed=x+s['seed_offset']) for x in cfg['confirm_seeds']]
                if stage=='confirm' else [dict(distribution='integer' if stage=='smoke' else 'gaussian',seed=cfg['screen_seed']+s['seed_offset'])])
        result.append(dict(group_id=s['id']+'__'+stage,shape=s,tile=None,arms=arms,scopes=['call'],
                           timing=stage,inputs=inputs,offered_candidates=offered))
    return result


def make_function(a,shape):
    import jax
    import jax.numpy as jnp
    from .kernels_fullk_v002 import make_matmul
    fn=make_matmul(shape,tuple(a['tile']) if a['tile'] else None,a['implementation'],depth=a['depth'],
        output_dtype=a['output_dtype'],buffers=a['buffers'],vmem_limit_bytes=a['vmem_limit_bytes'])
    m,k,n=shape
    result=jax.eval_shape(fn,jax.ShapeDtypeStruct((m,k),jnp.bfloat16),jax.ShapeDtypeStruct((k,n),jnp.bfloat16))
    if result.shape!=(m,n) or result.dtype!=jnp.dtype(a['output_dtype']):raise ValueError('Output precision/shape mismatch')
    return fn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=10800);p.add_argument('--selection',type=Path)
    p.add_argument('--shape-start',type=int,default=0);p.add_argument('--shape-count',type=int)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.campaign.read_text());stage=args.phase.rsplit('-',1)[-1]
    if stage not in ('smoke','screen','confirm'):raise ValueError(stage)
    journal=Journal(args.output_dir,args.phase);planned=[];done=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        base.snapshot_sources(args.output_dir,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        selected=None
        if stage=='confirm':
            selection=json.loads(args.selection.read_text())
            if selection['allocation_id']!=args.allocation_id or selection['campaign_sha256']!=base.digest_file(args.campaign):raise ValueError('Selection identity/config mismatch')
            selected=selection['selected'];base.exclusive_json(args.output_dir/'selection_used.json',selection)
        planned=groups(cfg,shapes,stage,selected,args.shape_start,args.shape_count)
        base.exclusive_json(args.output_dir/'planned_cases.json',planned)
        if 'jax' in sys.modules:raise RuntimeError('Fresh TPU process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        os.environ.update(STRASSEN_MANAGE_CEILING='0',STRASSEN_VMEM_PROFILE='compact16')
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        prior.jax=jax;prior.make_function=make_function
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat())
        if env['backend']!='tpu' or env['device_count']!=1 or env['local_device_count']!=1 or env['process_count']!=1 or 'v6' not in env['identity']['device_kind'].lower():raise RuntimeError('Single v6e required')
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
            if expected[key]!=env['identity'][key]:raise RuntimeError('Identity mismatch: '+key)
        base.exclusive_json(args.output_dir/'environment.json',env)
        from .benchmark_mlsys_shapes_v001 import install_reference_policy
        cache=install_reference_policy(cfg)
        if stage=='smoke':
            def integer_inputs(shape,distribution,seed):
                rng=np.random.default_rng(seed);m,k,n=shape
                a=rng.integers(-1,2,(m,k),dtype=np.int8);a[:,np.arange(k)%16!=0]=0
                return a.astype(ml_dtypes.bfloat16),rng.integers(-1,2,(k,n),dtype=np.int8).astype(ml_dtypes.bfloat16)
            base.generate_inputs=integer_inputs
        runner=Runner(args,cfg,journal)
        for i,g in enumerate(planned):
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(planned))
            runner.run_group(g);cache.clear();gc.collect();done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(planned))
        if len(journal.cases)!=sum(len(g['arms'])*len(g['inputs']) for g in planned):raise RuntimeError('Missing terminal outcomes')
        if stage=='smoke' and any(r['status']!='ok' or r['correctness']['max_abs_error']!=0 for r in journal.cases):raise RuntimeError('Exact TPU qualification failed')
        if stage=='screen':
            choices=select(journal.cases,cfg,[g['shape'] for g in planned])
            base.exclusive_json(args.output_dir/'selections.json',dict(allocation_id=args.allocation_id,
                campaign_sha256=base.digest_file(args.campaign),selected=choices,rule=cfg['selection_policy']))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    base.exclusive_json(args.output_dir/'summary.json',dict(completed=error is None and len(done)==len(planned),
        error=error,stage=stage,completed_groups=done,planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='12 development geometries, separate output precisions. Tune short/full independently, freeze choices, confirm on three fresh inputs. No unseen-shape selector validation or LLM accuracy claim.'))
    journal.close();return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(main())
