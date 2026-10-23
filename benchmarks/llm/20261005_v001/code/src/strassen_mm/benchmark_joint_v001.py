"""Joint accumulator/tiling/buffering tuning with an explicit decision record."""
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


from .tuner_joint_v001 import groups, select, registry


def make_function(a, shape):
    from .kernels_joint_v001 import make_matmul
    return make_matmul(shape, tuple(a['tile']) if a['tile'] else None,
        implementation=a['implementation'], depth=a['depth'],
        accumulator=a.get('accumulator','products'), output_dtype=a['output_dtype'],
        buffers=a['buffers'], vmem_limit_bytes=a['vmem_limit_bytes'])


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
        if stage=='screen':
            unique={g['shape']['id']:g['shape'] for g in planned}
            base.exclusive_json(args.output_dir/'candidate_registry.json',{key:registry(s,cfg)[1] for key,s in unique.items()})
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
            choices=select(journal.cases,cfg,list({g['shape']['id']:g['shape'] for g in planned}.values()))
            base.exclusive_json(args.output_dir/'selections.json',dict(allocation_id=args.allocation_id,
                campaign_sha256=base.digest_file(args.campaign),selected=choices,rule=cfg['selection_policy']))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    base.exclusive_json(args.output_dir/'summary.json',dict(completed=error is None and len(done)==len(planned),
        error=error,stage=stage,completed_groups=done,planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='Six development geometries, separate output precisions; batch-normalized screening freezes accumulator/tile/panel/buffer choices before fresh paired confirmation. No unseen-shape or arbitrary-input accuracy guarantee.'))
    journal.close();return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(main())
