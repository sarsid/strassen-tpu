"""Fixed matched-tile contraction curves; FP32 and parent BF16 contracts separate."""
import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from . import benchmark_v001 as base
from . import benchmark_v6e_opt_v001 as prior
from .benchmark_two_level_tune_v001 import Journal, Runner as SimpleRunner


def make_function(arm,shape):
    from .kernels_fullk_v001 import make_matmul
    return make_matmul(shape,tuple(arm['tile']) if arm['tile'] else None,arm['implementation'],
        output_dtype=arm['output_dtype'],depth=arm['depth'],mode=arm.get('mode'),
        vmem_limit_bytes=arm['vmem_limit_bytes'])


def arms_for(s,smoke=False):
    output=s['output_dtype'];limit=(104 if output=='bfloat16' else 112)*1024**2
    arms=[]
    def add(name,implementation,tile=None,depth=0,**extra):
        arms.append(dict(arm_id=name,candidate_id=name,family=implementation,algorithm=implementation,
            variant=name,implementation=implementation,tile=tile,depth=depth,
            mode=('deferred' if depth==1 else 'hybrid') if implementation=='current' else None,
            compiler_options={},output_dtype=output,vmem_limit_bytes=limit,**extra))
    for mib in ([None] if smoke else [None,32,48,64,96,112]):
        add('Native_default' if mib is None else f'Native_{mib}MiB','native')
        if mib is not None:arms[-1]['compiler_options']={'xla_tpu_scoped_vmem_limit_kib':mib*1024}
    if not smoke and output=='float32':
        for depth in (1,2):add(f'S{depth}_incumbent','current',s['incumbents'][str(depth)]['tile'],depth,role='incumbent')
    families=[('Parent_S1','parent_s1',1),('Blocked_cubic','parent_cubic',0)]
    if output=='float32':families=[('S1','current',1),('S2','current',2)]+families
    for bm,bn in s['output_tiles']:
        for bk in s['panels']:
            for label,implementation,depth in families:
                add(f'{label}_{bm}_{bn}_{bk}',implementation,[bm,bn,bk],depth,role='panel_curve')
    return arms


def groups(cfg,shapes,smoke=False):
    if smoke:
        shapes=[dict(id='integer_'+dtype,m=256,k=1024,n=1024,output_dtype=dtype,
                     output_tiles=[[128,512]],panels=[512,1024],seed_offset=0) for dtype in ['float32','bfloat16']]
    return [dict(group_id=s['id'],shape=s,tile=None,arms=arms_for(s,smoke),scopes=['call'],
        timing='smoke' if smoke else 'measure',inputs=[dict(distribution='integer',seed=924500)] if smoke else
        [dict(distribution='gaussian',seed=x+s['seed_offset']) for x in cfg['confirm_seeds']]) for s in shapes]


class Runner(prior.Runner):
    run_group=SimpleRunner.run_group
    def compile_entries(self,group,a,b):
        entries=super().compile_entries(group,a,b)
        folder=self.args.output_dir/'compiled'/group['group_id'];folder.mkdir(parents=True,exist_ok=False)
        for entry in entries:
            if 'executable' not in entry:continue
            # Keep lowering evidence to inspect spills/layouts without another TPU run.
            try:
                (folder/(entry['arm_id']+'.hlo.txt')).write_text(entry['executable'].as_text())
                base.exclusive_json(folder/(entry['arm_id']+'.cost.json'),entry['executable'].cost_analysis())
            except Exception as exc:self.journal.emit('metadata_error',arm_id=entry['arm_id'],message=str(exc))
        return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=7200)
    p.add_argument('--shape-start',type=int,default=0);p.add_argument('--shape-count',type=int)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.campaign.read_text());smoke=args.phase.endswith('-smoke')
    journal=Journal(args.output_dir,args.phase);done=[];planned=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        shapes=shapes[args.shape_start:None if args.shape_count is None else args.shape_start+args.shape_count]
        planned=groups(cfg,shapes,smoke)
        base.snapshot_sources(args.output_dir,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        base.exclusive_json(args.output_dir/'planned_cases.json',planned)
        if 'jax' in sys.modules:raise RuntimeError('Fresh TPU process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        os.environ['STRASSEN_MANAGE_CEILING']='0';os.environ['STRASSEN_VMEM_PROFILE']='compact16'
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        prior.jax=jax;prior.make_function=make_function
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat())
        if env['backend']!='tpu' or env['device_count']!=1 or env['local_device_count']!=1 or env['process_count']!=1 or 'v6' not in env['identity']['device_kind'].lower():
            raise RuntimeError('Single v6e required')
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
            if expected[key]!=env['identity'][key]:raise RuntimeError('Identity mismatch: '+key)
        base.exclusive_json(args.output_dir/'environment.json',env)
        from .benchmark_mlsys_shapes_v001 import install_reference_policy
        cache=install_reference_policy(cfg)
        if smoke:
            def integer_inputs(shape,distribution,seed):
                rng=np.random.default_rng(seed);m,k,n=shape
                a=rng.integers(-1,2,(m,k),dtype=np.int8);a[:,np.arange(k)%16!=0]=0
                # At K=1024 at most 64 contributions: even BF16 output is exact.
                return a.astype(ml_dtypes.bfloat16),rng.integers(-1,2,(k,n),dtype=np.int8).astype(ml_dtypes.bfloat16)
            base.generate_inputs=integer_inputs
        runner=Runner(args,cfg,journal)
        for i,g in enumerate(planned):
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(planned))
            runner.run_group(g);cache.clear();gc.collect();done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(planned))
        if len(journal.cases)!=sum(len(g['arms'])*len(g['inputs']) for g in planned):raise RuntimeError('Missing terminal outcomes')
        if smoke and any(r['status']!='ok' or r['correctness']['max_abs_error']!=0 for r in journal.cases):
            raise RuntimeError('Compiled exact qualification failed')
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None and len(done)==len(planned),error=error,completed_groups=done,
        planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='Fixed aligned contraction curves, all candidates reported, no confirmation selection. Three fresh Gaussian inputs × 30 paired rounds. FP32 and BF16 outputs are separate contracts. Compilation/VMEM failures retained as infeasible candidates. Parent kernel pinned; current software and Gaussian inputs differ from upstream historical experiment.')
    base.exclusive_json(args.output_dir/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
