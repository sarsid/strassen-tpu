"""Fixed-policy boundary probe: existing S1/S2 versus edge and Native fringes."""
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


def make_function(arm, shape):
    if arm['family'] == 'native':
        from .kernels_v002 import make_matmul
        return make_matmul('native', shape, variant='plain')
    kwargs = dict(mode=arm['mode'], buffers=2, traversal='mn', vmem_limit_bytes=112*1024**2)
    if arm['policy'] == 'control':
        from .kernels_v6e_v003 import make_matmul
    else:
        from .kernels_padded_v001 import make_matmul
        kwargs['policy'] = arm['policy']
    return make_matmul(shape, tuple(arm['tile']), arm['depth'], **kwargs)


def groups(cfg, shapes, smoke=False):
    result = []
    if smoke:
        shapes = [dict(id='integer_'+str(i), m=s[0], k=s[1], n=s[2],
                       tiles={'1':[128,512,512], '2':[128,512,512]})
                  for i,s in enumerate([(256,1024,1024), (255,1023,1023),
                                        (257,1025,1025), (17,19,23)])]
    for i,s in enumerate(shapes):
        arms = [dict(arm_id='Native', candidate_id='Native', family='native',
                     algorithm='native', variant='plain', tile=None, compiler_options={})]
        for depth in (1,2):
            for policy, suffix in [('control',''), ('edge_tiles','_padded'), ('native_fringe','_native_fringe')]:
                label = 'S'+str(depth)+suffix
                arms.append(dict(arm_id=label, candidate_id=label, family=label,
                    algorithm='strassen' if policy=='control' else 'strassen_boundary',
                    variant=policy, policy=policy, depth=depth, tile=s['tiles'][str(depth)],
                    mode='deferred' if depth==1 else 'hybrid', compiler_options={}))
        inputs = [dict(distribution='integer',seed=924100+i)] if smoke else [
            dict(distribution='gaussian',seed=x+i) for x in cfg['confirm_seeds']]
        result.append(dict(group_id=s['id'],shape=s,tile=None,arms=arms,scopes=['call'],
                           timing='smoke' if smoke else 'measure',inputs=inputs))
    return result


class Runner(prior.Runner):
    # Reuse paired synchronized timings, without the older unconditional profiles.
    run_group = SimpleRunner.run_group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=5400)
    p.add_argument('--shape-start',type=int,default=0);p.add_argument('--shape-count',type=int)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.campaign.read_text());smoke=args.phase.endswith('-smoke')
    journal=Journal(args.output_dir,args.phase);done=[];planned=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        shapes=shapes[args.shape_start:None if args.shape_count is None else args.shape_start+args.shape_count]
        planned=groups(cfg,shapes,smoke)
        base.snapshot_sources(args.output_dir,args.campaign,args.campaign.parent/cfg['shape_manifest'],
                              args.campaign.parent/cfg['distribution_manifest'])
        base.exclusive_json(args.output_dir/'planned_cases.json',planned)
        if 'jax' in sys.modules:raise RuntimeError('Fresh TPU process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        prior.jax=jax;prior.make_function=make_function
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat())
        if env['backend']!='tpu' or env['device_count']!=1 or env['process_count']!=1 or env['local_device_count']!=1:
            raise RuntimeError('Single TPU required')
        if 'v6' not in env['identity']['device_kind'].lower():raise RuntimeError('v6e required')
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
            if expected[key]!=env['identity'][key]:raise RuntimeError('Identity mismatch: '+key)
        base.exclusive_json(args.output_dir/'environment.json',env)
        from .benchmark_mlsys_shapes_v001 import install_reference_policy
        cache=install_reference_policy(cfg)
        if smoke:
            def integer_inputs(shape,distribution,seed):
                rng=np.random.default_rng(seed);m,k,n=shape
                return rng.integers(-1,2,(m,k)).astype(ml_dtypes.bfloat16),rng.integers(-1,2,(k,n)).astype(ml_dtypes.bfloat16)
            base.generate_inputs=integer_inputs
        runner=Runner(args,cfg,journal)
        for i,g in enumerate(planned):
            def boundary_indices(size,count,rng):
                if size<=count:return np.arange(size,dtype=np.int32)
                required={0,size-1}
                for tile in g['shape']['tiles'].values():
                    for width in tile[:2]:
                        split=size//width*width
                        required.update(v for v in (split-1,split,split+1) if 0<=v<size)
                pool=np.array(sorted(set(range(size))-required),dtype=np.int32)
                return np.array(sorted(required | set(rng.choice(pool,count-len(required),replace=False).tolist())),dtype=np.int32)
            base.sample_indices=boundary_indices
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(planned))
            runner.run_group(g);cache.clear();gc.collect();done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(planned))
        if smoke and (len(journal.cases)!=sum(len(g['arms']) for g in planned)
                      or any(r['status']!='ok' or r['correctness']['max_abs_error']!=0 for r in journal.cases)):
            raise RuntimeError('All policies must pass compiled exact integer algebra')
        expected_cases=sum(len(g['arms'])*len(g['inputs']) for g in planned)
        if len(journal.cases)!=expected_cases:raise RuntimeError('Missing case outcomes')
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        journal.emit('run_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None and len(done)==len(planned),error=error,
        completed_groups=done,planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='Fixed policies, matched interior kernel/tile per depth; three fresh Gaussian inputs with 30 paired rounds. Complete-call latency includes all boundary handling. BF16 inputs/preadds, FP32 output. Sampled all-K FP64 errors and full-output finiteness. Native default is contextual, not independently tuned. No LLM accuracy claim.')
    base.exclusive_json(args.output_dir/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
