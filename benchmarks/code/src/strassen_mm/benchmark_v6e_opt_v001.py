"""V6e kernel schedule and tile tuning on confirmed v5e winning shapes."""
import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from . import benchmark_v001 as base
from .benchmark_two_level_tune_v001 import Journal, Runner as ReuseRunner


def candidates(cfg):
    return [dict(c,arm_id=c['candidate_id'],family=family) for family,items in cfg['candidate_families'].items() for c in items]


def make_function(arm, shape):
    from . import kernels_v002 as old
    family=arm['family']
    if family=='native':return old.make_matmul('native',shape,variant='plain')
    if family=='cubic':return old.make_matmul('cubic_full',shape,tuple(arm['tile']),vmem_limit_bytes=arm['vmem_limit_bytes'])
    if arm['implementation']=='optimized':
        from .kernels_v6e_v001 import make_matmul
        return make_matmul(shape,tuple(arm['tile']),arm['depth'],mode=arm['mode'],order=arm['order'],vmem_limit_bytes=arm['vmem_limit_bytes'])
    if arm['depth']==1:
        return old.make_matmul('strassen',shape,tuple(arm['tile']),variant='interleaved_output_accumulator',vmem_limit_bytes=arm['vmem_limit_bytes'])
    from .kernels_two_level_v001 import make_matmul
    return make_matmul(shape,tuple(arm['tile']),vmem_limit_bytes=arm['vmem_limit_bytes'])


class Runner(ReuseRunner):
    def compile_entries(self,group,a,b):
        shape=tuple(group['shape'][d] for d in ('m','k','n'));entries=[]
        for arm in group['arms']:
            self.check_deadline();context=dict(group_id=group['group_id'],arm_id=arm['arm_id'],scope='call')
            entry={**context,**arm};start=time.perf_counter_ns()
            self.journal.emit('compile_start',**context,depth=arm.get('depth',0),tile=arm.get('tile'))
            print('Compiling '+arm['arm_id'],flush=True)
            try:
                fn=make_function(arm,shape);metadata={**fn.metadata,**arm};entry.update(fn=fn,metadata=metadata)
                lowered=jax.jit(fn).lower(jax.ShapeDtypeStruct(a.shape,a.dtype),jax.ShapeDtypeStruct(b.shape,b.dtype))
                executable=lowered.compile(compiler_options=arm['compiler_options']) if arm['compiler_options'] else lowered.compile()
                memory=executable.memory_analysis()
                memory={k:getattr(memory,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')}
                total=sum(memory.values()) if all(isinstance(x,int) for x in memory.values()) else None
                self.journal.emit('compilation',**context,status='ok',compile_ms=(time.perf_counter_ns()-start)/1e6,
                    kernel_metadata=metadata,memory_analysis=memory)
                if total is not None and total>self.campaign['memory']['estimated_live_device_budget_gib']*1024**3:
                    entry.update(error='skipped_memory_preflight',error_message='Compiled executable exceeds frozen HBM budget');del executable
                else:entry['executable']=executable
            except Exception as exc:
                entry.update(error=base.error_status(exc),error_message=str(exc));self.emit_error(context,exc,'compile')
            entries.append(entry)
        return entries

    def run_group(self,group):
        shape=tuple(group['shape'][d] for d in ('m','k','n'));entries=None
        for case in group['inputs']:
            self.check_deadline();a,b=base.generate_inputs(shape,**case)
            if entries is None:entries=self.compile_entries(group,a,b)
            self.execute_case(group,case,entries,a,b)
            if self.args.phase.endswith('-profile') or self.args.phase.endswith('-confirm') and case==group['inputs'][0]:
                profile_dir=self.args.output_dir/'profiles'/group['group_id']
                profile_dir.mkdir(parents=True,exist_ok=False)
                live=[e for e in entries if 'executable' in e and not e.get('error')]
                for e in live:
                    try:
                        (profile_dir/(e['arm_id']+'.hlo.txt')).write_text(e['executable'].as_text())
                        base.exclusive_json(profile_dir/(e['arm_id']+'.cost.json'),e['executable'].cost_analysis())
                    except Exception as exc:
                        self.journal.emit('profile_metadata_error',arm_id=e['arm_id'],message=str(exc))
                ad,bd=jax.device_put(a),jax.device_put(b);jax.block_until_ready((ad,bd))
                try:
                    options=jax.profiler.ProfileOptions();options.python_tracer_level=0;options.host_tracer_level=1
                    with jax.profiler.trace(str(profile_dir),profiler_options=options):
                        for e in live:
                            for iteration in range(20):
                                with jax.profiler.TraceAnnotation(e['arm_id'],iteration=iteration):
                                    y=e['executable'](ad,bd);y.block_until_ready();y.delete()
                    self.journal.emit('profile_complete',group_id=group['group_id'],path=str(profile_dir),arms=len(live),iterations=20)
                except Exception as exc:
                    self.journal.emit('profile_error',group_id=group['group_id'],error_type=type(exc).__name__,message=str(exc))
                del ad,bd
            del a,b;gc.collect()
        del entries;jax.clear_caches();gc.collect()


def choose(cases,cfg,shapes):
    registered={c['arm_id']:c for c in candidates(cfg)};selection={}
    for shape in shapes:
        selection[shape['id']]={}
        for family in cfg['arms']:
            rows=[r for r in cases if r['shape_id']==shape['id'] and registered[r['arm_id']]['family']==family
                  and (r.get('timing') or {}).get('sample_count')==cfg['timing']['screen']['repeats']
                  and (r.get('correctness') or {}).get('finite')]
            passing=[r for r in rows if r['eligible_for_speedup_claim']]
            if not rows:
                selection[shape['id']][family]=dict(status='no_measured_candidate');continue
            row=min(passing or rows,key=lambda r:(r['timing']['mean_ms'],r['arm_id']))
            selection[shape['id']][family]=dict(candidate=registered[row['arm_id']],
                status='eligible' if passing else 'diagnostic_numerical_failure',screen_mean_ms=row['timing']['mean_ms'])
    return selection


def groups(cfg,shapes,stage,selection):
    registered=candidates(cfg);result=[]
    for i,s in enumerate(shapes):
        if stage=='screen':
            pool=[c for c in registered if c['family']!='native']
            random.Random(cfg['order_seed']+i).shuffle(pool)
            batches=[[c for c in registered if c['family']=='native']]
            batches += [pool[j:j+4] for j in range(0,len(pool),4)]
        elif stage=='confirm':
            chosen=[v['candidate'] for v in selection[s['id']].values() if 'candidate' in v]
            default=next(c for c in registered if c['candidate_id']=='native_default')
            if default not in chosen:chosen.append(default)
            batches=[chosen]
        elif stage=='profile':
            batches=[[c for c in registered if c.get('profile')]]
        else:
            picked={}
            for c in registered:
                if c['family']=='native':key=('native',)
                elif c['family']=='cubic':key=('cubic',)
                else:key=(c['implementation'],c['depth'],c.get('mode'),c.get('order'))
                if key not in picked:
                    picked[key]=dict(c)
                    if c['tile'] is not None:picked[key]['tile']=[128,1024,1024]
            batches=[list(picked.values())]
        for j,arms in enumerate(batches):
            inputs=([dict(distribution='gaussian',seed=x+i) for x in cfg['confirm_seeds']] if stage=='confirm' else
                    [dict(distribution='integer' if stage=='smoke' else 'gaussian',seed=cfg['screen_seed']+i)])
            result.append(dict(group_id=s['id']+'__'+stage+'_'+str(j),shape=s,tile=None,arms=arms,scopes=['call'],timing=stage,inputs=inputs))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=5400);p.add_argument('--selection',type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir
    cfg=json.loads(args.campaign.read_text());stage=args.phase.rsplit('-',1)[-1]
    journal=Journal(out,args.phase);error=None;done=[];planned=[]
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        if stage=='smoke':shapes=[dict(id='aligned_integer_probe',m=128,k=2048,n=1024)]
        if stage=='profile':shapes=[s for s in shapes if s['id'] in cfg['profile_shapes']]
        if stage not in ('smoke','profile','screen','confirm'):raise ValueError('Unknown stage')
        base.snapshot_sources(out,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        selected=None
        if stage=='confirm':
            selection=json.loads(args.selection.read_text())
            if selection['allocation_id']!=args.allocation_id or selection['campaign_sha256']!=base.digest_file(args.campaign):
                raise ValueError('Selection allocation/configuration mismatch')
            selected=selection['selected'];base.exclusive_json(out/'selection_used.json',selection)
        planned=groups(cfg,shapes,stage,selected);base.exclusive_json(out/'planned_cases.json',planned)
        if 'jax' in sys.modules:raise RuntimeError('Fresh TPU process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        global jax
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat())
        name=env['identity']['device_kind'].lower().replace(' ','')
        if not ('v6e' in name or 'v6lite' in name) or env['device_count']!=1 or env['local_device_count']!=1 or env['process_count']!=1 or env['backend']!='tpu':
            raise RuntimeError('Exactly one v6e device required')
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
            if expected[key]!=env['identity'][key]:raise RuntimeError('Allocation identity mismatch: '+key)
        base.exclusive_json(out/'environment.json',env);journal.emit('identity_check',status='matched')
        from .benchmark_mlsys_shapes_v001 import install_reference_policy
        cache=install_reference_policy(cfg)
        if stage=='smoke':
            def integer_inputs(shape,distribution,seed):
                rng=np.random.default_rng(seed);m,k,n=shape
                return rng.integers(-1,2,(m,k)).astype(ml_dtypes.bfloat16),rng.integers(-1,2,(k,n)).astype(ml_dtypes.bfloat16)
            base.generate_inputs=integer_inputs
        runner=Runner(args,cfg,journal)
        for i,group in enumerate(planned):
            journal.emit('group_start',group_id=group['group_id'],index=i+1,total=len(planned))
            runner.run_group(group);cache.clear();gc.collect();done.append(group['group_id'])
            journal.emit('group_complete',group_id=group['group_id'],index=i+1,total=len(planned))
        if stage=='smoke' and (len(journal.cases)!=sum(len(g['arms']) for g in planned) or any(r['status']!='ok' or r['correctness']['max_abs_error']!=0 for r in journal.cases)):
            raise RuntimeError('Every compiled implementation must pass exact integer algebra before screening')
        if stage=='screen':base.exclusive_json(out/'selections.json',dict(allocation_id=args.allocation_id,
            campaign_sha256=base.digest_file(args.campaign),selected=choose(journal.cases,cfg,shapes),
            rule='Fastest numerically eligible screen mean per family; if none, fastest finite candidate retained only as numerical-failure diagnostic. No confirmation reselection.'))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None and len(done)==len(planned),stage=stage,error=error,completed_groups=done,
        planned_group_count=len(planned),case_status_counts=dict(journal.status_counts),
        interpretation='Synthetic dense MM; BF16 inputs/preadds with FP32 output. Levels apply inside tiles. Schedule and tile search on four documented v5e winners; fresh confirmation with original kernels and independently tuned Native; not a full LLM test.')
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
