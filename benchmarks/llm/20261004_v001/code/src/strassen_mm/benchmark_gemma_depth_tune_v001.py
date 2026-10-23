"""Independent Native and Strassen depth tuning on one large Gemma projection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from strassen_mm import benchmark_v001 as base
from strassen_mm.benchmark_two_level_tune_v001 import Journal, Runner as ReuseRunner
from strassen_mm.benchmark_two_level_poc_v001 import make_function


def candidates(campaign):
    result = []
    for item in campaign['native_candidates']:
        result.append(dict(arm_id=item['candidate_id'],family='native',algorithm='native',variant='plain',tile=None,**item))
    for family in ('one_level','two_level'):
        for tile in campaign['tiles']:
            name=family+'_'+'_'.join(map(str,tile))
            result.append(dict(arm_id=name,candidate_id=name,family=family,algorithm='strassen',variant=family,tile=tile,compiler_options={}))
    assert len({c['candidate_id'] for c in result})==len(result)
    return result


def select_candidates(cases,campaign,shapes):
    selected={}
    for shape in shapes:
        selected[shape['id']]={}
        for family in campaign['arms']:
            allowed={c['candidate_id']:c for c in candidates(campaign) if c['family']==family}
            valid=[r for r in cases if r['shape_id']==shape['id'] and r['arm_id'] in allowed and
                   r['status']=='ok' and r['eligible_for_speedup_claim'] and r['timing']['sample_count']==campaign['timing']['screen']['repeats']]
            if not valid:raise ValueError('No eligible candidate for '+family)
            best=min(valid,key=lambda r:(r['timing']['mean_ms'],r['arm_id']))
            selected[shape['id']][family]=dict(candidate=allowed[best['arm_id']],screen_group_id=best['group_id'],
                screen_mean_ms=best['timing']['mean_ms'],screen_relative_l2=best['correctness']['relative_l2'])
    return selected


def make_groups(campaign,shapes,stage,selected=None):
    registered=candidates(campaign);groups=[]
    for i,shape in enumerate(shapes):
        if stage=='screen':
            arm_groups=[[c for c in registered if c['family']=='native']]
            arm_groups += [[c for c in registered if c['family']!='native' and c['tile']==tile] for tile in campaign['tiles']]
            random.Random(campaign['order_seed']+i).shuffle(arm_groups)
            for j,arms in enumerate(arm_groups):
                groups.append(dict(group_id=shape['id']+'__screen_'+str(j),shape=shape,tile=None,arms=arms,scopes=['call'],timing='screen',
                    inputs=[dict(distribution='gaussian',seed=campaign['screen_seed']+i)]))
        else:
            arms=[selected[shape['id']][family]['candidate'] for family in campaign['arms']]
            if any(a not in registered for a in arms):raise ValueError('Selected candidate outside frozen search')
            if [a['family'] for a in arms]!=campaign['arms']:raise ValueError('Wrong selected family')
            default=next(c for c in registered if c['candidate_id']=='native_default')
            if default not in arms:arms.append(default)
            groups.append(dict(group_id=shape['id']+'__confirm',shape=shape,tile=None,arms=arms,scopes=['call'],timing='confirm',
                inputs=[dict(distribution='gaussian',seed=s+i) for s in campaign['confirm_seeds']]))
    return groups


class Runner(ReuseRunner):
    def compile_entries(self,group,a,b):
        import jax
        from strassen_mm import kernels_v002 as kernels
        shape=tuple(group['shape'][d] for d in ('m','k','n'));entries=[]
        for arm in group['arms']:
            self.check_deadline()
            context=dict(group_id=group['group_id'],arm_id=arm['arm_id'],scope='call')
            entry={**context,**arm};start=time.perf_counter_ns()
            print('  compiling '+arm['arm_id'],flush=True)
            try:
                fn=(kernels.make_matmul('native',shape,variant='plain') if arm['family']=='native' else
                    make_function(arm['family'],shape,tuple(arm['tile'])))
                metadata={**fn.metadata,'compiler_options':arm['compiler_options'],'candidate_id':arm['candidate_id'],'family':arm['family']}
                entry.update(fn=fn,metadata=metadata)
                lowered=jax.jit(fn).lower(jax.ShapeDtypeStruct(a.shape,a.dtype),jax.ShapeDtypeStruct(b.shape,b.dtype))
                executable=lowered.compile(compiler_options=arm['compiler_options']) if arm['compiler_options'] else lowered.compile()
                memory=executable.memory_analysis()
                memory={k:getattr(memory,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                required=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')
                total=sum(memory[k] for k in required) if all(isinstance(memory[k],int) for k in required) else None
                self.journal.emit('compilation',**context,status='ok',compile_ms=(time.perf_counter_ns()-start)/1e6,
                    kernel_metadata=metadata,memory_analysis=memory,estimated_executable_live_bytes=total)
                if total is not None and total>8*1024**3:
                    entry.update(error='skipped_memory_preflight',error_message='Compiled executable exceeds frozen 8 GiB budget');del executable
                else:entry['executable']=executable
            except Exception as error:
                unsupported=bool(arm['compiler_options']) and any(s in str(error).lower() for s in ('unknown','unrecognized','unsupported','not supported','no such'))
                entry.update(error='unsupported_compiler_option' if unsupported else base.error_status(error),error_message=str(error))
                self.emit_error(context,error,'compile')
            entries.append(entry)
        return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-identity',type=Path,required=True)
    p.add_argument('--allocation-id',required=True);p.add_argument('--max-wall-seconds',type=float,default=1800)
    p.add_argument('--selection',type=Path)
    args=p.parse_args();stage=args.phase.rsplit('-',1)[-1]
    if stage not in ('screen','confirm'):raise ValueError('Expected screen or confirm')
    args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir;journal=Journal(out,args.phase)
    cfg=json.loads(args.campaign.read_text());groups=[];done=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/cfg['shape_manifest']).read_text())['shapes']
        base.snapshot_sources(out,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        selected=None
        if stage=='confirm':
            selection=json.loads(args.selection.read_text())
            if selection['allocation_id']!=args.allocation_id or selection['campaign_sha256']!=hashlib.sha256(args.campaign.read_bytes()).hexdigest():raise ValueError('Selection allocation/configuration mismatch')
            selected=selection['selected'];base.exclusive_json(out/'selection_used.json',selection)
        groups=make_groups(cfg,shapes,stage,selected);base.exclusive_json(out/'planned_cases.json',groups)
        if 'jax' in sys.modules:raise RuntimeError('Expected fresh process')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from strassen_mm.kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat());base.exclusive_json(out/'environment.json',env)
        journal.emit('identity_check',**base.verify_identity(env,args.expected_identity,cfg))
        runner=Runner(args,cfg,journal)
        for i,g in enumerate(groups):
            print(f'[{i+1}/{len(groups)}] {g["group_id"]} starting',flush=True)
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(groups))
            runner.run_group(g);done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(groups))
            print(f'[{i+1}/{len(groups)}] complete; {dict(journal.status_counts)}',flush=True)
        if stage=='screen':base.exclusive_json(out/'selections.json',dict(allocation_id=args.allocation_id,
            campaign_sha256=hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
            rule='Fastest eligible arithmetic mean per family; exact ties by candidate ID.',selected=select_candidates(journal.cases,cfg,shapes)))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc());print(error,flush=True)
    summary=dict(completed=error is None and len(done)==len(groups),completed_groups=done,planned_group_count=len(groups),
        case_status_counts=dict(journal.status_counts),error=error,stage=stage,
        interpretation='One Gemma shape, eight tiles per Strassen depth, four Native compiler settings. Independent selection then three fresh seeds; complete-call timing includes padding. Native tiles remain compiler-managed. Synthetic MM only.')
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
