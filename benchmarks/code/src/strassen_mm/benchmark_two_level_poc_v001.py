"""One-level versus two-level Pallas Strassen on four larger shapes."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base


def make_function(arm, shape, tile):
    from strassen_mm import kernels_v002 as kernels
    if arm == 'one_level':
        return kernels.make_matmul('strassen',shape,tile,variant='interleaved_output_accumulator',vmem_limit_bytes=48*1024**2)
    if arm == 'two_level':
        from strassen_mm.kernels_two_level_v001 import make_matmul
        return make_matmul(shape,tile,vmem_limit_bytes=48*1024**2)
    raise ValueError(arm)


class Runner(base.Runner):
    def compile_entries(self, group, a, b):
        import jax
        shape = tuple(group['shape'][d] for d in ('m','k','n'))
        entries = []
        for arm in group['arms']:
            self.check_deadline()
            context = {'group_id':group['group_id'],'arm_id':arm['arm_id'],'scope':'call'}
            entry = {**context,**arm}
            start = time.perf_counter_ns()
            print('  compiling '+arm['arm_id'],flush=True)
            try:
                tile = group['tile']
                fn = make_function(arm['arm_id'],shape,tuple(tile))
                executable = jax.jit(fn).lower(jax.ShapeDtypeStruct(a.shape,a.dtype),jax.ShapeDtypeStruct(b.shape,b.dtype)).compile()
                memory = executable.memory_analysis()
                memory = {k:getattr(memory,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                known = all(isinstance(memory[k],int) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))
                total = sum(memory[k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')) if known else None
                self.journal.emit('compilation',**context,status='ok',compile_ms=(time.perf_counter_ns()-start)/1e6,
                                  kernel_metadata=fn.metadata,memory_analysis=memory,estimated_executable_live_bytes=total)
                if total is not None and total > 8*1024**3:
                    entry.update(error='skipped_memory_preflight',error_message='Compiled executable exceeds the frozen 8 GiB budget')
                    del executable
                else:
                    entry.update(fn=fn,metadata=fn.metadata,executable=executable)
            except Exception as error:
                entry.update(error=base.error_status(error),error_message=str(error))
                self.emit_error(context,error,'compile')
            entries.append(entry)
        return entries

    def run_group(self, group):
        import jax
        shape = tuple(group['shape'][d] for d in ('m','k','n'))
        case = group['inputs'][0]
        a,b = base.generate_inputs(shape,**case)
        entries = self.compile_entries(group,a,b)
        self.execute_case(group,case,entries,a,b)
        del entries,a,b
        jax.clear_caches()
        gc.collect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-identity',type=Path,required=True)
    p.add_argument('--allocation-id',required=True);p.add_argument('--max-wall-seconds',type=float,default=2400)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir;journal=base.Journal(out,args.phase)
    campaign=json.loads(args.campaign.read_text());groups=[];done=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/campaign['shape_manifest']).read_text())['shapes']
        base.snapshot_sources(out,args.campaign,args.campaign.parent/campaign['shape_manifest'],args.campaign.parent/campaign['distribution_manifest'])
        if 'jax' in sys.modules:raise RuntimeError('Expected fresh process')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from strassen_mm.kernels_v002 import enable_qualified_mosaic_v7_compat
        environment=base.capture_environment(args,campaign,enable_qualified_mosaic_v7_compat())
        base.exclusive_json(out/'environment.json',environment)
        journal.emit('identity_check',**base.verify_identity(environment,args.expected_identity,campaign))
        arms=[{'arm_id':name,'algorithm':'strassen','variant':name} for name in campaign['arms']]
        groups=[{'group_id':s['id']+'__tile_'+'_'.join(map(str,tile)),'shape':s,'tile':tile,'arms':arms,'scopes':['call'],'timing':'poc',
                 'inputs':[{'distribution':'gaussian','seed':campaign['seed']+i}]} for i,s in enumerate(shapes) for tile in campaign['tiles']]
        base.exclusive_json(out/'planned_cases.json',groups)
        runner=Runner(args,campaign,journal)
        for i,g in enumerate(groups):
            print(f'[{i+1}/{len(groups)}] {g["group_id"]} starting',flush=True)
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(groups))
            runner.check_deadline();runner.run_group(g);done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(groups))
            print(f'[{i+1}/{len(groups)}] complete; {dict(journal.status_counts)}',flush=True)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        journal.emit('run_error',**error,traceback=traceback.format_exc())
        print(error,flush=True)
    summary={'completed':error is None and len(done)==8,'completed_groups':done,'planned_group_count':len(groups),
             'case_status_counts':dict(journal.status_counts),'error':error,
             'interpretation':'Four shapes, two shared tiles, one versus two levels only. One seed per shape. Matched-tile exploratory comparisons, not independent optimal tuning.'}
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
