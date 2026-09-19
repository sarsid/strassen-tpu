"""N8 matched MM+BF16 epilogues: layout, fusion and early finalization.

Complete-call and explicitly reused prepared-buffer scopes are both measured.
Packed-unfused controls share the paired gate/up layout with fused kernels;
their difference isolates epilogue fusion for the same product schedule. Early
Strassen versus standard fused shares layout and tiles but may change FP32 sum
order as recorded by each kernel. Inputs here are synthetic Gaussian, never
misrepresented as checkpoint activations or inference-quality evidence.
"""
from __future__ import annotations
import gc
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from . import support_n9_v001 as support
base=support.base


def arms_for(group,campaign,selections=None):
    tile=campaign['matched_tile'];kind=group['kind'];shape=group['shape_mkn']
    arms=[{'arm_id':'native_joint_graph','algorithm':'native','variant':'plain','tile':tile}]
    for algorithm in ('cubic_full','cubic_quadrant','strassen'):
        variant='plain' if algorithm=='cubic_full' else 'interleaved'
        for label,extra in (('unpacked_unfused',{}),('packed_unfused',{'packed':True}),('fused',{'fused':True})):
            # Residual has no packing transform. One unfused control is sufficient.
            if kind=='residual' and label=='packed_unfused': continue
            arms.append({'arm_id':algorithm+'_'+label,'algorithm':algorithm,'variant':variant,'tile':tile,**extra})
        if algorithm=='strassen':
            arms.append({'arm_id':'strassen_early','algorithm':'strassen','variant':'interleaved','tile':tile,
                         'fused':True,'early':True})
    if selections:
        matches=[row for row in selections['by_shape'].values() if row.get('shape_mkn')==shape]
        if len(matches)>1: raise ValueError('Ambiguous N5 selection for application geometry')
        if matches:
            for family in ('cubic','strassen'):
                value=matches[0].get(family)
                if value:
                    arms.append({'arm_id':family+'_n5_selected_unfused','algorithm':value['algorithm'],
                       'variant':value['variant'],'tile':value['tile'],'selection_provenance':value})
    return arms


def sample_indices(size,count,rng):
    if size<=count: return np.arange(size,dtype=np.int32)
    middle=rng.choice(np.arange(1,size-1),count-2,replace=False)
    return np.sort(np.concatenate(([0,size-1],middle))).astype(np.int32)


def independent_reference(a,b,residual,kind,rows,cols):
    lhs=a[rows].astype(np.float64)
    first=lhs@b[:,cols].astype(np.float64)
    first=first.astype(ml_dtypes.bfloat16).astype(np.float32)
    if kind=='swiglu':
        second=(lhs@b[:,cols+b.shape[1]//2].astype(np.float64)).astype(ml_dtypes.bfloat16).astype(np.float32)
        activated=(first/(np.float32(1)+np.exp(-first))).astype(ml_dtypes.bfloat16).astype(np.float32)
        return (activated*second).astype(ml_dtypes.bfloat16).astype(np.float32)
    return (first+residual[np.ix_(rows,cols)].astype(np.float32)).astype(ml_dtypes.bfloat16).astype(np.float32)


def correctness(output,reference,rows,cols,gate):
    finite=bool(jax.device_get(jnp.all(jnp.isfinite(output))))
    actual=np.asarray(output[np.ix_(rows,cols)],dtype=np.float32)
    error=actual.astype(np.float64)-reference.astype(np.float64)
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(reference.astype(np.float64)),1e-30))
    absolute=float(np.max(np.abs(error)));scale=float(np.max(np.abs(reference)))
    return {'finite':finite,'relative_l2':relative,'max_abs_error':absolute,'max_abs_reference':scale,
        'passed':bool(finite and relative<=gate['relative_l2_max'] and absolute<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*scale),
        'reference_scope':'sampled NumPy FP64 dot across allK using exact BF16 operands; matched BF16 epilogue boundaries',
        'sample_rows':rows.tolist(),'sample_columns':cols.tolist(),'gate':gate}


def main(argv=None):
    args=support.parser('N8').parse_args(argv);started=time.monotonic();campaign=support.begin(args)
    journal=base.Journal(args.output_dir,'N8');completed=False;error=None;results=[];selected={}
    budget=support.Budget(args.max_wall_seconds)
    try:
        environment=support.initialize_device(args,campaign,journal)
        global jax,jnp,np,ml_dtypes
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from .kernels_n8_v001 import make_epilogue
        selections=None
        if campaign.get('n5_selections'):
            path=support.resolve(args.campaign,campaign['n5_selections']);selections=json.loads(path.read_text())
            if selections['environment_identity']!=environment['identity']:
                raise ValueError('N5 selection environment identity differs from this application run')
            shutil.copyfile(path,args.output_dir/'n5_selections.json')
        planned=[{**group,'arms':arms_for(group,campaign,selections)} for group in campaign['groups']]
        base.exclusive_json(args.output_dir/'planned_cases.json',planned)
        for index,group in enumerate(planned):
            budget.check();group_dir=args.output_dir/f'group-{index:02d}';group_dir.mkdir()
            print(f"N8 [{index+1}/{len(planned)}] {group['id']} starting",flush=True)
            rng=np.random.Generator(np.random.PCG64(campaign['seed']+index))
            m,k,n=group['shape_mkn'];width=n//2 if group['kind']=='swiglu' else n
            ah=(rng.standard_normal((m,k),dtype=np.float32)/np.float32(math.sqrt(k))).astype(ml_dtypes.bfloat16)
            bh=rng.standard_normal((k,n),dtype=np.float32).astype(ml_dtypes.bfloat16)
            rh=rng.standard_normal((m,n),dtype=np.float32).astype(ml_dtypes.bfloat16) if group['kind']=='residual' else None
            rows=sample_indices(m,campaign['sample_rows'],rng);cols=sample_indices(width,campaign['sample_columns'],rng)
            reference=independent_reference(ah,bh,rh,group['kind'],rows,cols)
            with (group_dir/'sampled_reference.npy').open('xb') as f: np.save(f,reference,allow_pickle=False)
            base.exclusive_json(group_dir/'inputs.json',{'seed':campaign['seed']+index,'generator':'NumPy PCG64 Gaussian; A divided bysqrtK; B and residual standardnormal; quantizeonceBF16',
                'shape_mkn':[m,k,n],'sample_rows':rows.tolist(),'sample_columns':cols.tolist(),
                'a_sha256':__import__('hashlib').sha256(ah.tobytes()).hexdigest(),'b_sha256':__import__('hashlib').sha256(bh.tobytes()).hexdigest(),
                'residual_sha256':__import__('hashlib').sha256(rh.tobytes()).hexdigest() if rh is not None else None})
            inputs=(jax.device_put(ah),jax.device_put(bh)) if rh is None else (jax.device_put(ah),jax.device_put(bh),jax.device_put(rh))
            del ah,bh,rh;gc.collect();jax.block_until_ready(inputs)
            entries={};group_results=[]
            for arm in group['arms']:
                budget.check();key=arm['arm_id']
                try:
                    fn=make_epilogue(arm['algorithm'],tuple(group['shape_mkn']),tuple(arm['tile']),kind=group['kind'],
                        fused=arm.get('fused',False),early=arm.get('early',False),packed=arm.get('packed',False),variant=arm['variant'])
                    compile_start=time.monotonic()
                    prepare=jax.jit(fn.prepare).lower(*inputs).compile()
                    call=jax.jit(fn).lower(*inputs).compile()
                    prepared=prepare(*inputs);jax.block_until_ready(prepared)
                    kernel=jax.jit(fn.prepared).lower(*prepared).compile()
                    entry={'arm':arm,'fn':fn,'prepare':prepare,'call':call,'prepared_kernel':kernel,'prepared':prepared,
                           'samples':{'call':[],'prepared_kernel':[]},'correctness':{}}
                    journal.emit('compile',group_id=group['id'],arm_id=key,compile_seconds=time.monotonic()-compile_start,
                                 metadata=fn.metadata,prepared_shapes=[list(a.shape) for a in prepared])
                    for scope,operands in (('call',inputs),('prepared_kernel',prepared)):
                        output=entry[scope](*operands);output.block_until_ready()
                        entry['correctness'][scope]=correctness(output,reference,rows,cols,campaign['correctness']['gate'])
                        del output
                    entries[key]=entry
                except Exception as problem:
                    row={'group_id':group['id'],'arm_id':key,'status':'compile_or_correctness_error',
                         'error_type':type(problem).__name__,'error':str(problem),'eligible_for_speedup_claim':False}
                    journal.emit('case_result',**row);group_results.append(row)
            timing=campaign['application_timing']
            for scope in ('call','prepared_kernel'):
                eligible_arms=[key for key,entry in entries.items() if entry['correctness'][scope]['passed']]
                for key in eligible_arms:
                    entry=entries[key];operands=inputs if scope=='call' else entry['prepared']
                    for _ in range(timing['warmups']):
                        budget.check();entry[scope](*operands).block_until_ready()
                order=list(eligible_arms);rng.shuffle(order)
                for repeat in range(timing['repeats']):
                    if not order: break
                    for key in order[repeat%len(order):]+order[:repeat%len(order)]:
                        budget.check();entry=entries[key];operands=inputs if scope=='call' else entry['prepared']
                        start=time.perf_counter_ns();output=entry[scope](*operands);output.block_until_ready()
                        elapsed=(time.perf_counter_ns()-start)/1e6;entry['samples'][scope].append(elapsed)
                        journal.emit('timing_sample',group_id=group['id'],arm_id=key,scope=scope,repeat=repeat,elapsed_ms=elapsed)
                        del output
            for key,entry in entries.items():
                # Preparation gets separate measured samples, never attributed to the prepared kernel.
                prep_samples=[]
                for repeat in range(timing['preparation_repeats']):
                    budget.check();start=time.perf_counter_ns();prepared=entry['prepare'](*inputs);jax.block_until_ready(prepared)
                    prep_samples.append((time.perf_counter_ns()-start)/1e6);del prepared
                for scope in ('call','prepared_kernel'):
                    samples=entry['samples'][scope];check=entry['correctness'][scope]
                    passed=check['passed'] and len(samples)==timing['repeats']
                    row={'group_id':group['id'],'arm_id':key,'scope':scope,'status':'ok' if passed else 'failed_correctness',
                         'algorithm':entry['arm']['algorithm'],'variant':entry['arm']['variant'],'tile':entry['arm']['tile'],
                         'correctness':check,'metadata':entry['fn'].metadata,'raw_ms':samples,
                         'mean_ms':float(np.mean(samples)) if samples else None,'median_ms':float(np.median(samples)) if samples else None,
                         'std_ms':float(np.std(samples,ddof=1)) if len(samples)>1 else None,
                         'preparation_raw_ms':prep_samples,'eligible_for_speedup_claim':passed,
                         'input_scope':'synthetic Gaussian, no model-quality qualification'}
                    journal.emit('case_result',**row);group_results.append(row)
            selected[group['id']]={'shape_mkn':group['shape_mkn'],'kind':group['kind']}
            for family in ('native','cubic','strassen'):
                candidates=[row for row in group_results if row.get('scope')=='call' and row.get('eligible_for_speedup_claim')
                            and ('cubic' if row.get('algorithm','').startswith('cubic') else row.get('algorithm'))==family]
                winner=min(candidates,key=lambda row:row['mean_ms']) if candidates else None
                if winner:
                    choice=next(arm for arm in group['arms'] if arm['arm_id']==winner['arm_id'])
                    selected[group['id']][family]={**choice,'mean_ms':winner['mean_ms'],'source_group':group['id']}
                else: selected[group['id']][family]=None
            base.exclusive_json(group_dir/'summary.json',group_results);results.extend(group_results)
            # Remove references held by loop variables as well as the entry map.
            entries.clear();del inputs,reference,entries
            if 'entry' in locals(): del entry
            if 'fn' in locals(): del fn
            if 'prepared' in locals(): del prepared
            gc.collect();jax.clear_caches()
            print(f"N8 [{index+1}/{len(planned)}] {group['id']} complete",flush=True)
        base.exclusive_json(args.output_dir/'application_selections.json',{'phase':'N8','environment_identity':environment['identity'],
            'campaign_sha256':base.digest_file(args.campaign),'by_group':selected,
            'scope':'Performance choices on synthetic Gaussian; require independent N9 model-quality qualification',
            'selection_rule':'Lowest complete-call mean among numerically eligible family members; fixed attempts; no N9 quality data used'})
        completed=True
    except BaseException as problem:
        error={'type':type(problem).__name__,'message':str(problem)};journal.emit('run_error',**error,traceback=traceback.format_exc())
    finally: support.seal(args,journal,started,completed,results,error)
    return 0 if completed else 1


if __name__=='__main__': raise SystemExit(main())
