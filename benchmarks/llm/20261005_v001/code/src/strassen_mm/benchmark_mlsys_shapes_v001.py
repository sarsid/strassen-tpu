"""Five-method development campaign: independent tuning and fresh confirmation.

This module only reuses previously executed kernels. Native tuning is scoped
compiler VMEM tuning; custom tuning searches explicit BM,BN,BK tuples. A family
without a numerically eligible candidate remains visible and cannot claim a win.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from strassen_mm import benchmark_v001 as base

FAMILIES = ('native','cubic','one_level','two_level')
ALIGNMENTS = {'cubic':(8,128,128),'one_level':(16,256,256),'two_level':(32,512,512)}


def candidates(campaign):
    result = []
    for family in FAMILIES:
        for candidate in campaign['candidate_families'][family]:
            result.append(dict(copy.deepcopy(candidate), family=family, arm_id=candidate['candidate_id']))
    return result


def validate_campaign(campaign, shapes):
    registered = candidates(campaign)
    if list(campaign['arms']) != list(FAMILIES):
        raise ValueError('The four tuned families must remain explicit')
    if len({a['candidate_id'] for a in registered}) != len(registered):
        raise ValueError('Duplicate candidate IDs')
    if len({len(campaign['candidate_families'][f]) for f in FAMILIES[1:]}) != 1:
        raise ValueError('Custom families must have equal candidate attempt budgets')
    for arm in registered:
        if arm['family'] == 'native':
            if arm['algorithm'] != 'native' or arm['variant'] != 'plain' or arm['tile'] is not None:
                raise ValueError('Native tiles are compiler-managed')
            if set(arm['compiler_options']) - {'xla_tpu_scoped_vmem_limit_kib'}:
                raise ValueError('Unregistered Native option')
        else:
            tile = arm['tile']
            if len(tile) != 3 or any(type(v) is not int or v <= 0 or v % a for v,a in zip(tile,ALIGNMENTS[arm['family']])):
                raise ValueError('Invalid tile alignment for '+arm['candidate_id'])
            if arm['compiler_options']:
                raise ValueError('Custom compiler options must remain frozen')
    default = [a for a in registered if a['family']=='native' and not a['compiler_options']]
    if len(default) != 1 or default[0]['candidate_id'] != 'native_default':
        raise ValueError('Exactly one named default Native candidate is required')
    coords = [tuple(s[d] for d in ('m','k','n')) for s in shapes]
    if len(coords) != len(set(coords)) or len({s['id'] for s in shapes}) != len(shapes):
        raise ValueError('Shape IDs and actual M,K,N must both be unique')
    if any(type(v) is not int or v <= 0 for row in coords for v in row):
        raise ValueError('Positive dimensions required')
    policy = campaign['selection_policy']
    if not 1 <= policy['top_k'] <= policy['max_confirm_per_family']:
        raise ValueError('Invalid finalist cap')
    return {family:len(campaign['candidate_families'][family]) for family in FAMILIES}


def make_function(arm, shape):
    from strassen_mm import kernels_v002 as kernels
    if arm['family'] == 'two_level':
        from strassen_mm.kernels_two_level_v001 import make_matmul
        return make_matmul(shape, tuple(arm['tile']), vmem_limit_bytes=48*1024**2)
    return kernels.make_matmul(arm['algorithm'], shape, tuple(arm['tile']) if arm['tile'] else None,
        variant=arm['variant'], vmem_limit_bytes=None if arm['family']=='native' else 48*1024**2)


def install_reference_policy(campaign):
    """Use FP64 over every K, caching only the current exact input objects."""
    cache = {}
    def reference(a, b, cfg, seed):
        key = (id(a), id(b), seed)
        if cache.get('key') == key and cache.get('a') is a and cache.get('b') is b:
            return cache['value']
        np = base.np
        m,k = a.shape; n = b.shape[1]
        full = (m*n <= cfg['memory']['max_reference_output_elements'] and
                2*m*k*n <= cfg['memory']['max_full_reference_flops'])
        rng = np.random.Generator(np.random.PCG64(seed+911))
        rows = np.arange(m,dtype=np.int32) if full else base.sample_indices(m,cfg['correctness']['sample_rows'],rng)
        cols = np.arange(n,dtype=np.int32) if full else base.sample_indices(n,cfg['correctness']['sample_columns'],rng)
        ar = np.asarray(a[rows,:],dtype=np.float64); bc = np.asarray(b[:,cols],dtype=np.float64)
        ref = ar @ bc
        info = dict(reference_scope='full' if full else 'sampled_cross_product', reference_backend='host_numpy_fp64',
            reference_input_dtype='quantized_bf16', reference_accumulation_dtype='float64', all_k_used=k,
            sample_rows=rows.tolist(), sample_columns=cols.tolist(), sample_count=int(ref.size),
            normwise_denominator=float(np.linalg.norm(ar)*np.linalg.norm(bc)),
            metric_scope='Max error and error norms cover reference sample; finiteness covers complete output')
        result = ref,rows,cols,info
        cache.clear(); cache.update(key=key,a=a,b=b,value=result)
        return result
    base.make_reference = reference
    return cache


def timing_valid(row, repeats):
    timing = row.get('timing') or {}
    value = timing.get('mean_ms')
    return (row.get('scope') == 'call' and timing.get('sample_count') == repeats and
            isinstance(value,(int,float)) and math.isfinite(value) and value > 0)


def select_candidates(cases, campaign, shapes, samples=None):
    registered = {a['candidate_id']:a for a in candidates(campaign)}
    repeats = campaign['timing']['screen']['repeats']; policy = campaign['selection_policy']
    selected = {}
    for shape in shapes:
        row = {}; selected[shape['id']] = row
        for family in FAMILIES:
            allowed = [a for a in registered.values() if a['family']==family]
            observed = [r for r in cases if r.get('shape_id')==shape['id'] and r.get('arm_id') in {a['arm_id']for a in allowed}]
            by_id = {r['arm_id']:r for r in observed}
            if len(observed) != len(by_id):
                raise ValueError('Duplicate screening outcomes: '+shape['id']+'/'+family)
            if set(by_id) != {a['arm_id'] for a in allowed}:
                raise ValueError('Incomplete candidate screen: '+shape['id']+'/'+family)
            valid = [r for r in observed if timing_valid(r,repeats) and r['status']=='ok' and r['eligible_for_speedup_claim']]
            valid.sort(key=lambda r:(r['timing']['mean_ms'],r['arm_id']))
            fastest = valid[0] if valid else None
            eligible = fastest is not None
            if fastest is None:
                fallback = [r for r in observed if timing_valid(r,repeats) and (r.get('correctness') or {}).get('finite')]
                fallback.sort(key=lambda r:(r['timing']['mean_ms'],r['arm_id']))
                fastest = fallback[0] if fallback else None
            desired = []
            if valid:
                boundary = valid[0]['timing']['mean_ms']*(1+policy['near_fraction'])
                desired = [r for i,r in enumerate(valid) if i<policy['top_k'] or r['timing']['mean_ms']<=boundary]
            elif fastest:
                desired = [fastest]
            chosen = desired[:policy['max_confirm_per_family']]
            row[family] = dict(winner=registered[fastest['arm_id']] if fastest else None,
                eligible=eligible, confirmation_candidates=[registered[r['arm_id']] for r in chosen],
                omitted_by_cap=[registered[r['arm_id']] for r in desired[len(chosen):]],
                screen_mean_ms=fastest['timing']['mean_ms'] if fastest else None,
                screen_group_id=fastest['group_id'] if fastest else None,
                fallback_reason=None if eligible else 'No eligible candidate: retain fastest finite measured candidate for accuracy reporting only, or no candidate if none executed.',
                attempts=[dict(candidate_id=a['candidate_id'],status=by_id[a['arm_id']]['status'],
                    eligible=bool(by_id[a['arm_id']].get('eligible_for_speedup_claim')),
                    timing=by_id[a['arm_id']].get('timing'),correctness=by_id[a['arm_id']].get('correctness')) for a in allowed],
                uncertainty_scope='Screening top/near candidates are a shortlist, not a confidence interval on dimensions. Fresh confirmation supplies paired latency intervals. Search and shortlist are bounded.')
    return selected


def plan_groups(campaign, shapes, stage, selection=None, start=0, count=None):
    validate_campaign(campaign,shapes)
    if stage not in ('smoke','screen','confirm'):
        raise ValueError('Unknown stage')
    if start < 0 or count is not None and count <= 0 or start >= len(shapes):
        raise ValueError('Invalid shape slice')
    chosen = list(enumerate(shapes))[start:None if count is None else start+count]
    if stage == 'smoke':
        chosen = [(i,s)for i,s in chosen if s['id'] in campaign['smoke_shape_ids']]
    registered = candidates(campaign); groups=[]
    default = next(a for a in registered if a['candidate_id']=='native_default')
    for index,shape in chosen:
        if stage == 'screen':
            shuffled = {f:[a for a in registered if a['family']==f] for f in FAMILIES}
            rng = random.Random(campaign['order_seed']+index)
            for arms in shuffled.values():rng.shuffle(arms)
            batches = [[shuffled[f][j] for f in FAMILIES if j<len(shuffled[f])]
                       for j in range(max(map(len,shuffled.values())))]
            inputs = [dict(distribution='gaussian',seed=campaign['screen_seed']+index)]
        elif stage == 'smoke':
            arms = [default] + [next(a for a in registered if a['family']==f and a['tile']==[1024,1024,512]) for f in FAMILIES[1:]]
            batches = [arms];inputs=[dict(distribution='gaussian',seed=campaign['smoke_seed']+index)]
        else:
            if selection is None or shape['id'] not in selection:
                raise ValueError('Missing selection for '+shape['id'])
            arms = [a for f in FAMILIES for a in selection[shape['id']][f]['confirmation_candidates']]
            if default not in arms:arms.append(default)
            if any(a not in registered for a in arms) or len({a['arm_id']for a in arms})!=len(arms):
                raise ValueError('Invalid frozen finalist inventory')
            batches=[arms]; inputs=[dict(distribution='gaussian',seed=s+index) for s in campaign['confirm_seeds']]
        for j,arms in enumerate(batches):
            group = dict(group_id=shape['id']+'__'+stage+'__'+str(j),shape=shape,tile=None,arms=copy.deepcopy(arms),
                inputs=copy.deepcopy(inputs),timing=stage,scopes=['call'],manifest_index=index)
            if stage=='confirm':
                group['headline_candidates']={f:(selection[shape['id']][f]['winner'] or {}).get('candidate_id') for f in FAMILIES}
                group['headline_candidates']['native_default']='native_default'
            groups.append(group)
    return groups


make_groups = plan_groups


class Journal(base.Journal):
    def __init__(self,*args):
        super().__init__(*args); self.cases=[]; self.samples={}; self.groups={}
    def emit(self,event,**fields):
        group=self.groups.get(fields.get('group_id'))
        if group:
            fields.setdefault('shape_id',group['shape']['id'])
            fields.setdefault('shape_mkn',[group['shape'][d]for d in ('m','k','n')])
            arm=next((a for a in group['arms']if a['arm_id']==fields.get('arm_id')),None)
            if arm:
                for key in ('candidate_id','family','tile','compiler_options'):fields.setdefault(key,arm.get(key))
                if group.get('headline_candidates'):
                    fields.setdefault('headline_roles',[label for label,value in group['headline_candidates'].items() if value==arm['candidate_id']])
        super().emit(event,**fields)
        if event=='case_result':self.cases.append(base.json_safe(fields))
        if event=='sample':
            key=(fields['group_id'],fields['arm_id'],fields['seed'])
            self.samples.setdefault(key,[]).append(dict(round=fields['round'],elapsed_ms=fields['elapsed_ms']))


class Runner(base.Runner):
    def __init__(self,*args):
        super().__init__(*args);self.cached_inputs=None
    def compile_entries(self,group,a,b):
        import jax
        shape=tuple(group['shape'][d]for d in ('m','k','n')); entries=[]
        for arm in group['arms']:
            self.check_deadline();context=dict(group_id=group['group_id'],arm_id=arm['arm_id'],scope='call')
            entry={**context,**arm}; started=time.perf_counter_ns()
            print('  compiling '+arm['arm_id'],flush=True)
            try:
                fn=make_function(arm,shape)
                metadata={**fn.metadata,'compiler_options':arm['compiler_options'],'candidate_id':arm['candidate_id'],'family':arm['family']}
                entry.update(fn=fn,metadata=metadata)
                lowered=jax.jit(fn).lower(jax.ShapeDtypeStruct(a.shape,a.dtype),jax.ShapeDtypeStruct(b.shape,b.dtype))
                executable=lowered.compile(compiler_options=arm['compiler_options']) if arm['compiler_options'] else lowered.compile()
                memory=executable.memory_analysis()
                memory={k:getattr(memory,k,None)for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                keys=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')
                total=sum(memory[k]for k in keys)if all(isinstance(memory[k],int)for k in keys)else None
                self.journal.emit('compilation',**context,status='ok',compile_ms=(time.perf_counter_ns()-started)/1e6,
                    kernel_metadata=metadata,memory_analysis=memory,estimated_executable_live_bytes=total)
                if total is not None and total>self.campaign['memory']['estimated_live_device_budget_gib']*1024**3:
                    entry.update(error='skipped_memory_preflight',error_message='Compiled executable exceeds frozen device memory budget');del executable
                else:entry['executable']=executable
            except Exception as error:
                unsupported=bool(arm['compiler_options'])and any(s in str(error).lower()for s in ('unknown','unrecognized','unsupported','not supported','no such'))
                entry.update(error='unsupported_compiler_option'if unsupported else base.error_status(error),error_message=str(error))
                self.emit_error(context,error,'compile')
            entries.append(entry)
        return entries
    def run_group(self,group):
        import jax
        shape=tuple(group['shape'][d]for d in ('m','k','n'));entries=None
        for case in group['inputs']:
            self.check_deadline();key=(shape,case['distribution'],case['seed'])
            try:
                if self.cached_inputs is None or self.cached_inputs[0]!=key:
                    self.cached_inputs=None;gc.collect()
                    self.cached_inputs=(key,*base.generate_inputs(shape,**case))
                _,a,b=self.cached_inputs
            except Exception as error:
                self.emit_error(dict(group_id=group['group_id'],**case),error,'input_generation')
                for arm in group['arms']:
                    self.journal.emit('case_result',group_id=group['group_id'],**case,arm_id=arm['arm_id'],scope='call',
                        status='input_error',eligible_for_speedup_claim=False,error_message=str(error))
                continue
            if entries is None:entries=self.compile_entries(group,a,b)
            self.execute_case(group,case,entries,a,b)
        del entries
        jax.clear_caches();gc.collect()


def comparison_interval(journal,group,reference_id,candidate_id,policy):
    """Hierarchical paired bootstrap over fresh inputs and rounds within input."""
    np=base.np; seeds=[i['seed']for i in group['inputs']];left=[];right=[]
    for seed in seeds:
        a=journal.samples.get((group['group_id'],reference_id,seed),[])
        b=journal.samples.get((group['group_id'],candidate_id,seed),[])
        aa={s['round']:s['elapsed_ms']for s in a};bb={s['round']:s['elapsed_ms']for s in b}
        if not aa or aa.keys()!=bb.keys():return None
        keys=sorted(aa);left.append([aa[k]for k in keys]);right.append([bb[k]for k in keys])
    left=np.asarray(left);right=np.asarray(right)
    token=group['group_id']+'|'+reference_id+'|'+candidate_id
    rng=np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:16],16))
    repetitions=policy['bootstrap_replicates'];estimates=[]
    for _ in range(repetitions):
        draw_seeds=rng.integers(0,len(seeds),len(seeds));rounds=rng.integers(0,left.shape[1],left.shape)
        aa=left[draw_seeds[:,None],rounds];bb=right[draw_seeds[:,None],rounds]
        estimates.append(float(bb.mean()/aa.mean()))
    low,high=np.quantile(estimates,[.025,.975])
    return dict(candidate_over_reference_latency_ratio=float(right.mean()/left.mean()),
        ci95=[float(low),float(high)],reference_mean_ms=float(left.mean()),candidate_mean_ms=float(right.mean()),
        fresh_seed_count=len(seeds),rounds_per_seed=int(left.shape[1]),bootstrap_replicates=repetitions,
        interval_method='Hierarchical paired bootstrap: resample input seeds, then paired rounds within seeds; pointwise 95% intervals conditional on frozen screen choices, not simultaneous confidence over the entire search.')


def summarize_confirmation(journal,groups,selected,campaign):
    summaries={}
    for group in groups:
        shape_id=group['shape']['id'];summary={'headline':{},'tile_uncertainty':{}};summaries[shape_id]=summary
        native=selected[shape_id]['native']['winner']
        reference_id=native['arm_id']if native else 'native_default'
        reference_cases=[r for r in journal.cases if r.get('group_id')==group['group_id']and r.get('arm_id')==reference_id]
        reference_valid=len(reference_cases)==len(group['inputs'])and all(r.get('eligible_for_speedup_claim')and timing_valid(r,campaign['timing']['confirm']['repeats'])for r in reference_cases)
        reference_valid=bool(reference_valid and selected[shape_id]['native']['eligible'])
        for label,arm_id in group['headline_candidates'].items():
            if arm_id is None:
                summary['headline'][label]={'available':False};continue
            outcomes=[r for r in journal.cases if r.get('group_id')==group['group_id']and r.get('arm_id')==arm_id]
            valid=len(outcomes)==len(group['inputs'])and all(r.get('eligible_for_speedup_claim')and timing_valid(r,campaign['timing']['confirm']['repeats'])for r in outcomes)
            screen_eligible=True if label=='native_default'else selected[shape_id][label]['eligible']
            summary['headline'][label]=dict(available=True,candidate_id=arm_id,all_inputs_eligible=valid,
                screen_selection_eligible=bool(screen_eligible),
                valid_numerical_comparison_vs_tuned_native=bool(valid and reference_valid and screen_eligible),
                correctness=[dict(seed=r['seed'],status=r['status'],metrics=r.get('correctness'))for r in outcomes],
                comparison_vs_tuned_native=comparison_interval(journal,group,reference_id,arm_id,campaign['selection_policy']))
        for family in FAMILIES:
            record=selected[shape_id][family];winner=record['winner'];rows=[]
            if winner:
                for arm in record['confirmation_candidates']:
                    interval=comparison_interval(journal,group,winner['arm_id'],arm['arm_id'],campaign['selection_policy'])
                    outcomes=[r for r in journal.cases if r.get('group_id')==group['group_id']and r.get('arm_id')==arm['arm_id']]
                    rows.append(dict(candidate_id=arm['candidate_id'],tile=arm['tile'],comparison_to_frozen_winner=interval,
                        all_inputs_eligible=len(outcomes)==len(group['inputs'])and all(r.get('eligible_for_speedup_claim')for r in outcomes),
                        consistent_with_equal_latency=bool(interval and interval['ci95'][0]<=1<=interval['ci95'][1]),
                        within_five_percent_upper_bound=bool(interval and interval['ci95'][1]<=1.05)))
            summary['tile_uncertainty'][family]=dict(frozen_winner=winner,confirmed_candidates=rows,
                omitted_by_cap=record['omitted_by_cap'],scope='Discrete confirmed tuples only. Intervals do not certify unsampled tiles, Cartesian ranges, global optimality, or post-confirmation reselection.')
    return dict(schema_version=1,by_shape=summaries)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--expected-identity',type=Path,required=True)
    parser.add_argument('--allocation-id',required=True);parser.add_argument('--selection',type=Path)
    parser.add_argument('--max-wall-seconds',type=float,default=36000)
    parser.add_argument('--shape-start',type=int,default=0);parser.add_argument('--shape-count',type=int)
    args=parser.parse_args(argv);stage=args.phase.rsplit('-',1)[-1]
    cfg=json.loads(args.campaign.read_text());manifest_path=args.campaign.parent/cfg['shape_manifest']
    manifest=json.loads(manifest_path.read_text());shapes=manifest['shapes'];validate_campaign(cfg,shapes)
    selected=None;selection_record=None
    if stage=='confirm':
        if args.selection is None:raise ValueError('Confirmation requires frozen selection')
        selection_record=json.loads(args.selection.read_text())
        if (selection_record['allocation_id']!=args.allocation_id or selection_record['campaign_sha256']!=base.digest_file(args.campaign)
                or selection_record['shape_manifest_sha256']!=base.digest_file(manifest_path)):
            raise ValueError('Selection configuration, inventory, or allocation mismatch')
        selected=selection_record['selected']
    groups=plan_groups(cfg,shapes,stage,selected,args.shape_start,args.shape_count)
    if not groups:raise ValueError('No planned groups')
    args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir;journal=Journal(out,args.phase)
    journal.groups={g['group_id']:g for g in groups};done=[];error=None
    try:
        base.snapshot_sources(out,args.campaign,manifest_path,args.campaign.parent/cfg['distribution_manifest'])
        base.exclusive_json(out/'planned_cases.json',groups)
        base.exclusive_json(out/'plan_counts.json',dict(groups=len(groups),shape_count=len({g['shape']['id']for g in groups}),
            outcomes=sum(len(g['arms'])*len(g['inputs'])for g in groups),shape_start=args.shape_start,shape_count_requested=args.shape_count))
        if selection_record:base.exclusive_json(out/'selection_used.json',selection_record)
        if 'jax'in sys.modules:raise RuntimeError('Expected fresh process')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from strassen_mm.kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat());base.exclusive_json(out/'environment.json',env)
        journal.emit('identity_check',**base.verify_identity(env,args.expected_identity,cfg))
        if selection_record and selection_record['environment_identity']!=env['identity']:
            raise ValueError('Full environment identity changed between screen and confirmation')
        reference_cache=install_reference_policy(cfg);runner=Runner(args,cfg,journal)
        for i,group in enumerate(groups):
            print(f'[{i+1}/{len(groups)}] {group["group_id"]} starting',flush=True)
            journal.emit('group_start',group_id=group['group_id'],index=i+1,total=len(groups))
            runner.run_group(group);done.append(group['group_id'])
            journal.emit('group_complete',group_id=group['group_id'],index=i+1,total=len(groups))
            print(f'[{i+1}/{len(groups)}] complete; {dict(journal.status_counts)}',flush=True)
        if stage=='screen':
            chosen=[s for s in shapes if s['id']in {g['shape']['id']for g in groups}]
            base.exclusive_json(out/'selections.json',dict(allocation_id=args.allocation_id,
                campaign_sha256=base.digest_file(args.campaign),shape_manifest_sha256=base.digest_file(manifest_path),
                environment_identity=env['identity'],rule=cfg['headline_rule'],
                shape_start=args.shape_start,shape_count=len(chosen),selected=select_candidates(journal.cases,cfg,chosen)))
        if stage=='confirm':base.exclusive_json(out/'confirmation_statistics.json',summarize_confirmation(journal,groups,selected,cfg))
        reference_cache.clear()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc());print(error,flush=True)
    summary=dict(completed=error is None and len(done)==len(groups),completed_groups=done,planned_group_count=len(groups),
        planned_outcomes=sum(len(g['arms'])*len(g['inputs'])for g in groups),observed_outcomes=len(journal.cases),
        case_status_counts=dict(journal.status_counts),error=error,stage=stage,
        interpretation='Five-method bounded independent tuning; complete-call timing, exact BF16 inputs, all-K FP64 reference, frozen screening winner and fresh confirmation. Development shapes, not untouched selector validation; synthetic operands, not actual model execution.')
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed']else 1


if __name__=='__main__':raise SystemExit(main())
