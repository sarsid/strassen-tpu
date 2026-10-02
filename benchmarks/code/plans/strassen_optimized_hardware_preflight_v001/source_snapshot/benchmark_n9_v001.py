"""N9 real-model resident-layer timings and streamed corpus quality.

Large checkpoint files stay in a frozen external cache; their complete verified
manifest is copied into these results. No full-corpus logits are saved. All
policies score the same 32,736 next-token positions, using their own propagated
states. Residency timings use the same native incoming state for every policy.
The entire native implementation must pass an independent official CPU reference
before model quality proceeds. This is inference/prefill, not generation or task
benchmarking, and the streamed total includes weights/activations transfer costs.
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

QUALIFICATION={'max_relative_l2':0.03,'max_mean_kl':0.01,'max_abs_nll_delta':0.05,'min_top1_agreement':0.90}
QUALITY={'max_abs_nll_delta':0.01,'max_mean_kl':0.02,'min_top1_agreement':0.97}


def metrics(raw):
    count=int(raw['positions'])
    if not count: raise ValueError('No scored model positions')
    native=float(raw['reference_nll_sum'])/count;candidate=float(raw['candidate_nll_sum'])/count
    return {**raw,'reference_nll':native,'candidate_nll':candidate,'nll_delta':candidate-native,
      'reference_perplexity':math.exp(native) if native<700 else None,
      'candidate_perplexity':math.exp(candidate) if candidate<700 else None,
      'mean_kl':float(raw['kl_sum'])/count,'top1_agreement':float(raw['top1_equal'])/count,
      'relative_l2':math.sqrt(float(raw['squared_error_sum'])/max(float(raw['reference_squared_sum']),1e-30))}


def combine(target,item):
    if target is None: return dict(item)
    for key,value in item.items():
        if key=='max_abs_error': target[key]=max(target[key],value)
        elif key=='all_finite': target[key]=target[key] and value
        else: target[key]+=value
    return target


def eligible(result,thresholds):
    return bool(result['all_finite'] and abs(result['nll_delta'])<=thresholds['max_abs_nll_delta']
        and result['mean_kl']<=thresholds['max_mean_kl']
        and result['top1_agreement']>=thresholds['min_top1_agreement']
        and ('max_relative_l2' not in thresholds or result['relative_l2']<=thresholds['max_relative_l2']))


class Study:
    def __init__(self,args,campaign,journal):
        self.args=args;self.campaign=campaign;self.journal=journal
        self.budget=support.Budget(args.max_wall_seconds)

    def record(self,event,**data): self.journal.emit(event,**data)

    def host_metrics(self,reference,candidate,targets):
        value=model.compare_logits(reference,candidate,targets)
        return {key:val.item() for key,val in jax.device_get(value).items()}

    def compile(self,config,length,policy,weights,label):
        self.budget.check();start=time.monotonic()
        fn=model.build_layer(config,length,policy)
        spec=jax.ShapeDtypeStruct((length,config['hidden_size']),jnp.bfloat16)
        executable=fn.lower(spec,weights).compile()
        self.record('compile',arm_id=label,sequence_length=length,compile_seconds=time.monotonic()-start)
        return executable

    def full_forward(self,checkpoint,ids,fn,*,return_hidden=False):
        self.budget.check()
        hidden=jax.device_put(checkpoint.embeddings(ids))
        for index in range(checkpoint.config['num_hidden_layers']):
            self.budget.check()
            host=checkpoint.layer(index);weights=jax.device_put(host)
            hidden=fn(hidden,weights);hidden.block_until_ready()
            del host,weights;gc.collect()
        norm=jax.device_put(np.array(checkpoint.tensor('model.norm.weight'),copy=True))
        weight=jax.device_put(checkpoint.head())
        logits=model.head_logits(hidden,norm,weight,checkpoint.config['rms_norm_eps'])
        logits.block_until_ready()
        if return_hidden:
            normalized=model.rms(hidden,norm,checkpoint.config['rms_norm_eps'])
            normalized.block_until_ready()
        del norm,weight,hidden;gc.collect()
        return (logits,normalized) if return_hidden else logits

    def run_model(self,item,index):
        self.budget.check();directory=self.args.output_dir/f'model-{index:02d}';directory.mkdir()
        if item.get('status')=='blocked_access':
            result={'model_id':item['model_id'],'status':'blocked_access',
                    'reason':item.get('reason','Official weights require unavailable permission'),
                    'eligible_for_speedup_claim':False,'quality_measured':False}
            base.exclusive_json(directory/'model_summary.json',result)
            self.record('case_result',**result);return result
        path=support.resolve(self.args.campaign,item['model_manifest'])
        for name in ('model_manifest','tokens_manifest','reference_manifest'):
            input_path=support.resolve(self.args.campaign,item[name])
            if base.digest_file(input_path)!=item['input_sha256'][name]:
                raise ValueError('Frozen application input manifest hash changed: '+name)
        checkpoint=model.Checkpoint(path);config=checkpoint.config
        if checkpoint.manifest['model_id']!=item['model_id'] or checkpoint.manifest['revision']!=item['revision']:
            raise ValueError('Checkpoint does not match preregistered model/revision')
        manifest_sha=base.digest_file(path)
        base.exclusive_json(directory/'model_manifest.json',checkpoint.manifest)
        tokens_path=support.resolve(self.args.campaign,item['tokens_manifest'])
        token_manifest=json.loads(tokens_path.read_text())
        if token_manifest['model_manifest_sha256']!=manifest_sha or token_manifest['model_revision']!=item['revision']:
            raise ValueError('Tokens do not belong to this exact model manifest')
        ids_path=support.checked_file(tokens_path.parent,token_manifest['token_array_path'],token_manifest['token_array_sha256'])
        ids=np.load(ids_path,allow_pickle=False)
        if ids.shape!=(32,1024) or ids.dtype!=np.dtype('<i4') or ids.min()<0 or ids.max()>=config['vocab_size']:
            raise ValueError('Expected32x1024 valid frozen int32 token IDs')
        shutil.copyfile(ids_path,directory/'token_ids.npy');shutil.copyfile(tokens_path,directory/'tokens_manifest.json')
        reference_path=support.resolve(self.args.campaign,item['reference_manifest'])
        reference_manifest=json.loads(reference_path.read_text())
        if reference_manifest['model_manifest_sha256']!=manifest_sha or reference_manifest['tokens_sha256']!=base.digest_file(ids_path):
            raise ValueError('Independent reference does not use identical model/tokens')
        refdir=directory/'official_reference';refdir.mkdir()
        reference_arrays={}
        for name,row in reference_manifest['files'].items():
            f=support.checked_file(reference_path.parent,row['path'],row['sha256'])
            reference_arrays[name]=np.load(f,allow_pickle=False);shutil.copyfile(f,refdir/f.name)
        shutil.copyfile(reference_path,refdir/'reference_manifest.json')
        if not np.array_equal(reference_arrays['input_ids'],ids.reshape(-1)[:64].reshape(1,64)):
            raise ValueError('Independent reference token positions differ')
        policies={'native':model.native_policy(),**item['policies']}
        if 'native' in item['policies'] or not item['policies']: raise ValueError('Explicit nonnative policies required')
        base.exclusive_json(directory/'policies.json',policies)
        self.record('model_start',model_id=item['model_id'],revision=item['revision'],scored_positions=32736,
                    thresholds={'qualification':QUALIFICATION,'quality':QUALITY},policies=policies)
        print(f"N9 {item['model_id']}: independent native qualification",flush=True)
        weights=jax.device_put(checkpoint.layer(0))
        reference_fn=self.compile(config,64,policies['native'],weights,'native_reference64')
        del weights
        logits,hidden=self.full_forward(checkpoint,ids.reshape(-1)[:64],reference_fn,return_hidden=True)
        raw=self.host_metrics(jax.device_put(reference_arrays['logits'][0,:63]),logits[:63],jax.device_put(ids[0,1:64]))
        qualified=metrics(raw);passed=eligible(qualified,QUALIFICATION)
        qualified.update(passed=passed,thresholds=QUALIFICATION,reference_scope='official Transformers CPU BF16 first64tokens;63scored')
        with (directory/'jax_reference_logits.npy').open('xb') as f: np.save(f,np.asarray(logits,dtype=np.float32),allow_pickle=False)
        with (directory/'jax_reference_final_hidden.npy').open('xb') as f: np.save(f,np.asarray(hidden,dtype=np.float32),allow_pickle=False)
        hidden_ref=reference_arrays['final_hidden'][0];hidden_ours=np.asarray(hidden,dtype=np.float32)
        qualified['final_hidden_relative_l2']=float(np.linalg.norm(hidden_ours-hidden_ref)/max(np.linalg.norm(hidden_ref),1e-30))
        base.exclusive_json(directory/'qualification.json',qualified)
        self.record('native_qualification',model_id=item['model_id'],**qualified)
        del logits,hidden,reference_arrays,reference_fn;gc.collect()
        if not passed:
            result={'model_id':item['model_id'],'status':'failed_native_qualification',
                    'qualification':qualified,'quality_measured':False,'eligible_for_speedup_claim':False}
            base.exclusive_json(directory/'model_summary.json',result);self.record('case_result',**result);return result
        weights=jax.device_put(checkpoint.layer(0));functions={};failures={}
        for arm,policy in policies.items():
            try: functions[arm]=self.compile(config,1024,policy,weights,arm)
            except Exception as error:
                if arm=='native': raise
                failures[arm]={'status':'compile_failed','type':type(error).__name__,'message':str(error)}
                self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm])
        del weights;gc.collect()
        # All arms begin from the same real embeddings; each propagates its own state.
        initial=[jax.device_put(checkpoint.embeddings(window)) for window in ids]
        states={arm:list(initial) for arm in functions};del initial
        resident={arm:[] for arm in functions};rng=np.random.Generator(np.random.PCG64(self.campaign['seed']))
        timing=self.campaign['resident_timing']
        for layer in range(config['num_hidden_layers']):
            self.budget.check();host=checkpoint.layer(layer);weights=jax.device_put(host)
            jax.block_until_ready(weights)
            incoming=states['native'][0]
            for arm in list(functions):
                try:
                    for _ in range(timing['warmups']): functions[arm](incoming,weights).block_until_ready()
                except Exception as error:
                    if arm=='native': raise
                    failures[arm]={'status':'execution_failed','layer':layer,'type':type(error).__name__,'message':str(error)}
                    self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm]);del functions[arm],states[arm]
            order=list(functions);rng.shuffle(order)
            for repeat in range(timing['repeats']):
                for arm in order[repeat%len(order):]+order[:repeat%len(order)]:
                    if arm not in functions: continue
                    self.budget.check()
                    try:
                        start=time.perf_counter_ns();output=functions[arm](incoming,weights);output.block_until_ready()
                        elapsed=(time.perf_counter_ns()-start)/1e6;resident[arm].append(elapsed)
                        self.record('resident_layer_sample',model_id=item['model_id'],layer=layer,arm_id=arm,
                                    repeat=repeat,elapsed_ms=elapsed,scope='device resident full layer, same native incoming first-window state')
                        del output
                    except Exception as problem:
                        if arm=='native': raise
                        failures[arm]={'status':'resident_timing_failed','layer':layer,'repeat':repeat,
                                       'type':type(problem).__name__,'message':str(problem)}
                        self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm]);del functions[arm],states[arm]
            for arm in list(functions):
                try:
                    for window in range(32):
                        self.budget.check();states[arm][window]=functions[arm](states[arm][window],weights)
                        states[arm][window].block_until_ready()
                except Exception as problem:
                    if arm=='native': raise
                    failures[arm]={'status':'quality_propagation_failed','layer':layer,'window':window,
                                   'type':type(problem).__name__,'message':str(problem)}
                    self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm]);del functions[arm],states[arm]
            del incoming,weights,host;gc.collect()
            self.record('quality_layer_complete',model_id=item['model_id'],layer=layer,arms=list(functions))
            print(f"N9 {item['model_id']}: propagated layer {layer+1}/{config['num_hidden_layers']} across32windows",flush=True)
        norm=jax.device_put(np.array(checkpoint.tensor('model.norm.weight'),copy=True));weight=jax.device_put(checkpoint.head())
        totals={arm:None for arm in functions};window_results=[]
        for window in range(32):
            subtotal={arm:None for arm in functions}
            for lo in range(0,1023,128):
                self.budget.check();hi=min(lo+128,1023)
                native=model.head_logits(states['native'][window][lo:hi],norm,weight,config['rms_norm_eps'])
                targets=jax.device_put(ids[window,lo+1:hi+1])
                for arm in list(functions):
                    try:
                        candidate=native if arm=='native' else model.head_logits(states[arm][window][lo:hi],norm,weight,config['rms_norm_eps'])
                        raw=self.host_metrics(native,candidate,targets)
                        subtotal[arm]=combine(subtotal[arm],raw);totals[arm]=combine(totals[arm],raw)
                        del candidate
                    except Exception as problem:
                        if arm=='native': raise
                        failures[arm]={'status':'quality_head_failed','window':window,'block':lo,
                                       'type':type(problem).__name__,'message':str(problem)}
                        self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm])
                        del functions[arm],states[arm],subtotal[arm],totals[arm]
                del native,targets
            row={'window':window,'metrics':{arm:metrics(raw) for arm,raw in subtotal.items()}}
            window_results.append(row);self.record('quality_window',model_id=item['model_id'],**row)
        quality={arm:metrics(raw) for arm,raw in totals.items()}
        for arm,value in quality.items():
            if value['positions']!=32736: raise ValueError('Scored-token count does not match preregistration')
            value.update(passed=eligible(value,QUALITY),thresholds=QUALITY)
        base.exclusive_json(directory/'quality_windows.json',window_results)
        base.exclusive_json(directory/'quality.json',quality)
        del norm,weight,states;gc.collect()
        # Compile/first-use already completed by corpus work. Streamed wall timing
        # includes mmap reads, tensor layout copies, device transfer, every layer,
        # final RMSNorm and the entire 1024-position vocabulary projection.
        streamed={arm:[] for arm in functions};order=list(functions);rng.shuffle(order)
        for repeat in range(self.campaign['streamed_repeats']):
            for arm in order[repeat%len(order):]+order[:repeat%len(order)]:
                if arm in failures: continue
                self.budget.check()
                # Warm the full-width head shape outside the measured call once;
                # chunk128 quality and sequence64 qualification compile other shapes.
                try:
                    if repeat==0:
                        warm=self.full_forward(checkpoint,ids[0],functions[arm]);del warm;gc.collect()
                    start=time.perf_counter_ns();logits=self.full_forward(checkpoint,ids[0],functions[arm]);logits.block_until_ready()
                    elapsed=(time.perf_counter_ns()-start)/1e6;streamed[arm].append(elapsed)
                    self.record('streamed_forward_sample',model_id=item['model_id'],arm_id=arm,repeat=repeat,elapsed_ms=elapsed,
                                input_window=0,sequence_length=1024,scope='single-window full-forward with host/disk/layout/device transfers; excludes download/tokenization/compile')
                    del logits;gc.collect()
                except Exception as problem:
                    if arm=='native': raise
                    failures[arm]={'status':'streamed_forward_failed','repeat':repeat,
                                   'type':type(problem).__name__,'message':str(problem)}
                    self.record('policy_error',model_id=item['model_id'],arm_id=arm,**failures[arm])
        result={'model_id':item['model_id'],'revision':item['revision'],'status':'completed','quality_measured':True,
                'qualification':qualified,'quality':quality,'failures':failures,
                'resident_layer_mean_ms':{arm:float(np.mean(resident[arm])) for arm in functions
                    if len(resident[arm])==config['num_hidden_layers']*timing['repeats']},
                'resident_sample_coverage':{arm:{'observed':len(values),'expected':config['num_hidden_layers']*timing['repeats'],
                    'complete':len(values)==config['num_hidden_layers']*timing['repeats']} for arm,values in resident.items()},
                'streamed_forward_samples_ms':streamed,
                'streamed_forward_mean_ms':{arm:float(np.mean(values)) for arm,values in streamed.items()
                    if len(values)==self.campaign['streamed_repeats']},
                'streamed_descriptive_statistics':{arm:{'count':len(values),'mean_ms':float(np.mean(values)),
                    'std_ms':float(np.std(values,ddof=1)) if len(values)>1 else None,
                    'min_ms':float(np.min(values)),'max_ms':float(np.max(values))} for arm,values in streamed.items() if values},
                'uncertainty_scope':'Three paired streamed repeats are descriptive; resident samples are paired by layer and repeat in results.jsonl. Layers are not independent statistical replicates.',
                'eligible_for_speedup_claim':{arm:arm in quality and quality[arm]['passed'] and arm not in failures
                    and len(streamed.get(arm,[]))==self.campaign['streamed_repeats'] for arm in policies},
                'scope':'Teacher-forced full-model prefill quality; resident-layer and transfer-inclusive streamed full-forward performance; no generation/task capability claim'}
        base.exclusive_json(directory/'model_summary.json',result)
        for arm,value in quality.items():
            self.record('case_result',model_id=item['model_id'],arm_id=arm,
                        status=failures[arm]['status'] if arm in failures else ('ok' if value['passed'] else 'failed_quality'),
                        quality=value,eligible_for_speedup_claim=result['eligible_for_speedup_claim'][arm])
        for arm,failure in failures.items():
            if arm not in quality:
                self.record('case_result',model_id=item['model_id'],arm_id=arm,status=failure['status'],
                            error=failure,quality=None,eligible_for_speedup_claim=False)
        del functions,checkpoint;gc.collect();jax.clear_caches()
        return result


def main(argv=None):
    args=support.parser('N9').parse_args(argv);started=time.monotonic()
    campaign=support.begin(args);journal=base.Journal(args.output_dir,'N9');results=[];completed=False;error=None
    try:
        environment=support.initialize_device(args,campaign,journal)
        if campaign.get('selection_environment_identity')!=environment['identity']:
            raise ValueError('Frozen N8 policy-selection environment differs from current runtime')
        global jax,jnp,np,model
        import jax
        import jax.numpy as jnp
        import numpy as np
        from . import model_n9_v001 as model
        if campaign.get('qualification_thresholds')!=QUALIFICATION or campaign.get('quality_thresholds')!=QUALITY:
            raise ValueError('Quality thresholds must match preregistered constants')
        base.exclusive_json(args.output_dir/'planned_cases.json',campaign['models'])
        study=Study(args,campaign,journal)
        for index,item in enumerate(campaign['models']):
            try: results.append(study.run_model(item,index))
            except Exception as problem:
                result={'model_id':item['model_id'],'status':'failed','error_type':type(problem).__name__,
                        'error':str(problem),'eligible_for_speedup_claim':False}
                journal.emit('model_error',**result,traceback=traceback.format_exc());results.append(result)
                gc.collect();jax.clear_caches()
        completed=all(row['status'] in ('completed','blocked_access','failed_native_qualification') for row in results)
    except BaseException as problem:
        error={'type':type(problem).__name__,'message':str(problem)};journal.emit('run_error',**error,traceback=traceback.format_exc())
    finally: support.seal(args,journal,started,completed,results,error)
    return 0 if completed else 1


if __name__=='__main__': raise SystemExit(main())
