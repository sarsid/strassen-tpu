"""Five-family real-operand tuning, resident-layer and streamed model inference.

Original N9 model sizes are explicit: Qwen3-0.6B, Mistral7B, Gemma3-1B.
No random matrix or larger shape-only result is labelled actual model execution.
Only MLP gate/up and down vary; attention and model boundaries remain Native.
"""
import argparse
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
from . import benchmark_v001 as base
from .benchmark_two_level_tune_v001 import Journal as PriorJournal


class Journal(PriorJournal):
    def __init__(self,*args):
        super().__init__(*args);self.samples={};self.groups={}
    def emit(self,event,**fields):
        super().emit(event,**fields)
        if event=='sample':
            key=(fields['group_id'],fields['arm_id'],fields['seed'])
            self.samples.setdefault(key,[]).append(dict(round=fields['round'],elapsed_ms=fields['elapsed_ms']))


def vector_metrics(ref,candidate):
    ref=np.asarray(ref,dtype=np.float64);candidate=np.asarray(candidate,dtype=np.float64)
    error=candidate-ref
    return dict(relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(ref),1e-30)),
        max_abs_error=float(np.max(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
        mean_abs_error=float(np.mean(np.abs(error))),finite=bool(np.isfinite(candidate).all()))


def timing_ci(values,seed):
    values=np.asarray(values,np.float64)
    if not len(values):return dict(sample_count=0)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return dict(sample_count=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),
        latency_ci95_ms=np.quantile(means,[.025,.975]).tolist(),
        uncertainty_scope='Descriptive bootstrap of individual timing rounds; not independent hardware sessions.')


class Study:
    def __init__(self,args,cfg,journal):
        self.args,self.cfg,self.journal=args,cfg,journal
        self.deadline=time.monotonic()+args.max_wall_seconds
        self.prefixes={};self.layers={};self.full_jit_cache={}

    def check(self):
        if time.monotonic()>self.deadline:raise TimeoutError('Frozen application wall budget exhausted')

    def emit(self,event,**row):self.journal.emit(event,**row)

    def mm(self,arm,shape):
        self.check();fn=grid.make_function(arm,shape)
        specs=[jax.ShapeDtypeStruct(s,jnp.bfloat16) for s in ((shape[0],shape[1]),(shape[1],shape[2]))]
        start=time.monotonic();lowered=jax.jit(fn).lower(*specs)
        executable=lowered.compile(compiler_options=arm.get('compiler_options',{}))
        self.emit('model_projection_compile',candidate_id=arm['candidate_id'],shape_mkn=shape,
                  compile_seconds=time.monotonic()-start,kernel_metadata=fn.metadata)
        return executable

    def layer(self,cp,length,policy,index=0):
        c=cp.config;kind=(gm.validate_config(c)['layer_types'][index] if c['model_type']=='gemma3_text' else 'full_attention')
        key=(length,kind,json.dumps(policy,sort_keys=True))
        if key not in self.layers:
            nativekey=(length,kind)
            if nativekey not in self.prefixes:self.prefixes[nativekey]=composed.prefix(c,length,index)
            h,inner=c['hidden_size'],c['intermediate_size']
            self.layers[key]=composed.Layer(self.prefixes[nativekey],self.mm(policy['gateup'],(length,h,2*inner)),
                composed.activation(c),self.mm(policy['down'],(length,inner,h)),composed.suffix(c))
        return self.layers[key]

    def forward(self,cp,ids,policy,*,logits=True):
        self.check();hidden=jax.device_put(cp.embeddings(ids))
        for index in range(cp.config['num_hidden_layers']):
            self.check();host=cp.layer(index);weights=jax.device_put(host)
            hidden=self.layer(cp,len(ids),policy,index)(hidden,weights);hidden.block_until_ready()
            del host,weights;gc.collect()
        if not logits:return hidden,None
        norm=jax.device_put(np.array(cp.tensor('model.norm.weight'),copy=True))
        normalized=composed.adapter(cp.config).rms(hidden,norm,cp.config['rms_norm_eps'])
        normalized.block_until_ready()
        weight=jax.device_put(cp.head())
        output=composed.head(cp,hidden,norm,weight);output.block_until_ready()
        del norm,weight,hidden;gc.collect()
        return output,normalized

    def full_jit(self,cp,length,index):
        mod=composed.adapter(cp.config)
        kind=gm.validate_config(cp.config)['layer_types'][index] if mod is gm else 'full_attention'
        key=(length,kind)
        if key not in self.full_jit_cache:
            self.full_jit_cache[key]=(mod.build_layer(cp.config,length,mod.native_policy(),layer_index=index) if mod is gm else
                mod.build_layer(cp.config,length,mod.native_policy()))
        return self.full_jit_cache[key]

    def forward_native_full_jit(self,cp,ids):
        """Additional unfactored default-Native control; same streamed I/O loop."""
        self.check();hidden=jax.device_put(cp.embeddings(ids));mod=composed.adapter(cp.config)
        for index in range(cp.config['num_hidden_layers']):
            self.check();weights=jax.device_put(cp.layer(index))
            fn=self.full_jit(cp,len(ids),index)
            hidden=fn(hidden,weights);hidden.block_until_ready()
            del weights,fn;gc.collect()
        norm=jax.device_put(np.array(cp.tensor('model.norm.weight'),copy=True));weight=jax.device_put(cp.head())
        for lo in range(0,len(ids),128):
            value=composed.head(cp,hidden[lo:lo+128],norm,weight);value.block_until_ready();del value
        del norm,weight,hidden;gc.collect()

    def qualify(self,cp,ids,reference,native):
        from .benchmark_n9_v001 import metrics,eligible,QUALIFICATION
        official={}
        for name,row in reference['files'].items():
            path=self.reference_dir/row['path']
            if base.digest_file(path)!=row['sha256']:raise ValueError('Official reference array hash mismatch')
            official[name]=np.load(path,allow_pickle=False)
        if not np.array_equal(official['input_ids'].reshape(-1),ids.reshape(-1)[:64]):
            raise ValueError('Official reference does not use identical qualification tokens')
        logits,hidden=self.forward(cp,ids.reshape(-1)[:64],native)
        raw={k:v.item() for k,v in jax.device_get(qm.compare_logits(jax.device_put(official['logits'][0,:63]),
            logits[:63],jax.device_put(ids[0,1:64]))).items()}
        row=metrics(raw);row['hidden']=vector_metrics(official['final_hidden'][0],np.asarray(hidden))
        row['thresholds']=QUALIFICATION;row['passed']=eligible(row,QUALIFICATION)
        if cp.config['model_type']=='gemma3_text':
            row['gemma_final_hidden_limit']=.02
            row['passed']=bool(row['passed'] and row['hidden']['finite'] and row['hidden']['relative_l2']<=.02)
        self.emit('native_qualification',model_id=cp.manifest['model_id'],**row)
        base.exclusive_json(self.args.output_dir/'qualification.json',row)
        del logits,hidden,official;gc.collect()
        return row

    def real_operands(self,cp,ids,native):
        weights=jax.device_put(cp.layer(0));gatefn=self.layer(cp,len(ids[0]),native)
        result=[]
        for window in self.cfg['operand_windows']:
            x=jax.device_put(cp.embeddings(ids[window]))
            residual,normalized=gatefn.prefix(x,weights)
            inner=gatefn.activate(gatefn.gateup(normalized,weights['gateup']))
            jax.block_until_ready((normalized,inner))
            result.append(dict(gateup=np.asarray(normalized),down=np.asarray(inner)))
            self.emit('real_operand_capture',model_id=cp.manifest['model_id'],layer=0,window=window,
                source='Native actual checkpoint attention and MLP propagation from corpus tokens',
                activation_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in result[-1].items()})
            del x,residual,normalized,inner
        host=cp.layer(0)
        del weights,gatefn;gc.collect()
        return result,{k:host[k] for k in ('gateup','down')}

    def tune(self,cp,operands,weights):
        cfg=self.cfg;selected={};policies={}
        runner=grid.Runner(self.args,cfg,self.journal)
        registered=grid.candidates(cfg)
        for projection in ('gateup','down'):
            a,b=operands[0][projection],weights[projection]
            shape=dict(id=self.model_key+'_'+projection+'_m1024',m=a.shape[0],k=a.shape[1],n=b.shape[1])
            expected=next(x for x in json.loads((self.args.campaign.parent/cfg['shape_manifest']).read_text())['shapes'] if x['id']==shape['id'])
            if any(expected[k]!=shape[k] for k in ('m','k','n')):raise ValueError('Real projection shape differs from frozen config')
            batches=[[x for x in registered if x['family']=='native']]
            tiles=sorted({tuple(x['tile']) for x in registered if x['tile'] is not None})
            batches += [[x for x in registered if x['tile']==list(tile)] for tile in tiles]
            random.Random(cfg['order_seed']).shuffle(batches)
            cases_start=len(self.journal.cases)
            for index,arms in enumerate(batches):
                self.check();group=dict(group_id=shape['id']+'__screen_'+str(index),shape=shape,arms=arms,
                    tile=None,scopes=['call'],timing='screen')
                self.emit('group_start',group_id=group['group_id'],index=index+1,total=len(batches),stage='real_screen')
                entries=runner.compile_entries(group,a,b)
                runner.execute_case(group,dict(distribution='real_model_window0',seed=cfg['screen_seed']),entries,a,b)
                self.emit('group_complete',group_id=group['group_id'],stage='real_screen')
                del entries;jax.clear_caches();gc.collect()
            choices=grid.select_candidates(self.journal.cases[cases_start:],cfg,[shape])[shape['id']]
            selected[shape['id']]=choices
            base.exclusive_json(self.args.output_dir/(projection+'_selection.json'),dict(selected=choices,
                frozen_before_confirmation=True,screen_window=0,confirmation_windows=cfg['operand_windows'][1:]))
            finalists={x['candidate_id']:x for value in choices.values() for x in value['confirmation_candidates']}
            default=next(x for x in registered if x['candidate_id']=='native_default');finalists[default['candidate_id']]=default
            group=dict(group_id=shape['id']+'__confirm',shape=shape,arms=list(finalists.values()),tile=None,scopes=['call'],timing='confirm',
                inputs=[dict(distribution='real_model_window'+str(cfg['operand_windows'][wi+1]),seed=cfg['confirm_seeds'][wi]) for wi in range(3)],
                headline_candidates={family:(value['winner'] or {}).get('candidate_id') for family,value in choices.items()})
            group['headline_candidates']['native_default']='native_default'
            self.journal.groups[group['group_id']]=group
            entries=runner.compile_entries(group,a,b)
            self.emit('group_start',group_id=group['group_id'],stage='real_confirm')
            for wi,real in enumerate(operands[1:]):
                runner.execute_case(group,group['inputs'][wi],entries,real[projection],b)
            self.emit('group_complete',group_id=group['group_id'],stage='real_confirm')
            base.exclusive_json(self.args.output_dir/(projection+'_confirmation_statistics.json'),
                grid.summarize_confirmation(self.journal,[group],{shape['id']:choices},cfg))
            del entries;jax.clear_caches();gc.collect()
            for family,value in choices.items():
                if value['eligible'] and value['winner'] is not None:policies.setdefault(family,{})[projection]=value['winner']
            policies.setdefault('native_default',{})[projection]=default
        valid={family:policy for family,policy in policies.items() if set(policy)=={'gateup','down'}}
        # The screen winner stays frozen; confirmation never silently retunes it.
        for family,policy in list(valid.items()):
            if family=='native_default':continue
            for projection,arm in policy.items():
                shape_id=self.model_key+'_'+projection+'_m1024'
                rows=[r for r in self.journal.cases if r.get('shape_id')==shape_id and '__confirm' in r.get('group_id','') and r.get('arm_id')==arm['arm_id']]
                if len(rows)!=3 or any(r['status']!='ok' for r in rows):
                    self.emit('model_policy_ineligible',family=family,projection=projection,reason='Frozen winner did not pass every fresh real-input confirmation')
                    valid.pop(family,None);break
        base.exclusive_json(self.args.output_dir/'model_policies.json',dict(selected=selected,policies=valid,
            omitted_families=sorted(set(['native','cubic','one_level','two_level','native_default'])-set(valid))))
        return valid

    def full_study(self,cp,ids,policies):
        from .benchmark_n9_v001 import combine,metrics,eligible,QUALITY
        count=self.cfg['quality_windows'];length=ids.shape[1]
        initial=[jax.device_put(cp.embeddings(window)) for window in ids[:count]]
        states={arm:list(initial) for arm in policies};del initial
        resident={arm:[] for arm in policies};quality={};layer_errors=[]
        timing=self.cfg['resident_timing'];rng=random.Random(self.cfg['order_seed'])
        for index in range(cp.config['num_hidden_layers']):
            self.check();weights=jax.device_put(cp.layer(index));jax.block_until_ready(weights)
            incoming=states['native_default'][0]
            functions={arm:self.layer(cp,length,policy,index) for arm,policy in policies.items()}
            outputs={arm:fn(incoming,weights) for arm,fn in functions.items()};jax.block_until_ready(outputs)
            reference=np.asarray(outputs['native_default'],dtype=np.float32)
            for arm,output in outputs.items():
                row=dict(layer=index,arm_id=arm,metrics=vector_metrics(reference,np.asarray(output,dtype=np.float32)))
                layer_errors.append(row);self.emit('resident_layer_error',**row)
            del outputs,reference;gc.collect()
            for arm,fn in functions.items():
                for _ in range(timing['warmups']):fn(incoming,weights).block_until_ready()
            order=list(functions);rng.shuffle(order)
            for repeat in range(timing['repeats']):
                for arm in order[repeat%len(order):]+order[:repeat%len(order)]:
                    self.check();start=time.perf_counter_ns();value=functions[arm](incoming,weights);value.block_until_ready()
                    elapsed=(time.perf_counter_ns()-start)/1e6;resident[arm].append(elapsed)
                    self.emit('resident_layer_sample',layer=index,arm_id=arm,repeat=repeat,elapsed_ms=elapsed,
                        scope='Device-resident composed full layer, Python dispatch included, same Native incoming state')
                    del value
            # A full-JIT Native control exposes dispatch/fusion overhead in the composed implementation.
            mod=composed.adapter(cp.config)
            full=self.full_jit(cp,length,index)
            value=full(incoming,weights);value.block_until_ready();del value
            for repeat in range(timing['repeats']):
                start=time.perf_counter_ns();value=full(incoming,weights);value.block_until_ready()
                self.emit('native_full_jit_control_sample',layer=index,repeat=repeat,elapsed_ms=(time.perf_counter_ns()-start)/1e6,
                    scope='Additional default Native full-JIT resident layer control; not used for projection selection')
                del value
            for arm,fn in functions.items():
                for window in range(count):
                    self.check();states[arm][window]=fn(states[arm][window],weights);states[arm][window].block_until_ready()
            self.emit('quality_layer_complete',model_id=cp.manifest['model_id'],layer=index,completed=index+1,
                total=cp.config['num_hidden_layers'],arms=list(policies),windows=count)
            del incoming,weights,functions,full;gc.collect()
        norm=jax.device_put(np.array(cp.tensor('model.norm.weight'),copy=True));head=jax.device_put(cp.head())
        totals={arm:None for arm in policies}
        for window in range(count):
            subtotals={arm:None for arm in policies}
            for lo in range(0,length-1,128):
                self.check();hi=min(lo+128,length-1)
                reference=composed.head(cp,states['native_default'][window][lo:hi],norm,head)
                targets=jax.device_put(ids[window,lo+1:hi+1])
                for arm in policies:
                    candidate=(reference if arm=='native_default' else composed.head(cp,states[arm][window][lo:hi],norm,head))
                    raw={k:v.item() for k,v in jax.device_get(qm.compare_logits(reference,candidate,targets)).items()}
                    totals[arm]=combine(totals[arm],raw);subtotals[arm]=combine(subtotals[arm],raw)
                    del candidate
                del reference,targets
            self.emit('quality_window',window=window,metrics={arm:metrics(v) for arm,v in subtotals.items()})
        for arm,raw in totals.items():
            quality[arm]=metrics(raw);quality[arm]['passed']=eligible(quality[arm],QUALITY);quality[arm]['thresholds']=QUALITY
            if quality[arm]['positions']!=count*(length-1):raise ValueError('Scored token count mismatch')
        del norm,head,states;gc.collect()
        streamed={arm:[] for arm in policies};order=list(policies);rng.shuffle(order)
        for repeat in range(self.cfg['streamed_repeats']):
            for arm in order[repeat%len(order):]+order[:repeat%len(order)]:
                self.check();start=time.perf_counter_ns()
                hidden,normalized=self.forward(cp,ids[0],policies[arm],logits=False)
                norm=jax.device_put(np.array(cp.tensor('model.norm.weight'),copy=True));weight=jax.device_put(cp.head())
                for lo in range(0,length,128):
                    value=composed.head(cp,hidden[lo:lo+128],norm,weight);value.block_until_ready();del value
                elapsed=(time.perf_counter_ns()-start)/1e6;streamed[arm].append(elapsed)
                self.emit('streamed_model_sample',arm_id=arm,repeat=repeat,elapsed_ms=elapsed,
                    scope='Full1024-token teacher-forced prefill; checkpoint reads/layout/device transfer/all layers/full vocabulary head; no download/tokenization/compile')
                del hidden,normalized,norm,weight;gc.collect()
        # Warm the exact streamed control, including any global/local attention specialization.
        self.forward_native_full_jit(cp,ids[0])
        full_jit_streamed=[]
        for repeat in range(self.cfg['streamed_repeats']):
            self.check();start=time.perf_counter_ns();self.forward_native_full_jit(cp,ids[0])
            elapsed=(time.perf_counter_ns()-start)/1e6;full_jit_streamed.append(elapsed)
            self.emit('native_full_jit_streamed_control_sample',repeat=repeat,elapsed_ms=elapsed,
                scope='Additional default Native full-JIT layers in identical streamed checkpoint/head loop; warmed; separate descriptive control')
        base.exclusive_json(self.args.output_dir/'quality.json',quality)
        base.exclusive_json(self.args.output_dir/'resident_layer_errors.json',layer_errors)
        return dict(quality=quality,native_full_jit_streamed_control=timing_ci(full_jit_streamed,self.cfg['order_seed']),resident_layer_timings={arm:timing_ci(v,self.cfg['order_seed']) for arm,v in resident.items()},
            streamed_model_timings={arm:timing_ci(v,self.cfg['order_seed']) for arm,v in streamed.items()},
            resident_caveat='Pooled layer statistics are descriptive; use journal for paired per-layer analysis. Full-JIT Native control separately recorded.',
            streamed_caveat='Three repeats are descriptive; streamed transfer/dispatch costs differ from a fully resident serving engine.')

    def run(self,bundle):
        self.model_key=bundle['model_key'];cfg=self.cfg
        for name,expected in bundle['input_sha256'].items():
            if base.digest_file(Path(bundle[name]))!=expected:raise ValueError('Input manifest hash changed: '+name)
        manifest=json.loads(Path(bundle['model_manifest']).read_text())
        mod=gm if manifest['config']['model_type']=='gemma3_text' else qm
        cp=mod.Checkpoint(bundle['model_manifest'])
        if cp.manifest['model_id']!=bundle['model_id'] or cp.manifest['revision']!=bundle['revision']:raise ValueError('Wrong checkpoint identity')
        tokenpath=Path(bundle['tokens_manifest']);tokenmanifest=json.loads(tokenpath.read_text())
        ids_path=tokenpath.parent/tokenmanifest['token_array_path']
        if base.digest_file(ids_path)!=tokenmanifest['token_array_sha256']:raise ValueError('Token payload changed')
        if tokenmanifest['model_manifest_sha256']!=bundle['input_sha256']['model_manifest']:raise ValueError('Token/model provenance mismatch')
        ids=np.load(ids_path,allow_pickle=False)
        if ids.shape!=(32,1024) or ids.dtype!=np.dtype('<i4'):raise ValueError('Expected frozen32x1024tokens')
        referencepath=Path(bundle['reference_manifest']);self.reference_dir=referencepath.parent
        reference=json.loads(referencepath.read_text())
        if reference['tokens_sha256']!=base.digest_file(ids_path) or reference['model_manifest_sha256']!=bundle['input_sha256']['model_manifest']:
            raise ValueError('Independent oracle does not use identical inputs')
        base.exclusive_json(self.args.output_dir/'input_provenance.json',bundle)
        default=next(a for a in grid.candidates(cfg) if a['candidate_id']=='native_default')
        native={k:default for k in ('gateup','down')}
        qualification=self.qualify(cp,ids,reference,native)
        operands,weights=self.real_operands(cp,ids,native)
        policies=self.tune(cp,operands,weights)
        del operands,weights;gc.collect()
        result=dict(model_id=bundle['model_id'],model_key=self.model_key,qualification=qualification,
            actual_model_execution=True,model_size_scope='Original N9 model; not the larger synthetic shape-only model',
            policies=list(policies),quality_measured=False)
        if qualification['passed'] and 'native_default' in policies:
            result.update(self.full_study(cp,ids,policies),quality_measured=True,status='completed')
        else:
            result.update(status='blocked_full_model_quality',reason='Native independent qualification did not pass; real first-layer operand timings remain descriptive, no full-model quality/performance claim')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-identity',type=Path,required=True)
    p.add_argument('--allocation-id',required=True);p.add_argument('--max-wall-seconds',type=float,default=5400)
    p.add_argument('--model-input',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.campaign.read_text());journal=Journal(args.output_dir,args.phase);error=None;result=None
    try:
        base.snapshot_sources(args.output_dir,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        if 'jax' in sys.modules:raise RuntimeError('Fresh process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        global jax,jnp,np,qm,gm,composed,grid
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from . import model_n9_v001 as qm,model_gemma_v001 as gm,model_composed_v001 as composed
        from . import benchmark_mlsys_shapes_v001 as grid
        from .kernels_v002 import enable_qualified_mosaic_v7_compat
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        grid.install_reference_policy(cfg)
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat());base.exclusive_json(args.output_dir/'environment.json',env)
        journal.emit('identity_check',**base.verify_identity(env,args.expected_identity,cfg))
        bundle=json.loads(args.model_input.read_text())
        if bundle['status']!='ready':raise ValueError('Actual model input preparation was not successful')
        result=Study(args,cfg,journal).run(bundle)
        base.exclusive_json(args.output_dir/'model_summary.json',result)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));journal.emit('run_error',**error,traceback=traceback.format_exc())
    summary=dict(completed=error is None,error=error,model=result,case_status_counts=dict(journal.status_counts),
        interpretation='Real original N9 checkpoints. Numerical/model eligibility is separate from infrastructure completion.')
    base.exclusive_json(args.output_dir/'summary.json',summary);journal.close()
    return 0 if error is None else 1

if __name__=='__main__':raise SystemExit(main())
