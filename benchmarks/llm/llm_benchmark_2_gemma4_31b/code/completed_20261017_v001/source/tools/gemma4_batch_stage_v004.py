"""Versioned four-arm Gemma4 experiment, with approved FP32 fidelity and same-TPU BF16 quality gates."""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.metadata as md
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import gemma4_parent_model_v002 as model
from parent_exact_stage_v001 import verify_parent
from gemma4_validation_v001 import natural_sequences,qualify_evidence
from gemma4_error_metrics_v001 import chunked_error,CHUNK_ELEMENTS

ROOT=Path(__file__).resolve().parents[1]
ARMS=model.ARMS
REPS=(0,5)


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x


def error(reference,candidate):
    return chunked_error(reference,candidate,jax=jax,jnp=jnp)


def compile_fn(fn,args,opts,emit,**labels):
    start=time.monotonic();emit(dict(kind='compile_start',test='Compiling',**labels))
    lowered=jax.jit(fn,compiler_options=opts).lower(*args)
    emit(dict(kind='lowered',test='Compiler optimization',seconds=time.monotonic()-start,**labels))
    exe=lowered.compile();mem=exe.memory_analysis()
    emit(dict(kind='compiled',test='Compile complete',seconds=time.monotonic()-start,compiler_options=opts,
        memory={k:getattr(mem,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},**labels))
    return exe


def timed(exe,args,warmups=2,runs=5):
    for _ in range(warmups):jax.block_until_ready(exe(*args))
    samples=[]
    for _ in range(runs):
        t=time.perf_counter_ns();jax.block_until_ready(exe(*args));samples.append((time.perf_counter_ns()-t)/1e6)
    return samples


def round_robin(exes,args,warmups,runs,emit,**labels):
    samples={arm:[] for arm in ARMS};outputs={}
    for r in range(-warmups,runs):
        shift=(r+warmups)%4;order=ARMS[shift:]+ARMS[:shift]
        if r>=0 and r%2:order=tuple(reversed(order))
        for arm in order:
            emit(dict(kind='timing_sample',test='Warmup' if r<0 else 'Timing sample',arm=arm,round=r,rounds=runs,**labels))
            t=time.perf_counter_ns();out=exes[arm](*args[arm]);jax.block_until_ready(out)
            elapsed=(time.perf_counter_ns()-t)/1e6
            if r>=0:samples[arm].append(elapsed)
            outputs[arm]=out
    return samples,outputs


def errors(outputs):
    result={arm:error(outputs['native'],v) for arm,v in outputs.items()}
    return dict(errors=result,finite=all(v['finite'] for v in result.values()))


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model.MODEL,revision=model.REVISION,token=False,use_fast=True)


def tokens(batch,sequence,emit,natural=False,segments=1):
    tok=tokenizer()
    if natural:
        from datasets import load_dataset
        revision='b08601e04326c79dfdd32d625aee71d232d685c3'
        rows=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test',revision=revision)
        text='\n'.join(row['text'] for row in rows)
        allids=tok(text,add_special_tokens=False)['input_ids']
        ids,extra=natural_sequences(allids,batch,sequence,segments,tok.bos_token_id,np)
        label='WikiText-2 raw test contiguous tokens with BOS per independent sequence'
        extra.update(dataset_revision=revision,available_tokens=len(allids))
    else:
        module=ast.parse((model.PARENT/'experiments/qwen3/benchmark_qwen3_32b_streamed_inference.py').read_text())
        texts=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEXTS' for t in n.targets))
        result=tok([(texts[b%len(texts)]+'\n')*200 for b in range(batch)],max_length=sequence,padding='max_length',truncation=True,return_tensors='np')
        if not np.all(result['attention_mask']==1):raise ValueError('Unexpected padding in fixed prefill')
        ids=result['input_ids'].astype(np.int32)
        if tok.bos_token_id is None or not np.all(ids[:,0]==tok.bos_token_id):raise ValueError('Repetitive inputs must begin with BOS')
        label='Parent repetitive strings cycled by batch';extra={}
    emit(dict(kind='tokens',corpus=label,shape=list(ids.shape),sha256=hashlib.sha256(ids.tobytes()).hexdigest(),**extra))
    return ids


def profile_for(profiles,index):return profiles['full_attention' if index%6==5 else 'sliding_attention']


def block_exes(x,p,profile,index,a,emit):
    return {arm:compile_fn(model.build_layer(arm,profile,index,a.batch,a.sequence),(x,p),model.arm_options(arm,profile),emit,
                arm=arm,scope='complete_block',layer=index,attention_type=model.geometry(model.config,index)['layer_type']) for arm in ARMS}


def captures(x,p,profile,index,a,emit):
    """Segmented native capture avoids the large monolithic B64 compiler graph."""
    parts=model.components('native',profile,index,a.batch,a.sequence)
    def run(n,fn,args):
        exe=compile_fn(fn,args,{},emit,scope='projection_inputs',test_part=n,layer=index)
        value=exe(*args);jax.block_until_ready(value);del exe;return value
    normalized,q,k,v=run('qkv',parts[0],(x,p))
    attended=run('attention',parts[1],(q,k,v));del q,k,v
    residual,mlp=run('postattention',parts[2],(x,attended,p))
    middle=run('gateup',parts[3],(mlp,p))
    reconstructed=run('down',parts[4],(middle,residual,p))
    traces={s:normalized for s in ('q','k','v') if s in model.geometry(model.config,index)['projection_shapes_mkn']}
    traces.update(o=attended,gateup=mlp,down=middle)
    return traces,{'o':x,'down':residual},reconstructed


def candidates(site,arm,index,batch,sequence):
    m,k,n=model.geometry(model.config,index,batch,sequence)['projection_shapes_mkn'][site]
    # Full contraction is included whenever the analytic footprint permits it.
    bm_values=(min(1024,m),min(2048,m))
    if site in ('q','k'):bn_values=(1024,2048);bk_values=(768,1792,k)
    elif site=='gateup':bn_values=(1024,1536,3072);bk_values=(768,1792,k)
    else:
        bn_values=(768,1792);bk_values=tuple(dict.fromkeys((768,1024,1792,k//2,k)))
    seen=set();result=[]
    for bm,bn,bk in itertools.product(bm_values,bn_values,bk_values):
        tile=(bm,bn,bk)
        if tile in seen:continue
        seen.add(tile)
        estimate=8*bm*bn+4*(bm*bk+bk*bn)
        if m%bm or k%bk or n%bn or bm%16 or bn%256 or bk%256 or estimate>100*1024**2:continue
        result.append(dict(tile=list(tile),fused=arm=='strassen_tuned' and site in ('q','k','gateup'),estimate_bytes=estimate))
    if arm=='strassen_tuned' and site in ('o','down'):
        for bm,bk in itertools.product((128,256,512,1024),(768,1024,1792,k)):
            estimate=12*bm*n+4*(bm*bk+bk*n)
            if m%bm or k%bk or bk%256 or estimate>100*1024**2:continue
            result.append(dict(tile=[bm,n,bk],fused=True,estimate_bytes=estimate))
    return result


def tune(a,plan,emit):
    checkpoint=model.Checkpoint(a.cache,emit);ids=tokens(a.batch,a.sequence,emit);initial=model.initial_hidden(checkpoint,ids)
    profiles={}
    for index in REPS:
        kind=model.geometry(model.config,index)['layer_type'];profile=model.default_profile(a.batch,a.sequence)
        # Reach the representative layer with native propagation on real inputs.
        x=initial
        for previous in range(index):
            pr=model.default_profile(a.batch,a.sequence);p=model.load_layer(checkpoint,previous,pr,a.batch,a.sequence)
            fn=compile_fn(model.build_layer('native',pr,previous,a.batch,a.sequence),(x,p),{},emit,scope='tuning_input',layer=previous)
            x=fn(x,p);jax.block_until_ready(x);del p,fn
        raw=model.raw_layer(checkpoint,index);p=model.prepare_weights(raw,profile,index,a.batch,a.sequence)
        scores=[]
        for number,mib in enumerate(plan['native_vmem_mib'],1):
            opts={} if mib is None else {'xla_tpu_scoped_vmem_limit_kib':mib*1024}
            labels=dict(arm='native_tuned',layer=index,attention_type=kind,candidate_index=number,candidate_total=len(plan['native_vmem_mib']),candidate_mib=mib)
            try:
                exe=compile_fn(model.build_layer('native',profile,index,a.batch,a.sequence),(x,p),opts,emit,**labels)
                out=exe(x,p);jax.block_until_ready(out)
                if not bool(jax.device_get(jnp.all(jnp.isfinite(out)))):raise RuntimeError('Nonfinite Native candidate')
                samples=timed(exe,(x,p));scores.append((statistics.fmean(samples),opts))
                emit(dict(kind='native_tune',samples_ms=samples,mean_ms=scores[-1][0],**labels));del exe,out
            except Exception as e:emit(dict(kind='candidate_failure',error_type=type(e).__name__,error=str(e)[-1500:],**labels))
            jax.clear_caches();gc.collect()
        if not scores:raise RuntimeError('No Native tuning candidate passed')
        profile['native_options']=min(scores,key=lambda v:v[0])[1]
        traces,residuals,reconstructed=captures(x,p,profile,index,a,emit);del reconstructed,p
        for arm,sites in [('cubic_tuned',('gateup','o','down')),('strassen_tuned',('q','k','gateup','o','down'))]:
            for site in sites:
                if site=='k' and index==5:continue
                key='cubic' if arm=='cubic_tuned' else 'strassen';menu=candidates(site,arm,index,a.batch,a.sequence)
                emit(dict(kind='candidate_menu',arm=arm,site=site,layer=index,attention_type=kind,candidates=menu,selection='lowest finite five-sample mean; enumeration-order tie break'))
                scores=[]
                for number,entry in enumerate(menu,1):
                    chosen=copy.deepcopy(profile);chosen[key][site]=entry
                    labels=dict(arm=arm,site=site,layer=index,attention_type=kind,candidate_index=number,candidate_total=len(menu),tile=entry['tile'],fused=entry['fused'])
                    try:
                        pp=model.prepare_weights(raw,chosen,index,a.batch,a.sequence)
                        args=(traces[site],pp,residuals.get(site))
                        exe=compile_fn(model.projection(site,arm,chosen,index,a.batch,a.sequence),args,model.arm_options(arm,chosen),emit,**labels)
                        out=exe(*args);jax.block_until_ready(out)
                        if not bool(jax.device_get(jnp.all(jnp.isfinite(out)))):raise RuntimeError('Nonfinite projection candidate')
                        samples=timed(exe,args);scores.append((statistics.fmean(samples),entry))
                        emit(dict(kind='projection_tune',samples_ms=samples,mean_ms=scores[-1][0],**labels));del exe,out,pp,args
                    except Exception as e:emit(dict(kind='candidate_failure',error_type=type(e).__name__,error=str(e)[-1500:],**labels))
                    jax.clear_caches();gc.collect()
                if not scores:raise RuntimeError(f'No feasible candidate for {kind}/{arm}/{site}')
                profile[key][site]=min(scores,key=lambda v:v[0])[1]
                emit(dict(kind='site_selected',arm=arm,site=site,layer=index,attention_type=kind,selected=profile[key][site],mean_ms=min(v[0] for v in scores)))
        profiles[kind]=profile
        save(a.output/(kind+'-profile.json'),profile)
        del traces,residuals,raw,x;gc.collect();jax.clear_caches()
    save(a.output/'profile.json',dict(batch=a.batch,sequence=a.sequence,profiles=profiles,selection='independent per layer type and batch, lowest warmed mean',model=model.MODEL,revision=model.REVISION))


def qualify(a,plan,profiles,emit):
    checkpoint=model.Checkpoint(a.cache,emit);ids=tokens(a.batch,a.sequence,emit);x=model.initial_hidden(checkpoint,ids);results={}
    for index in range(6):
        profile=profile_for(profiles,index);p=model.load_layer(checkpoint,index,profile,a.batch,a.sequence)
        if index in REPS:
            traces,residuals,reconstructed=captures(x,p,profile,index,a,emit)
            del traces,residuals
            exes=block_exes(x,p,profile,index,a,emit);outputs={arm:fn(x,p) for arm,fn in exes.items()};jax.block_until_ready(outputs)
            result=errors(outputs);result['capture_reconstruction']=error(outputs['native'],reconstructed)
            result['passes']=result['finite'] and result['capture_reconstruction']['relative_l2']<=.01 and all(v['relative_l2']<=.02 for v in result['errors'].values())
            emit(dict(kind='qualification',layer=index,**result));results[str(index)]=result
            x=outputs['native'];del reconstructed,exes,outputs
        else:
            fn=compile_fn(model.build_layer('native',profile,index,a.batch,a.sequence),(x,p),{},emit,scope='qualification_propagation',layer=index)
            x=fn(x,p);jax.block_until_ready(x);del fn
        del p;gc.collect();jax.clear_caches()
    save(a.output/'summary.json',dict(results=results,all_pass=all(v['passes'] for v in results.values())))
    if not all(v['passes'] for v in results.values()):raise RuntimeError('Scientific qualification gate failed')


def projection_confirm(x,p,profile,index,a,plan,emit):
    traces,residuals,reconstructed=captures(x,p,profile,index,a,emit);del reconstructed
    for site,value in traces.items():
        args=(value,p,residuals.get(site))
        exes={arm:compile_fn(model.projection(site,arm,profile,index,a.batch,a.sequence),args,model.arm_options(arm,profile),emit,scope='isolated_projection',site=site,arm=arm,layer=index) for arm in ARMS}
        samples,outputs=round_robin(exes,{arm:args for arm in ARMS},plan['projection_warmups'],plan['projection_runs'],emit,site=site,layer=index)
        emit(dict(kind='projection_confirmation',site=site,layer=index,samples_ms=samples,mean_ms={k:statistics.fmean(v) for k,v in samples.items()},**errors(outputs)))
        del exes,outputs;gc.collect();jax.clear_caches()
    del traces,residuals;gc.collect()


def metric_chunk(hiddens,norm,head,targets,mask):
    logits=tuple(model.native.head_logits(h,norm,head,model.config).astype(jnp.float32) for h in hiddens)
    lp=tuple(jax.nn.log_softmax(x,axis=-1) for x in logits);prob=jnp.exp(lp[0]);m=mask.astype(jnp.float32);records=[]
    for i in range(4):
        d=logits[i]-logits[0]
        records.append((-jnp.sum(jnp.take_along_axis(lp[i],targets[:,None],axis=1)[:,0]*m),jnp.sum(jnp.sum(prob*(lp[0]-lp[i]),axis=1)*m),jnp.sum((jnp.argmax(logits[i],axis=1)==jnp.argmax(logits[0],axis=1))*mask),jnp.sum(d*d*m[:,None]),jnp.max(jnp.abs(d)*m[:,None])))
    return tuple(records),jnp.sum(logits[0]*logits[0]*m[:,None]),jnp.max(jnp.abs(logits[0])*m[:,None]),jnp.sum(mask),jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in logits]))


def task_metrics(hidden,checkpoint,ids,gates,a,emit):
    norm=jnp.asarray(checkpoint.tensor('model.norm.weight'));head=jnp.asarray(checkpoint.tensor('lm_head.weight').T.copy())
    totals={arm:[0.]*5 for arm in ARMS};count=0;sq=0.;mx=0.;finite=True;exe=None
    for segment in range(len(ids)):
        rows={arm:hidden[arm][segment].reshape(a.batch,a.sequence,-1)[:,:-1].reshape(-1,model.config['hidden_size']) for arm in ARMS}
        targets=ids[segment][:,1:].reshape(-1)
        for start in range(0,len(targets),256):
            n=min(256,len(targets)-start)
            args=(tuple(jnp.pad(rows[arm][start:start+n],((0,256-n),(0,0))) for arm in ARMS),norm,head,jnp.asarray(np.pad(targets[start:start+n],(0,256-n))),jnp.asarray(np.r_[np.ones(n,np.int32),np.zeros(256-n,np.int32)]))
            if exe is None:exe=compile_fn(metric_chunk,args,{},emit,scope='softcapped_logit_metrics')
            records,ss,mm,nn,ok=jax.device_get(exe(*args));count+=int(nn);sq+=float(ss);mx=max(mx,float(mm));finite=finite and bool(ok)
            for arm,values in zip(ARMS,records,strict=True):
                for k,v in enumerate(values):totals[arm][k]=max(totals[arm][k],float(v)) if k==4 else totals[arm][k]+float(v)
            if start%8192==0:emit(dict(kind='quality_scoring',test='Scoring next-token distributions',segment=segment+1,scored_positions=count,total_positions=len(ids)*a.batch*(a.sequence-1)))
        del rows;gc.collect()
    base=totals['native'][0]/count;result={}
    for arm,v in totals.items():
        nll=v[0]/count;delta=nll-base
        row=dict(nll=nll,perplexity=math.exp(nll),absolute_loss_delta=abs(delta),relative_perplexity_delta=math.expm1(delta),mean_kl_nats=v[1]/count,top1_agreement=v[2]/count,logit_l2_relative=math.sqrt(v[3]/max(sq,1e-30)),logit_maxnorm_relative=v[4]/max(mx,1e-30),logit_max_abs=v[4],positions=count,finite=finite)
        row['passes']=finite and row['absolute_loss_delta']<=gates['absolute_loss_delta'] and row['mean_kl_nats']<=gates['mean_kl_nats'] and row['top1_agreement']>=gates['top1_agreement'];result[arm]=row
    emit(dict(kind='quality_metrics',thresholds=gates,results=result));return result


def bootstrap(values):
    v=np.asarray(values,np.float64);draws=np.random.default_rng(20260829).choice(v,size=(20000,len(v)),replace=True).sum(1)
    return dict(sum_ms=float(v.sum()),ci95_ms=list(map(float,np.quantile(draws,[.025,.975]))))


def confirm(a,plan,profiles,emit):
    checkpoint=model.Checkpoint(a.cache,emit);ids=tokens(a.batch,a.sequence,emit);x=model.initial_hidden(checkpoint,ids)
    hidden={arm:x for arm in ARMS};del x;exes_by_type={};means={arm:[] for arm in ARMS}
    for index in range(60):
        kind=model.geometry(model.config,index)['layer_type'];profile=profile_for(profiles,index)
        t=time.monotonic();p=model.load_layer(checkpoint,index,profile,a.batch,a.sequence);prep=time.monotonic()-t
        if index in REPS:
            projection_confirm(hidden['native'],p,profile,index,a,plan,emit)
            exes_by_type[kind]=block_exes(hidden['native'],p,profile,index,a,emit)
            samples,outputs=round_robin(exes_by_type[kind],{arm:(hidden['native'],p) for arm in ARMS},plan['block_warmups'],plan['block_runs'],emit,scope='resident_block',layer=index,attention_type=kind)
            emit(dict(kind='block_confirmation',layer=index,attention_type=kind,samples_ms=samples,mean_ms={k:statistics.fmean(v) for k,v in samples.items()},**errors(outputs)));del outputs
        samples,hidden=round_robin(exes_by_type[kind],{arm:(hidden[arm],p) for arm in ARMS},plan['layer_warmups'],plan['layer_runs'],emit,scope='resident_layer',layer=index,attention_type=kind)
        err=errors(hidden)
        for arm in ARMS:means[arm].append(statistics.fmean(samples[arm]))
        emit(dict(kind='layer',layer=index,preparation_seconds=prep,samples_ms=samples,mean_ms={arm:means[arm][-1] for arm in ARMS},**err))
        if not err['finite']:raise RuntimeError('Nonfinite propagated hidden state')
        del p;gc.collect()
    sums={arm:sum(v) for arm,v in means.items()};comparisons={}
    for arm in ARMS[1:]:
        comparisons[arm]={base:dict(speedup=sums[base]/sums[arm],saved_ms=sums[base]-sums[arm],candidate_minus_baseline=bootstrap([x-y for x,y in zip(means[arm],means[base],strict=True)])) for base in ('native','native_tuned') if base!=arm}
    emit(dict(kind='aggregate_performance',scope='sum warmed resident layer means; embedding/head/transfers excluded',aggregate_ms=sums,comparisons=comparisons))
    del exes_by_type;jax.clear_caches();gc.collect()
    quality=task_metrics({arm:[v] for arm,v in hidden.items()},checkpoint,ids[None],plan['repetitive_quality_gates'],a,emit)
    save(a.output/'summary.json',dict(aggregate_ms=sums,comparisons=comparisons,repetitive_quality=quality,all_pass=all(v['passes'] for v in quality.values())))
    # Preserve performance even if a numerical gate fails; worker marks it ineligible.


def quality(a,plan,profiles,emit):
    ids=tokens(a.batch,a.sequence,emit,natural=True,segments=plan['quality_segments']);checkpoint=model.Checkpoint(a.cache,emit)
    initial=[model.initial_hidden(checkpoint,segment) for segment in ids];hidden={arm:list(initial) for arm in ARMS};del initial;exes={}
    for index in range(60):
        kind=model.geometry(model.config,index)['layer_type'];profile=profile_for(profiles,index)
        p=model.load_layer(checkpoint,index,profile,a.batch,a.sequence)
        if kind not in exes:exes[kind]=block_exes(hidden['native'][0],p,profile,index,a,emit)
        for segment in range(len(ids)):
            outputs={}
            for arm in ARMS:
                emit(dict(kind='quality_forward',test='Natural-text forward pass',layer=index,attention_type=kind,segment=segment+1,arm=arm))
                outputs[arm]=exes[kind][arm](hidden[arm][segment],p);jax.block_until_ready(outputs[arm])
            err=errors(outputs);emit(dict(kind='quality_layer',layer=index,segment=segment,**err))
            if not err['finite']:raise RuntimeError('Nonfinite quality hidden state')
            for arm in ARMS:hidden[arm][segment]=outputs[arm]
        del p,outputs;gc.collect()
    del exes;jax.clear_caches();gc.collect()
    metrics=task_metrics(hidden,checkpoint,ids,plan['natural_quality_gates'],a,emit)
    save(a.output/'summary.json',dict(results=metrics,all_pass=all(v['passes'] for v in metrics.values())))


def prefill(a,plan,profiles,emit):
    checkpoint=model.Checkpoint(a.cache,emit);ids=tokens(a.batch,a.sequence,emit);x=model.initial_hidden(checkpoint,ids);exes={}
    for index in REPS:
        profile=profile_for(profiles,index);p=model.load_layer(checkpoint,index,profile,a.batch,a.sequence)
        exes[model.geometry(model.config,index)['layer_type']]=block_exes(x,p,profile,index,a,emit);del p
    norm=jnp.asarray(checkpoint.tensor('model.norm.weight'));head=jnp.asarray(checkpoint.tensor('lm_head.weight').T.copy())
    def final(h,n,w):return model.native.head_logits(h.reshape(a.batch,a.sequence,-1)[:,-1],n,w,model.config)
    final_exe=compile_fn(final,(x,norm,head),{},emit,scope='last_position_head');del x,norm,head;gc.collect()
    transfers=[]
    def cached_only(row):
        if row.get('kind')=='checkpoint_transport':
            if not row['cache_hit']:raise RuntimeError('Network acquisition inside measured prefill')
            transfers.append(row)
    checkpoint=model.Checkpoint(a.cache,cached_only)
    samples={arm:[] for arm in ARMS}
    for r in range(-plan['offloaded_prefill_warmups'],plan['offloaded_prefill_runs']):
        shift=(r+plan['offloaded_prefill_warmups'])%4;order=ARMS[shift:]+ARMS[:shift]
        for arm in order:
            transfers.clear();emit(dict(kind='prefill_pass_start',test='Full offloaded prefill',arm=arm,round=r,warmup=r<0))
            # Do not log disk events inside the timed region. Phase/arm/round
            # remain visible; the independent controller supplies heartbeats.
            start=time.perf_counter_ns();hidden=model.initial_hidden(checkpoint,ids)
            for index in range(60):
                profile=profile_for(profiles,index);p=model.load_layer(checkpoint,index,profile,a.batch,a.sequence)
                hidden=exes[model.geometry(model.config,index)['layer_type']][arm](hidden,p);jax.block_until_ready(hidden);del p
            norm=jnp.asarray(checkpoint.tensor('model.norm.weight'));head=jnp.asarray(checkpoint.tensor('lm_head.weight').T.copy())
            logits=final_exe(hidden,norm,head);jax.block_until_ready(logits);elapsed=(time.perf_counter_ns()-start)/1e6
            finite=bool(jax.device_get(jnp.all(jnp.isfinite(logits))))
            if r>=0:samples[arm].append(elapsed)
            emit(dict(kind='offloaded_prefill',arm=arm,round=r,warmup=r<0,milliseconds=elapsed,finite=finite,cache_read_bytes=sum(v['bytes'] for v in transfers),cache_read_seconds=sum(v['seconds'] for v in transfers),shared_preparation_for_all_arms=True,last_position_logits_only=True))
            if not finite:raise RuntimeError('Nonfinite prefill logits')
            del hidden,norm,head,logits;gc.collect()
    save(a.output/'summary.json',dict(scope='serialized cached-disk-offloaded prefill; common preparation; recurring transfers included; no decode',samples_ms=samples,mean_ms={arm:statistics.fmean(v) for arm,v in samples.items()}))


def oracle(a,plan,emit):
    """Reuse completed FP32 fidelity only for byte-identical qualified sources."""
    from transformers.models.gemma4 import modeling_gemma4 as official
    summary=qualify_evidence(ROOT,plan,model.MODEL,model.REVISION,official.__file__)
    save(a.output/'summary.json',summary)
    emit(dict(kind='oracle_complete',test='Verified existing CPU FP32 formula control',**summary))
    if not summary['all_pass']:raise RuntimeError('Scientific FP32 formula qualification failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['oracle','tune','qualify','confirm','quality','prefill'],required=True)
    parser.add_argument('--batch',type=int,default=1);parser.add_argument('--sequence',type=int,default=1024)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/gemma4_parent_batch_v003.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--profile',type=Path);parser.add_argument('--profile-sha256');parser.add_argument('--progress',type=Path);a=parser.parse_args()
    plan=json.loads(a.plan.read_text())
    if a.batch not in plan['batches'] or a.sequence!=plan['sequence']:raise ValueError('Unregistered workload')
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();state={'status':'failed'}
    current={};stage_labels={'oracle':'CPU FP32 formula qualification','tune':'Tuning','qualify':'Numerical qualification','confirm':'Resident timing confirmation','quality':'Natural-text accuracy','prefill':'Offloaded prefill'}
    def emit(row):
        row=clean(dict(row,stage=a.stage,batch=None if a.stage=='oracle' else a.batch,sequence=32 if a.stage=='oracle' else a.sequence,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        line=json.dumps(row,allow_nan=False)
        with (a.output/'events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
        if a.progress:
            # Transport records do not erase the current experiment/test identity.
            if row['kind'] not in ('tensor','checkpoint_transport'):
                current.update({k:row[k] for k in ('batch','stage','utc','arm','layer','attention_type','site','candidate_index','candidate_total','tile','candidate_mib','test','round','rounds','segment','scored_positions','total_positions') if k in row})
                current.update(status='running',stage_label=stage_labels[a.stage],event=row['kind'],elapsed_stage_seconds=time.monotonic()-start)
            else:current.update(utc=row['utc'],transport=row['kind'])
            temp=a.progress.with_suffix('.tmp');temp.write_text(json.dumps(current)+'\n');temp.replace(a.progress)
    try:
        verify_parent();global jax,jnp,np
        model.setup();jax,jnp,np=model.jax,model.jnp,model.np
        versions={k:md.version(k) for k in ('jax','jaxlib','libtpu','transformers','tokenizers','datasets','torch')}
        if any(versions[k]!=v for k,v in {'jax':'0.11.2','jaxlib':'0.11.2','libtpu':'0.0.48','transformers':'5.5.0'}.items()):raise RuntimeError('Unregistered software stack')
        if len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Require one v6e TPU')
        emit(dict(kind='environment',test=stage_labels[a.stage],versions=versions,device=jax.devices()[0].device_kind,model=model.MODEL,revision=model.REVISION))
        emit(dict(kind='error_metric_method',method='device FP32 tile reductions; host FP64 fsum',chunk_elements=CHUNK_ELEMENTS,timed=False))
        if a.stage in ('oracle','tune'):globals()[a.stage](a,plan,emit)
        else:
            if not a.profile or hashlib.sha256(a.profile.read_bytes()).hexdigest()!=a.profile_sha256:raise ValueError('Frozen profile hash mismatch')
            frozen=json.loads(a.profile.read_text())
            if (frozen['batch'],frozen['sequence'])!=(a.batch,a.sequence):raise ValueError('Wrong frozen workload')
            emit(dict(kind='frozen_profile',sha256=a.profile_sha256,profile=frozen))
            globals()[a.stage](a,plan,frozen['profiles'],emit)
        state['status']='completed'
    except BaseException as e:
        state.update(error_type=type(e).__name__,error=str(e)[-2000:]);emit(dict(kind='stage_failure',**state));raise
    finally:state['wall_seconds']=time.monotonic()-start;save(a.output/'completion.json',state)


if __name__=='__main__':main()
