"""Four-arm extension of pinned parent kernels; one process per batch/stage."""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import parent_stage_v001 as transport
from parent_exact_stage_v001 import verify_parent, GATE_ENV

ARMS = ('native', 'native_tuned', 'cubic_tuned', 'strassen_tuned')
PARENT_ARMS = dict(native='regular_xla', native_tuned='regular_xla',
                   cubic_tuned='gated_cubic', strassen_tuned='gated_strassen')
ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
    return value


def options(mib):
    return {} if mib is None else {'xla_tpu_scoped_vmem_limit_kib': mib * 1024}


def default_profile(batch, sequence):
    bm = min(2048, batch * sequence)
    return dict(product_tile=[bm,1024,5120], cubic_tile=[bm,2048,512],
                qk_tile=[bm,1024,5120], native_options={},
                cubic_sites={'o':[bm,512,8192], 'down':[bm,2560,1024]},
                strassen_sites={'o':[bm,512,8192], 'down':[bm,2560,1024]})


def setup(a, emit):
    global jax, jnp, np, layer, stream, common, cubic, sp, gate_tuner, site_tuner, natural
    verify_parent()
    for key in list(os.environ):
        if key.startswith('QWEN3_'): del os.environ[key]
    transport.configure(a.output)
    os.environ.update(QWEN3_SEQUENCE=str(a.sequence), STRASSEN_MANAGE_CEILING='0',
                      QWEN3_STRASSEN_POLICY='up_only', QWEN3_PRODUCT_PANELS='')
    os.environ.update(GATE_ENV)
    # Import the dimension owner first, before modules copy TOKENS at import.
    layer = importlib.import_module('benchmark_qwen3_32b_layer')
    layer.BATCH, layer.TOKENS = a.batch, a.batch * a.sequence
    stream = importlib.import_module('benchmark_qwen3_32b_streamed_inference')
    gate_tuner = importlib.import_module('benchmark_qwen3_scaling_tile_tune')
    site_tuner = importlib.import_module('benchmark_qwen3_site_tile_tune')
    natural = importlib.import_module('benchmark_qwen3_streamed_natural_gate')
    # Module imports above set process flags. Reset before initializing a device.
    # Native has no scoped-memory override; other arms use explicit JIT options.
    os.environ['LIBTPU_INIT_ARGS'] = '--xla_tpu_use_enhanced_launch_barrier=true'
    import jax as jax_module
    import jax.numpy as jnp_module
    import numpy as np_module
    jax, jnp, np = jax_module, jnp_module, np_module
    common, cubic, sp = stream.common, stream.cubic, stream.sp
    stream.PRODUCT_AWARE, stream.FUSED_QK = True, True
    stream.EXTENDED_SITES = ('o','down')
    stream.STRASSEN_LIMIT, stream.CUBIC_LIMIT = 120*1024**2, 124*1024**2
    site_tuner.STRASSEN_LIMIT = 120*1024**2
    transport.install_transport(layer, a.cache, emit)
    for module in (layer, stream, natural, gate_tuner, site_tuner): module.emit = emit
    versions = {k:md.version(k) for k in ('jax','jaxlib','libtpu','transformers','tokenizers','datasets')}
    if any(versions[k] != v for k,v in {'jax':'0.7.2','jaxlib':'0.7.2','libtpu':'0.0.21.1'}.items()):
        raise RuntimeError('Require original parent JAX/TPU stack')
    if len(jax.devices()) != 1 or 'v6' not in jax.devices()[0].device_kind.lower():
        raise RuntimeError('Require one v6e TPU')
    emit(dict(kind='environment', stage=a.stage, batch=a.batch, sequence=a.sequence,
              versions=versions, device=jax.devices()[0].device_kind,
              flags=os.environ['LIBTPU_INIT_ARGS'], arms=ARMS,
              parent_commit='95be1fb088656a89813b04492e1d77c66b36ccf9',
              changes='explicit batch, four arms, independent tuning, scoped measurements; parent kernels unchanged'))


def set_profile(profile, arm='strassen_tuned'):
    stream.PRODUCT_TILE = tuple(profile['product_tile'])
    stream.CUBIC_TILE = tuple(profile['cubic_tile'])
    stream.QK_TILE = tuple(profile['qk_tile'])
    stream.SITE_TILES = {s:tuple(t) for s,t in profile[
        'cubic_sites' if arm == 'cubic_tuned' else 'strassen_sites'].items()}


def arm_options(arm, profile):
    return {} if arm == 'native' else profile['native_options'] if arm == 'native_tuned' else options(48)


def compile_fn(fn, args, opts, emit, **label):
    start = time.monotonic()
    emit(dict(kind='compile_start', **label))
    lowered = jax.jit(fn, compiler_options=opts).lower(*args)
    exe = lowered.compile()
    memory = exe.memory_analysis()
    emit(dict(kind='compiled', seconds=time.monotonic()-start, compiler_options=opts,
              memory={k:getattr(memory,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}, **label))
    return exe


def timed(exe, args, warmups=2, runs=5):
    for _ in range(warmups): jax.block_until_ready(exe(*args))
    samples=[]
    for _ in range(runs):
        start=time.perf_counter_ns(); jax.block_until_ready(exe(*args))
        samples.append((time.perf_counter_ns()-start)/1e6)
    return samples


def round_robin(exes, args, warmups, runs, offset=0):
    samples={arm:[] for arm in ARMS}; outputs={}
    for r in range(-warmups,runs):
        index = r+warmups if r<0 else r
        shift=(offset+index)%len(ARMS); order=ARMS[shift:]+ARMS[:shift]
        if r>=0 and r%2: order=tuple(reversed(order))
        for arm in order:
            start=time.perf_counter_ns(); value=exes[arm](*args[arm]); jax.block_until_ready(value)
            if r>=0: samples[arm].append((time.perf_counter_ns()-start)/1e6)
            outputs[arm]=value
    return samples,outputs


def error_set(outputs):
    finite=all(bool(jax.device_get(jnp.all(jnp.isfinite(v)))) for v in outputs.values())
    return dict(finite=finite, errors={arm:common.device_error(v,outputs['native']) for arm,v in outputs.items() if arm!='native'})


def repetitive_tokens(emit):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(layer.REPOSITORY,revision=layer.REVISION,use_fast=True,token=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    texts=[(stream.TEXTS[b%len(stream.TEXTS)]+'\n')*100 for b in range(layer.BATCH)]
    encoded=tokenizer(texts,add_special_tokens=True,max_length=layer.SEQUENCE,padding='max_length',truncation=True,return_attention_mask=True,return_tensors='np')
    if not np.all(encoded['attention_mask']==1): raise RuntimeError('Unfilled prompt positions')
    ids=encoded['input_ids'].astype(np.int32)
    emit(dict(kind='tokens',corpus='parent repetitive strings cycled by batch',shape=list(ids.shape),sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    return ids


def initial_hidden(checkpoint, ids):
    embedding=checkpoint.tensor('model.embed_tokens.weight')
    value=jnp.asarray(embedding[ids].reshape(layer.TOKENS,layer.MODEL_DIM).copy())
    jax.block_until_ready(value); return value


def block_exes(x, params, profile, emit):
    exes={}
    for arm in ARMS:
        set_profile(profile,arm)
        exes[arm]=compile_fn(stream.make_product_layer(PARENT_ARMS[arm]),(x,params),arm_options(arm,profile),emit,arm=arm,scope='block')
    return exes


def tune(a, plan, emit):
    profile=default_profile(a.batch,a.sequence); set_profile(profile)
    checkpoint=layer.ShardedSafetensors(); ids=repetitive_tokens(emit)
    x=initial_hidden(checkpoint,ids); rope=layer.rope_values()
    params=stream.load_layer(checkpoint,0,*rope)
    native=stream.make_product_layer('regular_xla'); scores=[]
    for mib in plan['native_vmem_mib']:
        try:
            exe=compile_fn(native,(x,params),options(mib),emit,arm='native_tuned',candidate_mib=mib)
            value=exe(x,params); jax.block_until_ready(value)
            if not bool(jax.device_get(jnp.all(jnp.isfinite(value)))): raise RuntimeError('Nonfinite Native candidate')
            samples=timed(exe,(x,params)); scores.append((statistics.fmean(samples),mib))
            emit(dict(kind='native_tune',candidate_mib=mib,samples_ms=samples,mean_ms=statistics.fmean(samples)))
            del exe,value
        except Exception as e: emit(dict(kind='candidate_failure',arm='native_tuned',candidate_mib=mib,error_type=type(e).__name__,error=str(e)[-2000:]))
        jax.clear_caches(); gc.collect()
    if not scores: raise RuntimeError('No Native candidate compiled')
    profile['native_options']=options(min(scores,key=lambda row:row[0])[1])
    del x,params,checkpoint; gc.collect(); jax.clear_caches()
    # Parent gate/up screen is unchanged, including seeded synthetic inputs,
    # candidate order, memory estimates, and mean-of-five selection.
    gate_records=[]
    gate_tuner.emit=lambda row:(gate_records.append(row),emit(dict(row,phase='gate_tune')))
    original_time=gate_tuner.time_function
    def gate_time(fn,args):
        exe=compile_fn(fn,args,options(128),emit,scope='gate_tune')
        return timed(exe,args)
    gate_tuner.time_function=gate_time
    gate_tuner.main()
    gate_tuner.time_function=original_time
    winner=next(r['winners'] for r in gate_records if r.get('kind')=='chosen')
    profile['product_tile']=winner['product_strassen']['tile']
    profile['cubic_tile']=winner['cubic']['tile']
    jax.clear_caches(); gc.collect()
    for site in ('o','down'):
        depth,width,_=site_tuner.SITES[site]
        kept,filtered=site_tuner.enumerate_site(depth,width)
        emit(dict(kind='site_candidates',site=site,kept=kept,filtered=filtered))
        x,w=common.device_uniform_inputs((layer.TOKENS,depth,width),site_tuner.SEED)
        residual=common.device_uniform_inputs((layer.TOKENS,width,512),site_tuner.SEED+1)[0] if site=='down' else None
        for arm in ('cubic_tuned','strassen_tuned'):
            scores=[]
            for entry in kept:
                tile=entry['tile']; bm,bn,bk=tile
                try:
                    def fn(a,w,residual,bm=bm,bn=bn,bk=bk,arm=arm):
                        if arm=='strassen_tuned':
                            return sp.strassen_matmul(a,w,bm=bm,bn=bn,bk=bk,interleave_products=True,vmem_limit_bytes=120*1024**2,residual=residual,epilogue='residual_add' if residual is not None else None)
                        result=cubic.cubic_matmul(a,w,variant='blocked',bm=bm,bn=bn,bk=bk,vmem_limit_bytes=124*1024**2)
                        return result if residual is None else (result.astype(jnp.float32)+residual.astype(jnp.float32)).astype(jnp.bfloat16)
                    exe=compile_fn(fn,(x,w,residual),options(48),emit,scope='site_tune',site=site,arm=arm,tile=tile)
                    samples=timed(exe,(x,w,residual)); mean=statistics.fmean(samples); scores.append((mean,tile))
                    emit(dict(kind='site_tune',site=site,arm=arm,tile=tile,mean_ms=mean,samples_ms=samples)); del exe
                except Exception as e: emit(dict(kind='candidate_failure',site=site,arm=arm,tile=tile,error_type=type(e).__name__,error=str(e)[-2000:]))
                jax.clear_caches(); gc.collect()
            if not scores: raise RuntimeError('No candidate for '+arm+'/'+site)
            profile['cubic_sites' if arm=='cubic_tuned' else 'strassen_sites'][site]=min(scores,key=lambda row:row[0])[1]
        del x,w,residual; jax.clear_caches(); gc.collect()
    save(a.output/'profile.json',dict(profile=profile,batch=a.batch,sequence=a.sequence,selection='parent five-sample mean; frozen before confirmation',qk='fixed published tile; BM clipped for B1'))
    emit(dict(kind='profile_frozen',profile=profile))


def projection(site, arm, profile):
    """Same operations as the parent block, exposed for isolated timing."""
    def run(x,p,residual):
        if site in ('q','k'):
            weight=p.query if site=='q' else p.key
            norm=p.query_norm if site=='q' else p.key_norm
            heads=layer.HEADS if site=='q' else layer.KV_HEADS
            if arm=='strassen_tuned':
                bm,bn,bk=profile['qk_tile']
                flat=sp.strassen_matmul(x,p.query_laid if site=='q' else p.key_laid,bm=bm,bn=bn,bk=bk,interleave_products=True,epilogue='qk_norm_rope',rope_cos=p.rope_table_cos,rope_sin=p.rope_table_sin,rope_scale=p.query_norm_laid if site=='q' else p.key_norm_laid,rope_head_dim=layer.HEAD_DIM,rope_eps=layer.RMS_EPS,vmem_limit_bytes=120*1024**2)
                result=layer.rope_layout_to_heads(flat,heads,bn)
            else:
                result=sp.native_matmul(x,weight).reshape(layer.BATCH,layer.SEQUENCE,heads,layer.HEAD_DIM)
                result=layer.apply_rope(layer.rms_norm(result,norm),p.rope_cos,p.rope_sin)
            return result.reshape(layer.TOKENS,heads*layer.HEAD_DIM)
        if site=='v': return sp.native_matmul(x,p.value)
        weight=p.gate_up if site=='gateup' else p.attention_output if site=='o' else p.mlp_down
        if arm in ('native','native_tuned'): result=sp.native_matmul(x,weight)
        else:
            tile=profile['product_tile' if arm=='strassen_tuned' else 'cubic_tile'] if site=='gateup' else profile['strassen_sites' if arm=='strassen_tuned' else 'cubic_sites'][site]
            bm,bn,bk=tile
            if arm=='strassen_tuned':
                if site=='gateup':
                    return sp.strassen_matmul(x,p.gate_up_laid,bm=bm,bn=bn,bk=bk,interleave_products=True,epilogue='swiglu',product_aware_swiglu=True,vmem_limit_bytes=120*1024**2)
                return sp.strassen_matmul(x,weight,bm=bm,bn=bn,bk=bk,interleave_products=True,epilogue='residual_add' if site=='down' else None,residual=residual if site=='down' else None,vmem_limit_bytes=120*1024**2)
            result=cubic.cubic_matmul(x,weight,variant='blocked',bm=bm,bn=bn,bk=bk,vmem_limit_bytes=124*1024**2)
        if site=='gateup':
            gate,up=jnp.split(result,2,axis=-1)
            return (jax.nn.silu(gate.astype(jnp.float32))*up.astype(jnp.float32)).astype(jnp.bfloat16)
        return (result.astype(jnp.float32)+residual.astype(jnp.float32)).astype(jnp.bfloat16) if site=='down' else result
    return run


def projection_confirm(x,p,profile,emit):
    traces={}
    def trace(site,a,w):
        traces[site]=a
        return layer.native_projection(a,w)
    residual,normalized=layer.attention_prefix(x,p,gemm=trace)
    traces['gateup']=normalized
    full=sp.native_matmul(normalized,p.gate_up); g,u=jnp.split(full,2,axis=-1)
    traces['down']=(jax.nn.silu(g.astype(jnp.float32))*u.astype(jnp.float32)).astype(jnp.bfloat16)
    jax.block_until_ready((traces,residual)); del full,g,u
    for site in ('q','k','v','o','gateup','down'):
        args=(traces[site],p,residual)
        exes={arm:compile_fn(projection(site,arm,profile),args,arm_options(arm,profile),emit,arm=arm,site=site,scope='projection_confirmation') for arm in ARMS}
        samples,outputs=round_robin(exes,{arm:args for arm in ARMS},3,20)
        errors=error_set(outputs)
        emit(dict(kind='projection_confirmation',site=site,input_shape=list(traces[site].shape),samples_ms=samples,mean_ms={k:statistics.fmean(v) for k,v in samples.items()},**errors))
        if not errors['finite']: raise RuntimeError('Nonfinite projection confirmation')
        del exes,outputs; gc.collect()


def metric_chunk(rows, lm_head, targets, mask):
    logits=tuple(sp.native_matmul(h,lm_head).astype(jnp.float32) for h in rows)
    lp=tuple(jax.nn.log_softmax(v,axis=-1) for v in logits)
    m=mask.astype(jnp.float32); probability=jnp.exp(lp[0])
    records=[]
    for index in range(len(ARMS)):
        delta=logits[index]-logits[0]
        records.append((-jnp.sum(jnp.take_along_axis(lp[index],targets[:,None],axis=1)[:,0]*m),
            jnp.sum(jnp.sum(probability*(lp[0]-lp[index]),axis=1)*m),
            jnp.sum((jnp.argmax(logits[index],axis=1)==jnp.argmax(logits[0],axis=1))*mask),
            jnp.sum(delta*delta*m[:,None]),jnp.max(jnp.abs(delta)*m[:,None])))
    return tuple(records),jnp.sum(logits[0]*logits[0]*m[:,None]),jnp.max(jnp.abs(logits[0])*m[:,None]),jnp.sum(mask),jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in logits]))


def task_metrics(hidden, checkpoint, ids, gates, emit):
    norm=jnp.asarray(checkpoint.tensor('model.norm.weight')); head=stream.device_matrix(checkpoint,'lm_head.weight')
    totals={arm:[0.,0.,0.,0.,0.] for arm in ARMS}; count=0; native_sq=0.; native_max=0.; finite=True; exe=None
    for segment in range(len(ids)):
        rows={arm:layer.rms_norm(hidden[arm][segment],norm).reshape(layer.BATCH,layer.SEQUENCE,layer.MODEL_DIM)[:,:-1].reshape(-1,layer.MODEL_DIM) for arm in ARMS}
        targets=ids[segment][:,1:].reshape(-1)
        for start in range(0,len(targets),256):
            end=min(start+256,len(targets)); n=end-start
            args=(tuple(jnp.pad(rows[arm][start:end],((0,256-n),(0,0))) for arm in ARMS),head,jnp.asarray(np.pad(targets[start:end],(0,256-n))),jnp.asarray(np.r_[np.ones(n,np.int32),np.zeros(256-n,np.int32)]))
            if exe is None: exe=jax.jit(metric_chunk).lower(*args).compile()
            values,sq,mx,num,ok=jax.device_get(exe(*args)); finite=finite and bool(ok)
            count+=int(num); native_sq+=float(sq); native_max=max(native_max,float(mx))
            for arm,vs in zip(ARMS,values,strict=True):
                for k,v in enumerate(vs): totals[arm][k]=max(totals[arm][k],float(v)) if k==4 else totals[arm][k]+float(v)
            if start%(256*32)==0: emit(dict(kind='quality_scoring',segment=segment,scored_positions=count,segment_positions=len(targets),finite=finite))
        del rows; gc.collect()
    base=totals['native'][0]/count; result={}
    for arm,values in totals.items():
        nll=values[0]/count
        row=dict(nll=nll,perplexity=math.exp(nll) if nll<700 else float('inf'),absolute_loss_delta=abs(nll-base),relative_perplexity_delta=math.expm1(nll-base) if nll-base<700 else float('inf'),mean_kl_nats=values[1]/count,top1_agreement=values[2]/count,logit_l2_relative=math.sqrt(values[3]/max(native_sq,1e-30)),logit_maxnorm_relative=values[4]/max(native_max,1e-30),logit_max_abs=values[4],positions=count,finite=finite)
        row['passes']=finite and row['absolute_loss_delta']<=gates['absolute_loss_delta'] and row['mean_kl_nats']<=gates['mean_kl_nats'] and row['top1_agreement']>=gates['top1_agreement']
        result[arm]=row
    emit(dict(kind='quality_metrics',thresholds=gates,results=result)); return result


def confirm(a,plan,profile,emit):
    set_profile(profile); checkpoint=layer.ShardedSafetensors(); ids=repetitive_tokens(emit)
    x=initial_hidden(checkpoint,ids); rope=layer.rope_values(); p=stream.load_layer(checkpoint,0,*rope)
    projection_confirm(x,p,profile,emit); jax.clear_caches(); gc.collect()
    exes=block_exes(x,p,profile,emit)
    samples,outputs=round_robin(exes,{arm:(x,p) for arm in ARMS},plan['block_warmups'],plan['block_runs'])
    emit(dict(kind='block_confirmation',samples_ms=samples,mean_ms={k:statistics.fmean(v) for k,v in samples.items()},**error_set(outputs)))
    del outputs
    hidden={arm:x for arm in ARMS}; del x
    means={arm:[] for arm in ARMS}; all_finite=True
    for index in range(layer.NUM_LAYERS):
        start=time.perf_counter()
        if index: p=stream.load_layer(checkpoint,index,*rope)
        preparation=time.perf_counter()-start
        samples,hidden=round_robin(exes,{arm:(hidden[arm],p) for arm in ARMS},plan['layer_warmups'],plan['layer_runs'],index)
        errors=error_set(hidden); all_finite=all_finite and errors['finite']
        for arm in ARMS: means[arm].append(statistics.fmean(samples[arm]))
        emit(dict(kind='layer',layer=index,preparation_seconds=preparation if index else None,first_layer_preparation_excluded=True,samples_ms=samples,mean_ms={arm:means[arm][-1] for arm in ARMS},**errors))
        del p; gc.collect()
        if not errors['finite']: raise RuntimeError('Nonfinite propagated hidden state')
    sums={arm:sum(values) for arm,values in means.items()}; comparisons={}
    for arm in ARMS[1:]:
        comparisons[arm]={}
        for baseline in ('native','native_tuned'):
            if arm==baseline: continue
            interval=stream.bootstrap_sum_interval([x-y for x,y in zip(means[arm],means[baseline],strict=True)],20260829)
            comparisons[arm][baseline]=dict(speedup=sums[baseline]/sums[arm],saved_ms=sums[baseline]-sums[arm],candidate_minus_baseline=interval)
    emit(dict(kind='aggregate_performance',scope='sum of warmed resident layer means; transfers/embedding/head excluded',aggregate_ms=sums,comparisons=comparisons,all_finite=all_finite))
    results=task_metrics({arm:[value] for arm,value in hidden.items()},checkpoint,ids[None],plan['repetitive_quality_gates'],emit)
    save(a.output/'summary.json',finite_json(dict(aggregate_ms=sums,comparisons=comparisons,repetitive_quality=results)))


def quality(a,plan,profile,emit):
    set_profile(profile); natural.SEGMENTS=plan['quality_segments']; ids=natural.wikitext_segments()
    checkpoint=layer.ShardedSafetensors(); rope=layer.rope_values()
    initial=[initial_hidden(checkpoint,segment) for segment in ids]
    hidden={arm:list(initial) for arm in ARMS}; del initial
    p=stream.load_layer(checkpoint,0,*rope); exes=block_exes(hidden['native'][0],p,profile,emit)
    for index in range(layer.NUM_LAYERS):
        if index: p=stream.load_layer(checkpoint,index,*rope)
        for segment in range(len(ids)):
            outputs={arm:exes[arm](hidden[arm][segment],p) for arm in ARMS}; jax.block_until_ready(outputs)
            errors=error_set(outputs)
            emit(dict(kind='quality_layer',layer=index,segment=segment,**errors))
            if not errors['finite']: raise RuntimeError('Nonfinite quality hidden state')
            for arm in ARMS: hidden[arm][segment]=outputs[arm]
        del p,outputs; gc.collect()
    results=task_metrics(hidden,checkpoint,ids,plan['natural_quality_gates'],emit)
    save(a.output/'summary.json',finite_json(dict(results=results,all_pass=all(v['passes'] for v in results.values()))))


def prefill(a,plan,profile,emit):
    set_profile(profile); checkpoint=layer.ShardedSafetensors(); ids=repetitive_tokens(emit); rope=layer.rope_values()
    x=initial_hidden(checkpoint,ids); p=stream.load_layer(checkpoint,0,*rope); exes=block_exes(x,p,profile,emit)
    norm=jnp.asarray(checkpoint.tensor('model.norm.weight')); head=stream.device_matrix(checkpoint,'lm_head.weight')
    def final(h,n,w):
        last=layer.rms_norm(h.reshape(layer.BATCH,layer.SEQUENCE,layer.MODEL_DIM)[:,-1],n)
        return sp.native_matmul(last,w)
    final_exe=jax.jit(final).lower(x,norm,head).compile()
    del x,p,norm,head; gc.collect()
    # All network acquisition must already have finished during confirm/quality.
    def cached_only(row):
        if row.get('kind')=='checkpoint_transport' and not row['cache_hit']:
            raise RuntimeError('Offloaded prefill unexpectedly downloaded checkpoint bytes')
        # Avoid logger disk I/O inside the measured pass; keep counters in memory.
        if row.get('kind')=='checkpoint_transport': transfers.append(row)
    transport.install_transport(layer,a.cache,cached_only)
    samples={arm:[] for arm in ARMS}
    for r in range(-plan['offloaded_prefill_warmups'],plan['offloaded_prefill_runs']):
        shift=(r+plan['offloaded_prefill_warmups'])%4; order=ARMS[shift:]+ARMS[:shift]
        for arm in order:
            transfers=[]
            emit(dict(kind='prefill_pass_start',arm=arm,round=r,warmup=r<0))
            start=time.perf_counter_ns(); hidden=initial_hidden(checkpoint,ids)
            for index in range(layer.NUM_LAYERS):
                params=stream.load_layer(checkpoint,index,*rope)
                hidden=exes[arm](hidden,params); jax.block_until_ready(hidden); del params
            norm=jnp.asarray(checkpoint.tensor('model.norm.weight')); head=stream.device_matrix(checkpoint,'lm_head.weight')
            logits=final_exe(hidden,norm,head); jax.block_until_ready(logits)
            milliseconds=(time.perf_counter_ns()-start)/1e6
            finite=bool(jax.device_get(jnp.all(jnp.isfinite(logits))))
            if r>=0: samples[arm].append(milliseconds)
            emit(dict(kind='offloaded_prefill',arm=arm,round=r,warmup=r<0,milliseconds=milliseconds,finite=finite,cache_read_bytes=sum(v['bytes'] for v in transfers),cache_read_seconds=sum(v['seconds'] for v in transfers),shared_preparation_for_all_arms=True,last_position_logits_only=True))
            if not finite: raise RuntimeError('Nonfinite prefill logits')
            del hidden,norm,head,logits; gc.collect()
    save(a.output/'summary.json',dict(scope='serialized cached-disk-offloaded prefill, common preparation, recurring transfers included; no decode',samples_ms=samples,mean_ms={arm:statistics.fmean(v) for arm,v in samples.items()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['tune','confirm','quality','prefill'],required=True)
    parser.add_argument('--batch',type=int,required=True); parser.add_argument('--sequence',type=int,default=1024)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/qwen32_parent_batch_v001.json')
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--profile',type=Path); a=parser.parse_args()
    plan=json.loads(a.plan.read_text())
    if a.batch not in plan['batches'] or a.sequence!=plan['sequence']: raise ValueError('Unregistered workload')
    if a.stage!='tune' and not a.profile: raise ValueError('Frozen profile required')
    a.output.mkdir(parents=True,exist_ok=False)
    def emit(row):
        row=finite_json(dict(row,batch=a.batch,sequence=a.sequence,stage=a.stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        line=json.dumps(row,allow_nan=False)
        with (a.output/'events.jsonl').open('a') as f: f.write(line+'\n')
        print(line,flush=True)
    start=time.monotonic(); status={'status':'failed'}
    try:
        setup(a,emit)
        if a.stage=='tune': tune(a,plan,emit)
        else:
            frozen=json.loads(a.profile.read_text())
            if (frozen['batch'],frozen['sequence'])!=(a.batch,a.sequence): raise ValueError('Wrong profile workload')
            emit(dict(kind='frozen_profile',sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),profile=frozen))
            globals()[a.stage](a,plan,frozen['profile'],emit)
        status['status']='completed'
    except BaseException as error:
        status.update(error_type=type(error).__name__,error=str(error)[-2000:]); emit(dict(kind='stage_failure',**status)); raise
    finally:
        status['wall_seconds']=time.monotonic()-start; save(a.output/'completion.json',status)


if __name__=='__main__': main()
