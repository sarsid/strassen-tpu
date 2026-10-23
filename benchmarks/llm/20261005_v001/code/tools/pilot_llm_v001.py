"""Bounded Qwen3-8B real layer pilot; five arms, separate store tracks.

Layer 0 only, repeated natural text from the preserved parent fixture. No final
logits/PPL or whole-model speed claim. Tuning and confirmation samples separated.
"""
import argparse, dataclasses, gc, hashlib, importlib, json, os, random, statistics, time
from pathlib import Path
import parent_stage_v001 as setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--cache',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def emit(row):
        row['utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    setup.configure(a.output)
    os.environ.update(QWEN3_MODEL='8b',QWEN3_SEQUENCE='512',STRASSEN_MANAGE_CEILING='0',QWEN3_FUSED_QK='0')
    layer=importlib.import_module('benchmark_qwen3_32b_layer');layer.BATCH=1;layer.TOKENS=512
    stream=importlib.import_module('benchmark_qwen3_32b_streamed_inference');stream.TEXTS=stream.TEXTS[:1]
    os.environ['LIBTPU_INIT_ARGS']='--xla_tpu_use_enhanced_launch_barrier=true'
    setup.install_transport(layer,a.cache,emit);layer.emit=emit;stream.emit=emit
    import jax, jax.numpy as jnp, numpy as np
    from strassen_mm import parent_fused_adapter_v003 as adapter
    from strassen_mm.kernels_fused_v004 import make_projection
    from strassen_mm.tuner_joint_v001 import arm
    assert len(jax.devices())==1 and 'v6' in jax.devices()[0].device_kind.lower()
    cp=layer.ShardedSafetensors();cos,sin=layer.rope_values();params=stream.load_layer(cp,0,cos,sin)
    ids=stream.make_tokens();np.save(a.output/'token_ids.npy',ids)
    embedding=cp.tensor('model.embed_tokens.weight');host=embedding[ids].reshape(layer.TOKENS,layer.MODEL_DIM).copy();del embedding
    start=time.perf_counter();x=jnp.asarray(host);x.block_until_ready();input_h2d=time.perf_counter()-start
    # Qualify the actual checkpoint layer independently before tuning it.
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
    torch.set_num_threads(1)
    config=Qwen3Config(**cp._get(layer.BASE+'/config.json').json());config._attn_implementation='eager'
    official=Qwen3DecoderLayer(config,0).to(torch.bfloat16).eval()
    def tt(value):return torch.tensor(np.asarray(value,dtype=np.float32),dtype=torch.bfloat16)
    state={'input_layernorm.weight':tt(params.attention_norm),'post_attention_layernorm.weight':tt(params.mlp_norm),
           'self_attn.q_norm.weight':tt(params.query_norm),'self_attn.k_norm.weight':tt(params.key_norm)}
    for site,field in [('q','query'),('k','key'),('v','value'),('o','attention_output')]:state['self_attn.'+site+'_proj.weight']=tt(getattr(params,field).T)
    gate,up=jnp.split(params.gate_up,2,axis=1)
    state.update({'mlp.gate_proj.weight':tt(gate.T),'mlp.up_proj.weight':tt(up.T),'mlp.down_proj.weight':tt(params.mlp_down.T)})
    official.load_state_dict(state,strict=True);tx=tt(x[:16][None]);positions=torch.arange(16)[None]
    rotary=Qwen3RotaryEmbedding(config)
    mask=torch.full((1,1,16,16),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(diagonal=1)
    with torch.inference_mode():oracle=official(tx,attention_mask=mask,position_embeddings=rotary(tx,positions))
    if isinstance(oracle,tuple):oracle=oracle[0]
    old_length,old_tokens=layer.SEQUENCE,layer.TOKENS
    layer.SEQUENCE=16;layer.TOKENS=16
    qpolicy={s:dict(arm(dtype='bfloat16'),architecture='v6e') for s in adapter.SITES}
    short_params=params._replace(rope_cos=params.rope_cos[:16],rope_sin=params.rope_sin[:16])
    qfn=adapter.build(layer,qpolicy)
    y=jax.jit(qfn)(x[:16],qfn.prepare_weights(short_params)).block_until_ready()
    layer.SEQUENCE=old_length;layer.TOKENS=old_tokens
    ref=oracle.float().numpy()[0];got=np.asarray(y,dtype=np.float32)
    rel=float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-30));passed=bool(np.isfinite(got).all() and rel<=.02)
    emit(dict(kind='actual_checkpoint_official_qualification',model=layer.REPOSITORY,layer=0,tokens=16,relative_l2=rel,max_abs=float(abs(got-ref).max()),passed=passed,scope='one real layer, not full-model qualification'))
    np.savez_compressed(a.output/'official_layer0_qualification.npz',actual=got,reference=ref)
    if not passed:raise RuntimeError('Actual Qwen checkpoint layer failed official qualification')
    del official,state,oracle;gc.collect()
    operands=jax.jit(lambda p,x:adapter.tuning_operands(layer,p,x))(params,x);jax.block_until_ready(operands)
    shapes=adapter.shapes(layer)
    def errors(actual,ref):
        u,v=np.asarray(actual,dtype=np.float32),np.asarray(ref,dtype=np.float32);d=u-v
        return dict(finite=bool(np.isfinite(u).all()),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(v),1e-30)),max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))),normalized_max=float(abs(d).max()/max(abs(v).max(),1e-30)))
    def timed(fn,values,count=8):
        for _ in range(3):jax.block_until_ready(fn(*values))
        samples=[]
        for _ in range(count):
            t=time.perf_counter_ns();jax.block_until_ready(fn(*values));samples.append((time.perf_counter_ns()-t)/1e6)
        return samples
    npolicy={s:dict(arm(dtype='bfloat16'),architecture='v6e') for s in adapter.SITES}
    baseline=adapter.build(layer,npolicy);nw=baseline.prepare_weights(params)
    reference=jax.jit(baseline)(x,nw).block_until_ready()
    native=[]
    for mib in (None,48,96):
        opts={} if mib is None else {'xla_tpu_scoped_vmem_limit_kib':mib*1024}
        t=time.monotonic();exe=jax.jit(baseline,compiler_options=opts).lower(x,nw).compile();samples=timed(exe,(x,nw))
        native.append((statistics.median(samples),opts,exe))
        emit(dict(kind='native_screen',options=opts,samples_ms=samples,compile_seconds=time.monotonic()-t))
    best=min(native,key=lambda z:z[0]);choices={};failures=[]
    for dtype in ('bfloat16','float32'):
        policies={family:{s:dict(v,output_dtype=dtype) for s,v in npolicy.items()} for family in ('cubic_tuned','s1','s2')}
        for site in ('gateup','down'):
            lhs,aux=operands[site];w=getattr(params,adapter.FIELDS[site]);spec=adapter.specification(site)
            ref=make_projection(dict(npolicy[site],output_dtype=dtype),shapes[site],spec).prepared(lhs,w,**aux)
            offered=[]
            for family,depth in [('cubic_tuned',0),('s1',1),('s2',2)]:
                for bk in (1024,shapes[site][1]):
                    candidate=dict(arm((512,1024,bk),depth,'outputs',1,dtype),architecture='v6e',implementation='cubic' if depth==0 else 'current')
                    offered.append((family,candidate))
            random.Random(20261003+len(site)).shuffle(offered);winners={}
            for family,candidate in offered:
                row=dict(kind='candidate',site=site,family=family,dtype=dtype,arm=candidate)
                try:
                    projection=make_projection(candidate,shapes[site],spec);row['metadata']=projection.metadata
                    if projection.metadata['estimated_vmem_bytes']>112*1024**2:
                        row.update(status='pruned',reason='rough_vmem_estimate_above_112MiB');emit(row);continue
                    packed=projection.prepare_weights(w);jax.block_until_ready(packed)
                    def run(lhs,w,aux):return projection.prepared(lhs,w,**aux).astype(jnp.bfloat16)
                    t=time.monotonic();exe=jax.jit(run).lower(lhs,packed,aux).compile();row['compile_seconds']=time.monotonic()-t
                    got=exe(lhs,packed,aux).block_until_ready();err=errors(got,ref.astype(jnp.bfloat16));row['errors']=err
                    row['status']='eligible' if err['finite'] and err['relative_l2']<=.10 else 'numerical_rejection'
                    if row['status']=='eligible':
                        row['samples_ms']=timed(exe,(lhs,packed,aux));score=statistics.median(row['samples_ms'])
                        if family not in winners or score<winners[family][0]:winners[family]=(score,candidate)
                except Exception as e:row.update(status='failed',error_type=type(e).__name__,error=str(e)[-2000:]);failures.append(row)
                emit(row)
            for family in policies:
                if family not in winners:raise RuntimeError('No eligible '+family+' for '+site+'; cannot relabel Native')
                policies[family][site]=winners[family][1]
        choices[dtype]=policies
    # Selection is immutable before fresh randomized confirmation rounds.
    (a.output/'profiles.json').write_text(json.dumps(dict(policies=choices,native_tuned_options=best[1],fixed_native_sites=['q','k','v','o'],scope='two-site pilot only; not fully tuned six-site policies'),indent=2)+'\n')
    results=[]
    for dtype in ('bfloat16','float32'):
        executables={'native_default':(native[0][2],nw),'native_tuned':(best[2],nw)}
        for family,policy in choices[dtype].items():
            fn=adapter.build(layer,policy);pw=fn.prepare_weights(params);jax.block_until_ready(pw)
            t=time.monotonic();exe=jax.jit(fn,compiler_options=best[1]).lower(x,pw).compile()
            emit(dict(kind='block_compile',family=family,dtype=dtype,seconds=time.monotonic()-t))
            executables[family]=(exe,pw)
        for exe,pw in executables.values():timed(exe,(x,pw),count=0)
        samples={name:[] for name in executables};rng=random.Random(1003001)
        for round in range(15):
            names=list(executables);rng.shuffle(names)
            for name in names:
                exe,pw=executables[name];t=time.perf_counter_ns();y=exe(x,pw).block_until_ready();ms=(time.perf_counter_ns()-t)/1e6;samples[name].append(ms)
                emit(dict(kind='confirmation',round=round,algorithm=name,dtype=dtype,ms=ms))
        for name,(exe,pw) in executables.items():
            result=dict(algorithm=name,output_dtype=dtype,samples_ms=samples[name],median_ms=statistics.median(samples[name]),errors=errors(exe(x,pw),reference),scope='layer0 resident compute; includes consumer cast, excludes checkpoint IO and recurring transfers; Native store tracks share parent BF16 projection boundaries')
            results.append(result);emit(dict(kind='pilot_result',**result))
        del executables;gc.collect()
    (a.output/'summary.json').write_text(json.dumps(dict(status='completed',model_id=layer.REPOSITORY,revision=layer.REVISION,batch=1,sequence=512,layer=0,input_h2d_seconds=input_h2d,results=results,main_campaign_entries_completed=0,checkpoint_access='real official BF16 weights',whole_model_quality='not measured',token_sha256=hashlib.sha256(ids.tobytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
