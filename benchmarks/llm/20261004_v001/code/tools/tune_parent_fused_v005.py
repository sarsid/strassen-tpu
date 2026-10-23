"""Interleaved shortlist refinement; preserves the complete earlier screen."""
import argparse, dataclasses, gc, hashlib, importlib, importlib.metadata as md
import json, os, random, statistics, sys, time
from pathlib import Path
import parent_stage_v001 as setup


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--screen-events',type=Path,required=True);p.add_argument('--screen-profile',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def emit(value):
        line=json.dumps(value,sort_keys=True)
        with (args.output/'events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    setup.configure(args.output)
    os.environ['STRASSEN_MANAGE_CEILING']='0'
    layer=importlib.import_module('benchmark_qwen3_32b_layer')
    stream=importlib.import_module('benchmark_qwen3_32b_streamed_inference')
    # Importing parent modules sets a global Native cap. Remove that cap BEFORE
    # device initialization; whole-function compiler_options supply each arm's cap.
    os.environ['LIBTPU_INIT_ARGS']='--xla_tpu_use_enhanced_launch_barrier=true'
    setup.install_transport(layer,args.cache,emit);layer.emit=emit;stream.emit=emit
    import jax
    import jax.numpy as jnp
    import numpy as np
    from strassen_mm import parent_fused_adapter_v001 as adapter
    from strassen_mm.kernels_fused_v003 import make_projection
    from strassen_mm.tuner_joint_v001 import arm
    import benchmark_common as common
    if len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Expected one v6e')
    emit(dict(kind='environment',versions={n:md.version(n) for n in ('jax','jaxlib','libtpu')},device=jax.devices()[0].device_kind,
              libtpu_init_args=os.environ['LIBTPU_INIT_ARGS'],timing='synchronized prepared calls; consumer cast included; weight packing excluded',
              tuning_runs=32,tuning_warmups=2,confirmation_runs=30,confirmation_warmups=3,
              selection='median of eight four-round means; rotating/reversed interleaving; no Native normalization'))
    checkpoint=layer.ShardedSafetensors()
    cos,sin=layer.rope_values()
    params=stream.load_layer(checkpoint,0,cos,sin)
    token_ids=stream.make_tokens()
    expected='70b86198317db1dee0c351568cf420d78242aa74e0b9a6cbb14925eaa55ca2a5'
    if hashlib.sha256(token_ids.tobytes()).hexdigest()!=expected:raise RuntimeError('Parent token hash mismatch')
    embedding=checkpoint.tensor('model.embed_tokens.weight')
    x=jnp.asarray(embedding[token_ids].reshape(layer.TOKENS,layer.MODEL_DIM).copy());del embedding
    operands=jax.jit(lambda p,x:adapter.tuning_operands(layer,p,x))(params,x);jax.block_until_ready(operands)
    shape_map=adapter.shapes(layer)
    native_policy={s:dict(arm(dtype='bfloat16'),architecture='v6e') for s in adapter.SITES}
    native=adapter.build(layer,native_policy);nw=native.prepare_weights(params)
    native_raw=stream.make_product_layer('regular_xla')
    raw_exe=jax.jit(native_raw).lower(x,params).compile()
    new_exe=jax.jit(native).lower(x,nw).compile()
    reference=raw_exe(x,params);candidate=new_exe(x,nw);jax.block_until_ready((reference,candidate))
    equivalent=bool(np.array_equal(np.asarray(reference),np.asarray(candidate)))
    emit(dict(kind='native_adapter_equivalence',bitwise_equal=equivalent,errors=common.device_error(candidate,reference)))
    if not equivalent:raise RuntimeError('Native adapter does not reproduce the parent computation')
    def timed(executable,values,runs=8,warmups=2):
        for _ in range(warmups):jax.block_until_ready(executable(*values))
        times=[]
        for _ in range(runs):
            started=time.perf_counter_ns();jax.block_until_ready(executable(*values));times.append((time.perf_counter_ns()-started)/1e6)
        return times
    screen_profile=json.loads(args.screen_profile.read_text())
    screen_events=[json.loads(line) for line in args.screen_events.read_text().splitlines() if line.strip()]
    from parent_shortlist_v001 import shortlist, robust_score
    best_options=screen_profile['native_compiler_options']
    emit(dict(kind='screen_provenance',events_sha256=hashlib.sha256(args.screen_events.read_bytes()).hexdigest(),
              profile_sha256=hashlib.sha256(args.screen_profile.read_bytes()).hexdigest(),native_compiler_options=best_options,
              note='Native options frozen from prior screen; all previous measurements retained, no test quality used for selection'))
    choices={dtype:{family:dict(native_policy) for family in ('cubic','s1','s2')} for dtype in ('bfloat16','float32')}
    selection=[]
    # A common Native site control retains the parent's rounded projection;
    # it may win dispatch for any family. It is never called an S1/S2 kernel.
    def native_site(site,a,w,aux):
        z=layer.native_projection(a,w)
        if site in ('q','k'):
            heads=layer.HEADS if site=='q' else layer.KV_HEADS
            z=layer.apply_rope(layer.rms_norm(z.reshape(layer.BATCH,layer.SEQUENCE,heads,layer.HEAD_DIM),aux['scale']),cos,sin).reshape(layer.TOKENS,-1)
        elif site=='gateup':
            g,u=jnp.split(z,2,-1);z=(jax.nn.silu(g.astype(jnp.float32))*u.astype(jnp.float32)).astype(jnp.bfloat16)
        elif site in ('o','down'):z=(z.astype(jnp.float32)+aux['residual'].astype(jnp.float32)).astype(jnp.bfloat16)
        return z
    for site in ('q','k','o','gateup','down'):
        a,aux=operands[site];weight=getattr(params,adapter.FIELDS[site])
        base=jax.jit(lambda a,w,aux:native_site(site,a,w,aux),compiler_options=best_options).lower(a,weight,aux).compile()
        baseline=base(a,weight,aux);jax.block_until_ready(baseline)
        base_times=timed(base,(a,weight,aux))
        emit(dict(kind='native_site',site=site,samples_ms=base_times))
        for dtype in ('bfloat16','float32'):
            selected=shortlist(screen_events,site,dtype,screen_profile)
            emit(dict(kind='shortlist',site=site,dtype=dtype,arms=[item['arm_id'] for item in selected]))
            entries={'native':(base,(a,weight,aux),None)}
            for candidate in selected:
                try:
                    projection=make_projection(candidate,shape_map[site],adapter.specification(site,candidate['early']))
                    packed=projection.prepare_weights(weight);jax.block_until_ready(packed)
                    fn=lambda a,w,aux:projection.prepared(a,w,**aux).astype(jnp.bfloat16)
                    started=time.monotonic();exe=jax.jit(fn,compiler_options=best_options).lower(a,packed,aux).compile()
                    compile_s=time.monotonic()-started
                    value=exe(a,packed,aux);jax.block_until_ready(value)
                    errors=common.device_error(value,baseline)
                    finite=bool(jax.device_get(jnp.all(jnp.isfinite(value))))
                    eligible=finite and errors['l2_relative']<0.10
                    emit(dict(kind='refinement_compile',site=site,arm=candidate,metadata=projection.metadata,
                              compile_s=compile_s,errors_vs_parent_native=errors,finite=finite,eligible=eligible))
                    if eligible:entries[candidate['arm_id']]=(exe,(a,packed,aux),candidate)
                    del value,projection,fn
                except Exception as error:
                    emit(dict(kind='refinement_failure',site=site,arm=candidate,error_type=type(error).__name__,error=str(error)[-6000:]))
            names=tuple(entries);times={name:[] for name in names}
            for run in range(-2,32):
                offset=run%len(names);order=names[offset:]+names[:offset]
                if run%2:order=tuple(reversed(order))
                for name in order:
                    exe,values,_=entries[name]
                    started=time.perf_counter_ns();jax.block_until_ready(exe(*values))
                    elapsed=(time.perf_counter_ns()-started)/1e6
                    if run>=0:times[name].append(elapsed)
            scores={name:robust_score(samples) for name,samples in times.items()}
            emit(dict(kind='refinement_measurement',site=site,dtype=dtype,samples_ms=times,
                      mean_ms={name:statistics.fmean(v) for name,v in times.items()},selection_score_ms=scores,
                      scoring='median of eight four-round means; no normalization or sample deletion'))
            for family in ('cubic','s1','s2'):
                contenders=['native']+[name for name in names if name!='native' and entries[name][2]['family']==family]
                winner=min(contenders,key=lambda name:scores[name])
                if winner!='native':choices[dtype][family][site]=entries[winner][2]
                item=dict(site=site,dtype=dtype,family=family,selected=choices[dtype][family][site],
                          score_ms=scores[winner],mean_ms=statistics.fmean(times[winner]),native_score_ms=scores['native'])
                selection.append(item);emit(dict(kind='selected',**item))
            del entries,exe,values
            gc.collect()
    profile=dict(version='parent-fused-v005-refined',model='Qwen3-32B',batch=8,sequence=1024,
                 native_compiler_options=best_options,native_selected_mib=screen_profile['native_selected_mib'],profiles=choices,selection=selection,
                 native_precision='parent BF16 projection; unchanged in both custom-store comparisons',
                 custom_precision='accumulator epilogue; FP32/BF16 stored output then BF16 consumer',
                 tuning_input='parent repetitive-text embedding activations at real layer 0',
                 screen_events_sha256=hashlib.sha256(args.screen_events.read_bytes()).hexdigest(),
                 screen_profile_sha256=hashlib.sha256(args.screen_profile.read_bytes()).hexdigest(),
                 selection_rule='interleaved 32 calls; median of eight four-round means; Native fallback eligible',
                 confirmation_input='independent parent deterministic_hidden; not used for selection')
    # Freeze before confirmation: no result below may change this selection.
    (args.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    x=layer.deterministic_hidden()
    parent_control=stream.make_product_layer('gated_strassen')
    parent_control.prepare_weights=lambda p,cache=None:p
    parent_control.metadata=dict(version='unmodified_parent_streamed_policy',product_tile=list(stream.PRODUCT_TILE),
                                 qk_tile=list(stream.QK_TILE),site_tiles=stream.SITE_TILES)
    builders={'native_default':native,'native_tuned':native,'native_parent48':native,'parent_streamed_s1':parent_control}
    options={'native_default':{},'native_tuned':best_options,
             'native_parent48':{'xla_tpu_scoped_vmem_limit_kib':49152},
             'parent_streamed_s1':{'xla_tpu_scoped_vmem_limit_kib':49152}}
    for dtype,profiles in choices.items():
        for family,policy in profiles.items():
            key=family+'_'+dtype;builders[key]=adapter.build(layer,policy);options[key]=best_options
    prepared={};executables={};cache={}
    for name,builder in builders.items():
        try:
            prepared[name]=builder.prepare_weights(params,cache)
            started=time.monotonic();executables[name]=jax.jit(builder,compiler_options=options[name]).lower(x,prepared[name]).compile()
            emit(dict(kind='confirmation_compile',arm=name,seconds=time.monotonic()-started,metadata=builder.metadata,compiler_options=options[name]))
        except Exception as error:emit(dict(kind='confirmation_failure',arm=name,error_type=type(error).__name__,error=str(error)[-6000:]))
    jax.block_until_ready(prepared)
    names=tuple(executables);samples={name:[] for name in names};outputs={}
    for run in range(-3,30):
        offset=run%len(names);order=names[offset:]+names[:offset]
        if run%2:order=tuple(reversed(order))
        for name in order:
            started=time.perf_counter_ns();value=executables[name](x,prepared[name]);jax.block_until_ready(value)
            elapsed=(time.perf_counter_ns()-started)/1e6
            if run>=0:samples[name].append(elapsed)
            outputs[name]=value
    reference=outputs['native_default']
    confirmation=dict(kind='confirmation',samples_ms=samples,mean_ms={k:statistics.fmean(v) for k,v in samples.items()},
                      errors_vs_native={k:common.device_error(v,reference) for k,v in outputs.items()},
                      selection_unchanged=True,profile_sha256=hashlib.sha256((args.output/'profile.json').read_bytes()).hexdigest())
    emit(confirmation)
    (args.output/'summary.json').write_text(json.dumps(dict(status='completed',profile=profile,confirmation=confirmation),indent=2)+'\n')


if __name__=='__main__':main()
