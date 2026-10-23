"""Bounded v6e projection search with inspectable candidate dispositions.

The shortlist is independent of measured speed or held-out data. It covers
full/blocked contraction, both accumulator strategies and buffer counts before
spending remaining slots on diverse tiles. Actual timings select each winner.
"""
import dataclasses,json,random,statistics,time
import jax
import jax.numpy as jnp
import numpy as np
from .llm_campaign_v002 import candidates,geometry
from .model_fused_v005 import build_layer,native_policy,SITES
from .kernels_fused_v004 import make_projection
from .llm_stream_v001 import error_metrics


def shortlist(rows,limit=12,protected_tiles=()):
    offered=[r for r in rows if r['disposition']=='offered']
    def category(r):
        a=r['arm'];return (a['implementation'],a['accumulator'],a['buffers'],r['metadata']['full_contraction'])
    def rank(r):
        a=r['arm'];bm,bn,bk=a['tile']
        return (not r['preferred_v6e_256_leaf'],r['padded_work_ratio'],abs(bm-1024)+abs(bn-1024),r['metadata']['estimated_vmem_bytes'],r['candidate_id'])
    groups={}
    for r in offered:groups.setdefault(category(r),[]).append(r)
    for values in groups.values():values.sort(key=rank)
    protected=set(tuple(t) for t in protected_tiles)
    chosen=[]
    for values in groups.values():
        for row in list(values):
            if tuple(row['arm']['tile']) in protected and len(chosen)<limit:
                chosen.append(row);values.remove(row);row['protected_reason']='parent_v6e_profile_compatible_geometry'
    while len(chosen)<limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen)<limit:chosen.append(groups[key].pop(0))
    selected={r['candidate_id'] for r in chosen}
    for r in offered:
        r['search_disposition']='selected_for_measurement' if r['candidate_id'] in selected else 'outside_bounded_search'
    return chosen


def timed(fn,args,warmups=3,repeats=8):
    for _ in range(warmups):jax.block_until_ready(fn(*args))
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();jax.block_until_ready(fn(*args));values.append((time.perf_counter_ns()-start)/1e6)
    return values


def tune(cp,ids,dtype,output,emit,*,limit=12):
    """Calibrate on real layer-0 activations; full held-out model is separate.

    Tiles depend on static geometry. Gemma gets a separate representative of
    each attention class. Native options are selected on the complete layer;
    custom site choices use projection times and a complete-layer smoke gate.
    A site without any eligible candidate is explicitly recorded as Native.
    """
    from pathlib import Path
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    batch,sequence=ids.shape;c,shapes,specs=geometry(cp.config,batch,sequence)
    types=c.get('layer_types',['full_attention']*c['num_hidden_layers'])
    representatives={kind:types.index(kind) for kind in sorted(set(types))}
    profiles={name:dict(by_layer_type={},output_dtype=dtype) for name in ('native_default','native_tuned','cubic_tuned','s1','s2')}
    x=jnp.asarray(cp.embeddings(ids))
    # cp.embeddings applies Gemma's BF16 sqrt(hidden) scaling.
    for i in range(max(representatives.values())+1):
        kind=types[i];weights={k:jnp.asarray(v) for k,v in cp.layer(i).items()}
        baseline=build_layer(c,sequence,batch_size=batch,layer_index=i,output_dtype=dtype,capture=True)
        (y,traces)=baseline(x,weights);jax.block_until_ready((y,traces))
        if representatives[kind]!=i:
            x=y;del traces,weights;continue
        default=dict(policy=native_policy(dtype),compiler_options={})
        profiles['native_default']['by_layer_type'][kind]=default
        native_scores=[]
        for candidate in candidates(shapes['gateup'],specs['gateup'],dtype,'native_tuned'):
            options=candidate['arm']['compiler_options'];row=dict(kind='native_screen',attention_type=kind,layer=i,output_dtype=dtype,options=options)
            try:
                fn=build_layer(c,sequence,batch_size=batch,layer_index=i,output_dtype=dtype,compiler_options=options)
                start=time.monotonic();exe=fn.lower(x,weights).compile();row['compile_seconds']=time.monotonic()-start
                row['samples_ms']=timed(exe,(x,weights));row['status']='eligible';native_scores.append((statistics.median(row['samples_ms']),options))
            except Exception as error:row.update(status='failed',error_type=type(error).__name__,error=str(error)[-2000:])
            emit(row)
        if not native_scores:raise RuntimeError('No qualified Native complete-layer configuration')
        options=min(native_scores,key=lambda v:v[0])[1]
        profiles['native_tuned']['by_layer_type'][kind]=dict(policy=native_policy(dtype),compiler_options=options)
        selected={family:dict(policy=native_policy(dtype),compiler_options=options,fallbacks={}) for family in ('cubic_tuned','s1','s2')}
        for site in SITES:
            values=traces[site];lhs,aux=values['input'],values['aux'];reference=values['output'].astype(jnp.bfloat16)
            for family in selected:
                menu=list(candidates(shapes[site],specs[site],dtype,family))
                parent=[]
                if c['model_type']=='qwen3' and c['hidden_size']==5120 and c['num_attention_heads']==64:
                    parent=[(2048,2048,512)] if family=='cubic_tuned' else [(2048,512,8192) if site=='o' else (2048,2560,1024) if site=='down' else (2048,1024,5120)]
                search=shortlist(menu,limit,parent)
                (output/f'{kind}-{site}-{family}-menu.json').write_text(json.dumps(menu,indent=2)+'\n')
                random.Random(20261003+i+sum(map(ord,site+family+dtype))).shuffle(search);scores=[]
                for candidate in search:
                    row=dict(kind='candidate',attention_type=kind,layer=i,site=site,output_dtype=dtype,family=family,candidate_id=candidate['candidate_id'],arm=candidate['arm'])
                    try:
                        projection=make_projection(candidate['arm'],shapes[site],specs[site]);packed=projection.prepare_weights(weights[site]);jax.block_until_ready(packed)
                        def call(a,w,aux):return projection.prepared(a,w,**aux).astype(jnp.bfloat16)
                        start=time.monotonic();exe=jax.jit(call).lower(lhs,packed,aux).compile();row['compile_seconds']=time.monotonic()-start
                        result=exe(lhs,packed,aux);jax.block_until_ready(result);error=error_metrics(result,reference)
                        row.update(metadata=projection.metadata,errors=error,status='eligible' if error['finite'] and error['relative_l2']<=.1 else 'numerical_rejection')
                        if row['status']=='eligible':
                            row['samples_ms']=timed(exe,(lhs,packed,aux));scores.append((statistics.median(row['samples_ms']),candidate['arm']))
                        del packed,exe,result
                    except Exception as error:row.update(status='failed',error_type=type(error).__name__,error=str(error)[-2000:])
                    emit(row)
                if scores:selected[family]['policy'][site]=min(scores,key=lambda v:v[0])[1]
                else:selected[family]['fallbacks'][site]='No numerically eligible candidate within the declared search budget'
        for family,entry in selected.items():
            # Whole-layer compilation can fail even if isolated projections fit.
            # Preserve that failure; never silently replace the whole policy.
            fn=build_layer(c,sequence,entry['policy'],batch_size=batch,layer_index=i,output_dtype=dtype,rounding='accumulator',compiler_options=options)
            prepared=fn.prepare_weights(weights);jax.block_until_ready(prepared)
            start=time.monotonic();exe=fn.prepared.lower(x,prepared).compile()
            result=exe(x,prepared);jax.block_until_ready(result)
            row=dict(kind='tuned_layer_gate',family=family,attention_type=kind,output_dtype=dtype,compile_seconds=time.monotonic()-start,errors=error_metrics(result,y),fallbacks=entry['fallbacks'])
            emit(row)
            if not row['errors']['finite']:raise RuntimeError('Nonfinite calibrated complete layer')
            entry['selected_custom_sites']=[site for site,a in entry['policy'].items() if a['implementation']!='native']
            profiles[family]['by_layer_type'][kind]=entry
        x=y;del traces,weights
    result=dict(profiles=profiles,scope='v6e bounded calibration search; independent confirmation and held-out quality required',representative_layers=representatives,per_site_family_budget=limit,native_option_selection_scope='complete resident layer',custom_selection_scope='isolated fused projection with BF16 consumer cast',calibration_token_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest())
    (output/'profiles.json').write_text(json.dumps(result,indent=2)+'\n')
    return profiles
