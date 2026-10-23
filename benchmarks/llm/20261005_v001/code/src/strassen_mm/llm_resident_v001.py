"""Layer-resident diagnostics, explicitly separate from full-forward latency.

One layer/algorithm's parameters remain resident for all repeats. Arms rotate
by layer; each arm advances its own real hidden states. No transfers, head,
embedding, scoring or host packing enter these resident samples.
"""
import gc,random,statistics,time
import jax
import jax.numpy as jnp
from .model_fused_v005 import build_layer
from .llm_weight_cache_v001 import packed_layer

def measure(cp,ids,profiles,dtype,emit,warmups=3,repeats=15):
    from .llm_stream_v002 import StreamedModel
    # Reuse the exact embedding implementation without preparing a full model.
    stub=object.__new__(StreamedModel);stub.cp=cp;stub.c=cp.config;stub.embedding=cp.tensor('model.embed_tokens.weight')
    initial=stub.embeddings(ids);names=list(profiles);hidden={n:jax.device_put(initial) for n in names};jax.block_until_ready(hidden)
    builders={};records={n:[] for n in names};rng=random.Random(20261003)
    for i in range(cp.config['num_hidden_layers']):
        kind=cp.config.get('layer_types',['full_attention']*cp.config['num_hidden_layers'])[i]
        order=list(names);rng.shuffle(order)
        for name in order:
            entry=profiles[name].get('by_layer_type',{}).get(kind,profiles[name]);key=(name,kind)
            if key not in builders:builders[key]=build_layer(cp.config,ids.shape[1],entry['policy'],batch_size=ids.shape[0],layer_index=i,output_dtype=dtype,rounding='accumulator',compiler_options=entry.get('compiler_options',{}))
            builder=builders[key];host=packed_layer(cp,i,builder.metadata);params=jax.device_put(host);jax.block_until_ready(params)
            x=hidden[name]
            for _ in range(warmups):
                y=builder.prepared(x,params);jax.block_until_ready(y);y.delete()
            samples=[]
            for repeat in range(repeats):
                started=time.perf_counter_ns();y=builder.prepared(x,params);jax.block_until_ready(y);samples.append((time.perf_counter_ns()-started)/1e6)
                if repeat<repeats-1:y.delete()
            finite=bool(jax.device_get(jnp.all(jnp.isfinite(y))));x.delete();hidden[name]=y
            row=dict(layer=i,algorithm=name,output_dtype=dtype,samples_ms=samples,median_ms=statistics.median(samples),finite=finite)
            records[name].append(row);emit(dict(kind='resident_layer',**row))
            for leaf in jax.tree_util.tree_leaves(params):leaf.delete()
            del params,host
            if not finite:raise RuntimeError('Nonfinite resident layer output')
        gc.collect()
    for value in hidden.values():value.delete()
    return {name:dict(scope='sum of per-layer resident medians; excludes transfers, embedding and final head; not full-forward latency',per_layer=rows,aggregate_ms=sum(r['median_ms'] for r in rows),warmups_per_layer=warmups,repeats_per_layer=repeats,order='seeded randomized algorithm blocks within each layer') for name,rows in records.items()}
