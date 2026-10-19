"""Shared compiled 16-query decoder blocks for paired Native/S1 accuracy.

No benchmark scores are produced here. The pilot measures the actual generation
path on train questions before any extrapolation or downstream test execution.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types
import numpy as np
import jax
import jax.numpy as jnp
from accuracy_decoder_v001 import make_layer,native_projection,geometry
from accuracy_mm_v001 import make_projection,qwen_rms

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'third_party/strassen_tpu_public_95be1fb_v001'
MODELS={'qwen3_14b':('14b','Qwen/Qwen3-14B','40c069824f4251a91eefaf281ebe4c544efd3e18'),
        'qwen3_32b':('32b','Qwen/Qwen3-32B','9216db5781bf21249d130ec9da846c4624c16137'),
        'gemma4_31b':('gemma4','google/gemma-4-31B','5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89')}
ARMS=('native','strassen')


def exclusive_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


class Engine:
    def __init__(self,model,cache,emit):
        self.emit=emit;self.model=model;short,self.repo,self.revision=MODELS[model]
        self.config=json.loads((ROOT/f'plans/llm_accuracy_20261018_v001/metadata/{model}.config.json').read_text())
        self.qwen=short!='gemma4'
        if not self.qwen:self.config=self.config['text_config']
        os.environ['QWEN3_MODEL']=short if self.qwen else '32b'
        for path in (PARENT,PARENT/'experiments/qwen3'):sys.path.insert(0,str(path))
        import benchmark_qwen3_32b_layer as layer
        from parent_stage_v001 import install_transport
        if self.qwen and (layer.REPOSITORY,layer.REVISION)!=(self.repo,self.revision):raise ValueError('Checkpoint import mismatch')
        layer.emit=emit;install_transport(layer,cache/'http',emit)
        if self.qwen:self.checkpoint=layer.ShardedSafetensors()
        else:
            import gemma4_parent_model_v002 as gemma_transport
            gemma_transport.np=np
            self.checkpoint=gemma_transport.Checkpoint(cache/'http',emit)
        self.cache=cache/'canonical';self.cache.mkdir(parents=True,exist_ok=True)
        self.weights=[];self.prepared={};self.executables={};self.caches={};self.tile=(128,256,256)
        self.functions={}
        for index in range(self.config['num_hidden_layers']):
            kind='qwen' if self.qwen else self.config['layer_types'][index]
            for arm in ARMS:
                if (arm,kind) not in self.functions:self.functions[arm,kind]=make_layer(self.config,index,make_projection(arm,self.tile))
        emit(dict(kind='engine_contract',model=self.repo,revision=self.revision,tile=list(self.tile),depth=1,
                  sites=['q','k','v','o','gate','up','down'],strassen_prefill=True,strassen_decode=True,
                  fused_epilogues=False,precision='DEFAULT',dtype='bfloat16',batch_policy='matched fixed rows',
                  query_block=16,query_block_policy='same compiled decoder for prefill and decode',
                  padding_policy='future query positions excluded by causal and sequence masks',
                  checkpoint_weights='unchanged; canonical transpose only',hardware=[str(x) for x in jax.devices()],
                  software={k:md.version(k) for k in ('jax','jaxlib','libtpu','numpy','transformers','tokenizers')}))

    def tensor(self,name,transpose=False):
        key=hashlib.sha256((self.repo+self.revision+name+str(transpose)).encode()).hexdigest()
        path=self.cache/(key+'.bin');meta=self.cache/(key+'.json')
        if meta.exists() and path.exists():
            info=json.loads(meta.read_text())
            if path.stat().st_size!=info['bytes']:raise ValueError('Canonical weight length mismatch')
        else:
            raw=self.checkpoint.tensor(name)
            if transpose:raw=np.ascontiguousarray(raw.T)
            temporary=path.with_suffix('.partial');raw.tofile(temporary)
            digest=hashlib.sha256(memoryview(np.ascontiguousarray(raw).view(np.uint8)).cast('B')).hexdigest()
            info=dict(name=name,revision=self.revision,shape=list(raw.shape),dtype=str(raw.dtype),bytes=raw.nbytes,sha256=digest,transpose=transpose)
            temporary.replace(path);exclusive_json(meta,info);del raw
        mapped=np.memmap(path,mode='r',dtype=np.dtype(info['dtype']),shape=tuple(info['shape']))
        # Verify canonical bytes once on opening, not on every token.
        if hashlib.sha256(memoryview(mapped.view(np.uint8)).cast('B')).hexdigest()!=info['sha256']:raise ValueError('Canonical weight integrity failure')
        return mapped

    def prepare(self):
        start=time.monotonic();self.emit(dict(kind='prepare_start'))
        self.embedding=self.tensor('model.embed_tokens.weight')
        self.final_norm=self.tensor('model.norm.weight');self.head=self.tensor('lm_head.weight',True)
        for index in range(self.config['num_hidden_layers']):
            prefix=f'model.layers.{index}.';p={}
            for site in (('q','k','o') if geometry(self.config,index)['shared_raw_kv'] else ('q','k','v','o')):p[site]=self.tensor(prefix+f'self_attn.{site}_proj.weight',True)
            for site in ('gate','up','down'):p[site]=self.tensor(prefix+f'mlp.{site}_proj.weight',True)
            for key,name in (('norm1','input_layernorm'),('norm2','post_attention_layernorm'),('qnorm','self_attn.q_norm'),('knorm','self_attn.k_norm')):
                p[key]=self.tensor(prefix+name+'.weight')
            if not self.qwen:
                p['norm3']=self.tensor(prefix+'pre_feedforward_layernorm.weight')
                p['norm4']=self.tensor(prefix+'post_feedforward_layernorm.weight')
                p['layer_scalar']=self.tensor(prefix+'layer_scalar')
            self.weights.append(p);self.emit(dict(kind='weights_prepared',layer=index+1,layers=self.config['num_hidden_layers']))
        self.device_head=jnp.asarray(self.head);self.device_norm=jnp.asarray(self.final_norm);jax.block_until_ready((self.device_head,self.device_norm))
        self.emit(dict(kind='prepare_complete',seconds=time.monotonic()-start))

    def reset(self,batch,capacity):
        self.batch=batch;self.capacity=capacity
        shapes=[(batch,capacity,geometry(self.config,i)['kv_heads'],geometry(self.config,i)['head_dim']) for i in range(len(self.weights))]
        import ml_dtypes
        self.caches={arm:[(np.zeros(shape,dtype=ml_dtypes.bfloat16),np.zeros(shape,dtype=ml_dtypes.bfloat16))
                          for shape in shapes] for arm in ARMS}
        self.emit(dict(kind='kv_cache_reset',batch=batch,capacity=capacity,host_bytes=4*len(ARMS)*sum(math.prod(shape) for shape in shapes)))

    def executable(self,arm,args,index):
        kind='qwen' if self.qwen else self.config['layer_types'][index]
        signature=(arm,kind,args[0].shape,args[2].shape)
        if signature not in self.executables:
            start=time.monotonic();self.emit(dict(kind='compile_start',arm=arm,input_shape=list(args[0].shape),cache_shape=list(args[2].shape)))
            self.executables[signature]=jax.jit(self.functions[arm,kind]).lower(*args).compile()
            self.emit(dict(kind='compile_complete',arm=arm,seconds=time.monotonic()-start))
        return self.executables[signature]

    def head_logits(self,value):
        if self.qwen:
            value=qwen_rms(value,self.device_norm,self.config['rms_norm_eps'])
            return native_projection(value,self.device_head)
        from strassen_mm.model_gemma4_v001 import head_logits
        return head_logits(value,self.device_norm,self.device_head,self.config)

    def forward(self,tokens,positions,lengths,*,last_indices=None,label=None,return_hidden=False,retain_cache=True):
        start=time.monotonic();initial_compiles=len(self.executables)
        hidden={arm:jnp.asarray(np.asarray(self.embedding[tokens[arm]])) for arm in ARMS}
        if not self.qwen:
            from strassen_mm.model_gemma4_v001 import scaled_embeddings
            hidden={arm:scaled_embeddings(h,self.config['hidden_size']) for arm,h in hidden.items()}
        pids={arm:jnp.asarray(positions[arm],jnp.int32) for arm in ARMS}
        lens={arm:jnp.asarray(lengths[arm],jnp.int32) for arm in ARMS}
        transfer=compute=0.
        for index,host_p in enumerate(self.weights):
            t=time.monotonic();params={k:jnp.asarray(v) for k,v in host_p.items()};jax.block_until_ready(params);transfer+=time.monotonic()-t
            for arm in ARMS:
                t=time.monotonic();ck,cv=(jnp.asarray(v) for v in self.caches[arm][index]);jax.block_until_ready((ck,cv));transfer+=time.monotonic()-t
                value=hidden[arm];size=value.shape[1];width=16;pad=(-size)%width
                value=jnp.pad(value,((0,0),(0,pad),(0,0)));positions=pids[arm]
                if pad:
                    extra=positions[:,-1:]+jnp.arange(1,pad+1,dtype=jnp.int32)[None]
                    positions=jnp.concatenate((positions,extra),axis=1)
                outputs=[];t=time.monotonic()
                for offset in range(0,size+pad,width):
                    args=(value[:,offset:offset+width],params,ck,cv,positions[:,offset:offset+width],lens[arm])
                    exe=self.executable(arm,args,index)
                    out,ck,cv=exe(*args);jax.block_until_ready((out,ck,cv));outputs.append(out)
                hidden[arm]=jnp.concatenate(outputs,axis=1)[:,:size]
                jax.block_until_ready(hidden[arm]);compute+=time.monotonic()-t
                if retain_cache:self.caches[arm][index]=jax.device_get((ck,cv))
                del ck,cv,args,out,outputs,value,positions
            del params
            self.emit(dict(kind='layer_progress',phase=label,layer=index+1,layers=len(self.weights)))
        logits={}
        for arm in ARMS:
            value=hidden[arm]
            if last_indices is not None:value=value[jnp.arange(self.batch),jnp.asarray(last_indices[arm])]
            if return_hidden:logits[arm]=value;continue
            logits[arm]=np.asarray(self.head_logits(value),dtype=np.float32)
        elapsed=time.monotonic()-start
        self.emit(dict(kind='forward_complete',phase=label,seconds=elapsed,weight_and_kv_h2d_seconds=transfer,
                       decoder_compute_seconds=compute,includes_compilation=len(self.executables)>initial_compiles))
        return logits,elapsed,len(self.executables)>initial_compiles


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,default='qwen3_14b')
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--tokens',type=int,default=16);a=p.parse_args()
    if jax.default_backend()!='tpu' or len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Expected single v6e')
    a.output.mkdir(parents=True,exist_ok=False)
    def emit(value):
        value=dict(utc=datetime.now(timezone.utc).isoformat(),**value)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        print(json.dumps(value,allow_nan=False),flush=True)
    data=json.loads(a.inputs.read_text())
    if data.get('scope')!='train_generation_timing_pilot' or data['model']!=MODELS[a.model][1]:raise ValueError('Use explicitly separated train-only pilot inputs')
    engine=Engine(a.model,a.cache,emit);engine.prepare()
    rows=data['token_ids'];b=len(rows);lengths=np.array(list(map(len,rows)),np.int32)
    width=128*math.ceil(max(lengths)/128);capacity=128*math.ceil((max(lengths)+1024)/128)
    ids=np.zeros((b,width),np.int32)
    for i,row in enumerate(rows):ids[i,:len(row)]=row
    engine.reset(b,capacity)
    state={arm:[] for arm in ARMS};times=[]
    tokens={arm:ids for arm in ARMS};positions={arm:np.tile(np.arange(width,dtype=np.int32),(b,1)) for arm in ARMS}
    current_lengths={arm:lengths.copy() for arm in ARMS};last={arm:lengths-1 for arm in ARMS}
    for step in range(a.tokens):
        logits,seconds,compiled=engine.forward(tokens,positions,current_lengths,last_indices=last,label='prefill' if step==0 else f'decode_{step}')
        if step and not compiled:times.append(seconds)
        for arm in ARMS:
            selected=np.argmax(logits[arm],axis=-1).astype(np.int32);state[arm].append(selected.tolist())
            tokens[arm]=selected[:,None];positions[arm]=current_lengths[arm][:,None].copy()
            current_lengths[arm]=current_lengths[arm]+1;last[arm]=np.zeros(b,np.int32)
        emit(dict(kind='pilot_token',step=step+1,tokens=a.tokens,seconds=seconds,includes_compilation=compiled))
    summary=dict(scope='train_generation_timing_pilot; not downstream accuracy',model=engine.repo,batch=b,
                 input_lengths=lengths.tolist(),capacity=capacity,generated_tokens=a.tokens,
                 paired_decode_step_seconds=times,paired_decode_step_median_seconds=statistics.median(times) if times else None,
                 token_ids_by_step=state,checkpoint_revision=engine.revision,completed=True)
    exclusive_json(a.output/'summary.json',summary);emit(dict(kind='pilot_complete',summary='summary.json'))


if __name__=='__main__':main()
