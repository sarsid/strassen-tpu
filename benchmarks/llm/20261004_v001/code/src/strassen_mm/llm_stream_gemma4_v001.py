"""Gemma4-only streamed BF16 prefill with preserved campaign timing boundaries.

Model weights stay on the host; one prepared layer is transferred at a time.
No checkpoint/network IO or weight layout preparation occurs in timed inference.
Total elapsed forward includes embedding lookup/transfer, every layer transfer,
model compute, final normalization and all vocabulary logits (chunked by token).
Compilation is warmed separately. Resident layer sums are diagnostics, not a
claim that the complete model is resident or that its full forward is that fast.
"""
import json, math, time
from pathlib import Path
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from .model_gemma4_fused_v001 import build_layer, native_policy
from .gemma4_contract_v001 import validate_config
from .model_gemma4_v001 import head_logits

VERSION = "llm_stream_gemma4_v001"


def checkpoint(manifest):
    from .checkpoint_gemma4_v001 import Checkpoint
    return Checkpoint(manifest)


def pack_host(weights,metadata):
    """Exact host counterpart of the qualified projection packer, no JAX work."""
    result={}
    for name,value in weights.items():
        if name not in metadata['projections']:
            result[name]=np.ascontiguousarray(value);continue
        meta=metadata['projections'][name]
        if meta['implementation']=='native':result[name]=np.ascontiguousarray(value);continue
        _,k,n=meta['shape_mkn'];_,kp,npad=meta['padded_shape_mkn'];bn=meta['tile_bm_bn_bk'][1]
        spec=meta['epilogue'];kind=spec['kind'];rotary=kind in ('rope','qk_norm_rope')
        if kind in ('swiglu','geglu'):
            first,second=np.split(value,2,axis=-1);half=npad//2
            first=np.pad(first,((0,kp-k),(0,half-n//2)));second=np.pad(second,((0,kp-k),(0,half-n//2)))
        else:
            prepared=np.pad(value,((0,kp-k),(0,npad-n)))
            if not rotary:result[name]=np.ascontiguousarray(prepared);continue
            heads=prepared.reshape(kp,npad//spec['head_dim'],spec['head_dim']);first,second=np.split(heads,2,-1)
            first,second=first.reshape(kp,-1),second.reshape(kp,-1)
        result[name]=np.ascontiguousarray(np.stack((first.reshape(kp,-1,bn//2),second.reshape(kp,-1,bn//2)),axis=2).reshape(kp,npad))
    return result


def error_metrics(actual,reference):
    a,b=np.asarray(actual,dtype=np.float32),np.asarray(reference,dtype=np.float32)
    if a.shape!=b.shape:raise ValueError('Error comparison shapes differ')
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(b.ravel()),1e-30)),max_abs=float(np.max(np.abs(delta))),rmse=float(np.sqrt(np.mean(delta*delta))),normalized_max=float(np.max(np.abs(delta))/max(np.max(np.abs(b)),1e-30)))


def _summary_sums(logits,targets,reference):
    logq=jax.nn.log_softmax(logits.astype(jnp.float32),-1);rows=jnp.arange(logits.shape[0])
    result=dict(positions=jnp.asarray(logits.shape[0]),nll_sum=-jnp.sum(logq[rows,targets]),finite=jnp.all(jnp.isfinite(logits)))
    if reference is not None:
        ref=reference.astype(jnp.float32);x=logits.astype(jnp.float32);logp=jax.nn.log_softmax(ref,-1)
        result.update(finite=result['finite'] & jnp.all(jnp.isfinite(ref)),reference_nll_sum=-jnp.sum(logp[rows,targets]),kl_sum=jnp.sum(jnp.exp(logp)*(logp-logq)),top1_equal=jnp.sum(jnp.argmax(ref,-1)==jnp.argmax(x,-1)),squared_error_sum=jnp.sum((x-ref)**2),reference_squared_sum=jnp.sum(ref**2),max_abs=jnp.max(jnp.abs(x-ref)),reference_max_abs=jnp.max(jnp.abs(ref)),logit_elements=jnp.asarray(x.size))
    return result


summarize_chunk=jax.jit(_summary_sums)


def finish_metrics(chunks):
    totals={}
    for row in chunks:
        for key,value in row.items():
            if key=='finite':totals[key]=totals.get(key,True) and bool(value)
            elif key in ('max_abs','reference_max_abs'):totals[key]=max(totals.get(key,0.),float(value))
            else:totals[key]=totals.get(key,0.)+float(value)
    n=totals['positions']
    if n<=0:raise ValueError('Quality scoring requires at least one next-token position')
    if not totals['finite'] or any(not math.isfinite(float(v)) for v in totals.values()):
        return dict(positions=int(n),finite=False,status='nonfinite',nll=None,perplexity=None)
    result=dict(positions=int(n),finite=totals['finite'],nll=totals['nll_sum']/n)
    result['perplexity']=float(math.exp(result['nll'])) if result['nll']<700 else None
    if 'kl_sum' in totals:
        result.update(delta_nll=(totals['nll_sum']-totals['reference_nll_sum'])/n,mean_kl=totals['kl_sum']/n,top1_agreement=totals['top1_equal']/n,logit_relative_l2=math.sqrt(totals['squared_error_sum']/max(totals['reference_squared_sum'],1e-30)),logit_rmse=math.sqrt(totals['squared_error_sum']/totals['logit_elements']),logit_max_abs=totals['max_abs'],logit_normalized_max=totals['max_abs']/max(totals['reference_max_abs'],1e-30))
    return result


class StreamedModel:
    def __init__(self,cp,batch,sequence,profile=None,*,output_dtype='bfloat16',logit_chunk=128,interpret=False,host_layers=None,prepared_provider=None,attention_backend="xla"):
        if output_dtype != "bfloat16":raise ValueError("Gemma4 campaign supports BF16 output only")
        self.cp=cp;self.c=validate_config(cp.config);self.batch=batch;self.sequence=sequence;self.dtype=output_dtype;self.logit_chunk=logit_chunk
        if type(logit_chunk)!=int or logit_chunk<=0:raise ValueError('Positive token chunk required')
        if type(sequence)!=int or sequence<2:raise ValueError('At least two tokens required')
        if profile is None:
            entries={}
            for i,kind in enumerate(self.c['layer_types']):
                if kind not in entries:entries[kind]=dict(policy=native_policy(self.c,i),compiler_options={})
            profile=dict(by_layer_type=entries)
        self.profile=profile;self.builders={};self.functions={};self.host=[];self.metadata=[]
        start=time.perf_counter()
        for i in range(self.c['num_hidden_layers']):
            layer_type=self.c.get('layer_types',['full_attention']*self.c['num_hidden_layers'])[i]
            entry=profile.get('by_layer_type',{}).get(layer_type,profile)
            key=json.dumps([layer_type,entry],sort_keys=True)
            if key not in self.builders:
                self.builders[key]=build_layer(self.c,sequence,entry['policy'],batch_size=batch,layer_index=i,output_dtype=output_dtype,rounding='accumulator',compiler_options=entry.get('compiler_options',{}),interpret=interpret,attention_backend=attention_backend)
                self.functions[key]=self.builders[key].prepared
            fn=self.builders[key]
            if prepared_provider is None:
                weights=cp.layer(i) if host_layers is None else host_layers[i]
                prepared=pack_host(weights,fn.metadata)
            else:prepared=prepared_provider(i,fn.metadata)
            self.host.append((key,prepared))
            self.metadata.append(dict(layer=i,layer_type=layer_type,**fn.metadata))
        self.norm=np.array(cp.tensor('model.norm.weight'),copy=True)
        self.embedding=cp.tensor('model.embed_tokens.weight')
        # One shared host head buffer; transfer once per complete forward, not per token chunk.
        self.head=cp.head()
        self.setup_seconds=time.perf_counter()-start
        self.host_bytes=sum(a.nbytes for _,w in self.host for a in w.values())+self.head.nbytes+self.norm.nbytes
        self.head_fn=jax.jit(lambda hidden,norm,head:head_logits(hidden,norm,head,self.c))

    def embeddings(self,ids):
        x=np.array(self.embedding[np.asarray(ids)],copy=True)
        if self.c['model_type']=='gemma4_text':
            scale=np.asarray(math.sqrt(self.c['hidden_size']),dtype=ml_dtypes.bfloat16)
            x=(x.astype(np.float32)*scale.astype(np.float32)).astype(ml_dtypes.bfloat16)
        return x

    def forward(self,ids,*,capture=False,save_logits=None,reference_logits=None,score=False,emit=lambda row:None):
        ids=np.asarray(ids)
        if ids.shape!=(self.batch,self.sequence):raise ValueError('Tokens must match the frozen workload')
        if ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=self.c['vocab_size']:raise ValueError('Token IDs outside vocabulary')
        total_start=time.perf_counter();t=total_start
        x=jax.device_put(self.embeddings(ids));jax.block_until_ready(x)
        timings=dict(embedding_ms=(time.perf_counter()-t)*1000,layers=[],head_transfer_ms=0.,head_compute_ms=0.)
        traces=[]
        for i,(key,host) in enumerate(self.host):
            start=time.perf_counter();w=jax.device_put(host);jax.block_until_ready(w);transferred=time.perf_counter()
            x=self.functions[key](x,w);jax.block_until_ready(x);finished=time.perf_counter()
            timings['layers'].append(dict(layer=i,h2d_ms=1000*(transferred-start),resident_compute_ms=1000*(finished-transferred)))
            if capture:traces.append(np.asarray(jax.device_get(x),dtype=np.float32))
            for leaf in jax.tree_util.tree_leaves(w):leaf.delete()
            del w
            # Diagnostic callbacks are outside the individual compute/transfer
            # intervals, but their cost is part of the total elapsed forward.
            emit(dict(kind='forward_layer',**timings['layers'][-1]))
        t=time.perf_counter();norm,head=jax.device_put((self.norm,self.head));jax.block_until_ready((norm,head));timings['head_transfer_ms']=1000*(time.perf_counter()-t)
        flat=x.reshape(-1,self.c['hidden_size']);chunks=[];saved=[]
        for lo in range(0,len(flat),self.logit_chunk):
            hi=min(lo+self.logit_chunk,len(flat));t=time.perf_counter();logits=self.head_fn(flat[lo:hi],norm,head);jax.block_until_ready(logits);timings['head_compute_ms']+=1000*(time.perf_counter()-t)
            if save_logits is not None:
                Path(save_logits).mkdir(parents=True,exist_ok=True);file=Path(save_logits)/f'{lo:06d}.npy';np.save(file,np.asarray(logits,dtype=np.float32));saved.append(file.name)
            if score:
                positions=np.arange(lo,hi);valid=positions%self.sequence<self.sequence-1
                targets=ids.reshape(-1)[np.minimum(positions+1,ids.size-1)][valid]
                if valid.any():
                    reference=None if reference_logits is None else jnp.asarray(np.load(Path(reference_logits)/f'{lo:06d}.npy')[valid])
                    chunks.append(jax.device_get(summarize_chunk(logits[valid],jnp.asarray(targets),reference)))
                    if reference is not None:reference.delete()
            logits.delete()
        timings['elapsed_ms']=1000*(time.perf_counter()-total_start)
        timings['resident_layer_sum_ms']=sum(r['resident_compute_ms'] for r in timings['layers'])
        timings['layer_transfer_sum_ms']=sum(r['h2d_ms'] for r in timings['layers'])
        timings['scope']='complete streamed prompt forward; all token logits; synchronized; no transfer overlap'
        timings['instrumented_quality_pass']=bool(score or capture or save_logits is not None)
        # Scoring/capture/host IO are deliberately disallowed as inference timing claims.
        timings['eligible_for_inference_timing']=not timings['instrumented_quality_pass']
        norm.delete();head.delete();x.delete()
        return dict(timing=timings,hidden=traces,quality=finish_metrics(chunks) if score else None,logit_files=saved)

    def warmup(self,ids):
        # Real complete forward exercises all attention classes and the last logit chunk.
        return self.forward(ids)['timing']
