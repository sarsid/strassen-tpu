"""Real-checkpoint CPU/Torch/TPU arithmetic diagnosis; never starts tuning."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
import gemma4_numerics_probe_v001 as probe
import gemma4_batch_stage_v002 as stage

model=probe.model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='oracle');ap.add_argument('--batch',type=int,default=1);ap.add_argument('--sequence',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--progress',type=Path);a=ap.parse_args()
    if a.stage!='oracle' or a.batch!=1 or a.sequence!=32:raise ValueError('Diagnosis only: B1/S32 oracle')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();completion={'status':'failed'}
    def emit(value):
        row=dict(value,stage='oracle',batch=None,sequence=32,utc=datetime.now(timezone.utc).isoformat())
        line=json.dumps(row,allow_nan=False)
        with (a.output/'events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
        if a.progress:
            old=json.loads(a.progress.read_text()) if a.progress.exists() else {}
            if row['kind'] not in ('tensor','checkpoint_transport'):
                old.update(status='running',stage='oracle',stage_label='Matched-operation numerical diagnosis',test=row.get('operation',row.get('test',row['kind'])),event=row['kind'],elapsed_stage_seconds=time.monotonic()-started)
                for key in ('layer','attention_type','operation'): 
                    if key in row:old[key]=row[key]
            old['utc']=row['utc'];tmp=a.progress.with_suffix('.tmp');tmp.write_text(json.dumps(old)+'\n');tmp.replace(a.progress)
    try:
        model.setup();jax,jnp=model.jax,model.jnp;stage.jax,stage.jnp,stage.np=jax,jnp,np
        plan=json.loads(a.plan.read_text());selected=set(plan['diagnostic_layers'])
        versions={x:md.version(x) for x in ('jax','jaxlib','libtpu','transformers','torch')}
        if any(versions[k]!=v for k,v in plan['software'].items()):raise RuntimeError('Software version mismatch')
        if hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()!='6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935':raise RuntimeError('Official source mismatch')
        if 'v6' not in jax.devices('tpu')[0].device_kind.lower():raise RuntimeError('Require v6e')
        torch.set_num_threads(min(8,os.cpu_count() or 1));inspector=probe.Probe(('cpu','tpu'));cfg=Gemma4TextConfig(**model.config);cfg._attn_implementation='eager'
        emit(dict(kind='environment',versions=versions,devices={k:str(v) for k,v in inspector.devices.items()},scope='diagnosis only; no tuning or timings'))
        ids=stage.tokens(1,32,emit);tok=stage.tokenizer()
        emit(dict(kind='tokenization_audit',first_token=int(ids[0,0]),bos_token_id=tok.bos_token_id,starts_with_bos=bool(ids[0,0]==tok.bos_token_id),first_eight_token_ids=ids[0,:8].tolist(),repetitive_add_special_tokens=True,natural_loader_add_special_tokens=False))
        checkpoint=model.Checkpoint(a.cache,emit)
        values=checkpoint.tensor('model.embed_tokens.weight')[ids].copy()
        hidden=torch.from_numpy(values.view(np.uint16)).view(torch.bfloat16);hidden=hidden*torch.tensor(math.sqrt(cfg.hidden_size),dtype=torch.bfloat16);del values
        states={b:inspector.array(hidden.reshape(32,-1),b) for b in inspector.devices};compiled={};layers=[];operations=[]
        rotary=official.Gemma4TextRotaryEmbedding(cfg);position=torch.arange(32)[None]
        mask=torch.full((1,1,32,32),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(1)
        for index in range(60):
            kind=model.geometry(model.config,index)['layer_type'];emit(dict(kind='oracle_layer_start',layer=index,attention_type=kind,test='Loading real layer'))
            raw=model.raw_layer(checkpoint,index);block=probe.official_layer(raw,index,cfg);before=hidden;pos=rotary(hidden,position,kind)
            if index in selected:
                records=probe.capture(block,before,pos,mask);hidden=records['output']
                rows=inspector.layer(records,index,emit);operations.extend(rows)
                # Keep public-prompt activations, never checkpoint parameters.
                tensors={}
                for name,record in records.items():
                    if isinstance(record,dict):
                        for field,value in record.items():
                            if field!='weight' and isinstance(value,torch.Tensor):tensors[name+'.'+field]=value.detach().contiguous().view(torch.uint16).numpy()
                    elif isinstance(record,torch.Tensor):tensors[name]=record.detach().contiguous().view(torch.uint16).numpy()
                np.savez_compressed(a.output/f'official-activations-layer-{index:02d}.npz',**tensors)
                del records,tensors
            else:
                with torch.no_grad():hidden=block(before,position_embeddings=pos,attention_mask=mask)
            result=dict(kind='oracle_layer',layer=index,attention_type=kind,backends={})
            for backend,device in inspector.devices.items():
                with jax.default_device(device):
                    p={k:jnp.asarray(v) for k,v in raw.items()};p['cos'],p['sin']=model.native.rotary_tables(model.config,index,jnp.arange(32)[None])
                    key=(backend,kind)
                    if key not in compiled:
                        emit(dict(kind='compile_start',layer=index,test='Compiling Native '+backend))
                        compiled[key]=jax.jit(model.build_layer('native',model.default_profile(1,32),index,1,32)).lower(states[backend],p).compile()
                    isolated=compiled[key](inspector.array(before.reshape(32,-1),backend),p)
                    states[backend]=compiled[key](states[backend],p);jax.block_until_ready(states[backend])
                    result['backends'][backend]=dict(isolated=probe.metrics(hidden.reshape(32,-1),isolated),propagated=probe.metrics(hidden.reshape(32,-1),states[backend]))
                    del p,isolated
            result['cpu_vs_tpu']=probe.metrics(states['cpu'],states['tpu']);layers.append(result);emit(result)
            stage.save(a.output/f'layer-{index:02d}.json',result)
            del block,raw,before;gc.collect()
        norm=checkpoint.tensor('model.norm.weight');embedding=checkpoint.tensor('lm_head.weight')
        to_torch=lambda x:torch.from_numpy(x.view(np.uint16).copy()).view(torch.bfloat16)
        normfn=official.Gemma4RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).to(torch.bfloat16);normfn.weight=torch.nn.Parameter(to_torch(norm),requires_grad=False)
        with torch.no_grad():
            ref=torch.nn.functional.linear(normfn(hidden),to_torch(embedding));ref=torch.tanh(ref/cfg.final_logit_softcapping)*cfg.final_logit_softcapping
        reference=probe.host(ref).reshape(32,-1);predictive={}
        def lp(v):
            v=v.astype(np.float64);v-=v.max(-1,keepdims=True);return v-np.log(np.exp(v).sum(-1,keepdims=True))
        ref_lp=lp(reference[:-1]);targets=ids[0,1:];positions=np.arange(31)
        for backend,device in inspector.devices.items():
            with jax.default_device(device):
                logits=jax.jit(lambda h,n,w:model.native.head_logits(h,n,w,model.config))(states[backend],jnp.asarray(norm),jnp.asarray(embedding.T.copy()))
                candidate=probe.host(logits);q=lp(candidate[:-1]);same=ref_lp.argmax(-1)==q.argmax(-1)
                predictive[backend]=dict(logits=probe.metrics(reference,candidate),absolute_loss_delta=abs(float(np.mean(ref_lp[positions,targets]-q[positions,targets]))),mean_kl_nats=float(np.mean(np.sum(np.exp(ref_lp)*(ref_lp-q),axis=-1))),top1_agreement=float(same.mean()),disagreements=[])
                for t in np.flatnonzero(~same):
                    best=np.argsort(ref_lp[t])[-3:][::-1];predictive[backend]['disagreements'].append(dict(position=int(t),reference_tokens=best.tolist(),reference_logits=reference[t,best].tolist(),candidate_tokens=np.argsort(q[t])[-3:][::-1].tolist(),candidate_logits_at_reference_tokens=candidate[t,best].tolist()))
                emit(dict(kind='predictive_diagnostic',backend=backend,**predictive[backend]));del logits,candidate,q
        summary=dict(scope='diagnosis only; no performance experiment',layers=layers,operations=operations,predictive=predictive,all_pass=None,diagnosis_complete=True)
        stage.save(a.output/'summary.json',summary);completion['status']='completed'
    except BaseException as exc:
        completion.update(error_type=type(exc).__name__,error=str(exc));emit(dict(kind='stage_failure',**completion));raise
    finally:
        completion['wall_seconds']=time.monotonic()-started;stage.save(a.output/'completion.json',completion)


if __name__=='__main__':main()
