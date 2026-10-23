"""Independent CPU qualification of complete checkpoint-to-logit streaming.

Small synthetic checkpoints test implementation semantics, not LLM quality or
TPU speed. Official Transformers is the reference, including embedding scaling,
nonunit norms, grouped-query heads, Gemma local/global attention and softcapping.
"""
import argparse, hashlib, json, os, tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import torch,transformers,jax
    import jax.numpy as jnp
    import numpy as np
    from strassen_mm.llm_stream_v001 import checkpoint,StreamedModel,error_metrics,pack_host,finish_metrics
    from strassen_mm.model_fused_v005 import build_layer,native_policy,SITES
    from strassen_mm.tuner_joint_v001 import arm
    from strassen_mm.llm_host_pool_v001 import ModelPool
    from strassen_mm.llm_campaign_v002 import candidates,geometry
    from strassen_mm.tuner_llm_v001 import shortlist
    assert transformers.__version__=='4.56.2' and torch.__version__.split('+')[0]=='2.8.0'
    torch.set_num_threads(1);torch.manual_seed(20261003);rows=[]
    def record(kind,passed,**data):
        row=dict(kind=kind,status='passed' if passed else 'failed',**data)
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
        (a.output/'cases.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    for family,Config,Model in [('qwen3',transformers.Qwen3Config,transformers.Qwen3ForCausalLM),('mistral',transformers.MistralConfig,transformers.MistralForCausalLM),('gemma3_text',transformers.Gemma3TextConfig,transformers.Gemma3ForCausalLM)]:
        try:
            extra=dict(rope_scaling={'rope_type':'linear','factor':8.},final_logit_softcapping=3.) if family=='gemma3_text' else {}
            cfg=Config(vocab_size=97,hidden_size=96,intermediate_size=192,num_hidden_layers=6,
                num_attention_heads=2,num_key_value_heads=1,head_dim=32,sliding_window=16 if family=='gemma3_text' else None,
                rms_norm_eps=1e-6,rope_theta=1000000.,query_pre_attn_scalar=32,**extra)
            cfg._attn_implementation='eager';model=Model(cfg).to(torch.bfloat16).eval()
            with torch.no_grad():
                for name,value in model.named_parameters():
                    if 'norm' in name:value.copy_(torch.linspace(-.2,.3,value.numel()).reshape(value.shape)+(0 if family=='gemma3_text' else 1))
            hidden={};handles=[]
            for i,layer in enumerate(model.model.layers):
                def post(mod,args,value,i=i):hidden[i]=(value[0] if isinstance(value,tuple) else value).detach().float().numpy()
                handles.append(layer.register_forward_hook(post))
            ids=np.random.default_rng(20261003).integers(3,97,(2,40),dtype=np.int32)
            with torch.inference_mode():ref=model(torch.tensor(ids),use_cache=False).logits.float().numpy()
            for handle in handles:handle.remove()
            reference_logp=torch.log_softmax(torch.from_numpy(ref[:,:-1]),-1)
            refnll=float(-reference_logp.gather(-1,torch.tensor(ids[:,1:],dtype=torch.int64).unsqueeze(-1)).mean())
            with tempfile.TemporaryDirectory(prefix='stream-qualification-') as temporary:
                root=Path(temporary);weights=root/'weights';model.save_pretrained(weights)
                config=json.loads((weights/'config.json').read_text())
                files=[dict(path=str(f.relative_to(weights)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(weights.iterdir()) if f.name=='config.json' or f.suffix=='.safetensors']
                manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(config=config,revision='0'*40,cache_dir=str(weights),files=files,scope='generated fixture; not an official pretrained checkpoint')))
                cp=checkpoint(manifest)
                for dtype in ('bfloat16','float32'):
                    # Chunk 13 crosses sequence boundaries and leaves a final
                    # singleton containing no next-token scoring position.
                    stream=StreamedModel(cp,2,40,output_dtype=dtype,logit_chunk=13)
                    refs=root/'official';refs.mkdir(exist_ok=True)
                    flat=ref.reshape(-1,97)
                    for lo in range(0,len(flat),13):np.save(refs/f'{lo:06d}.npy',flat[lo:lo+13])
                    result=stream.forward(ids,capture=True,score=True,save_logits=root/dtype,reference_logits=refs)
                    for i,(actual,expected) in enumerate(zip(result['hidden'],hidden.values())):
                        error=error_metrics(actual,expected)
                        record('whole_model_hidden',error['finite'] and error['relative_l2']<=.02,family=family,dtype=dtype,layer=i,**error)
                    logits=np.concatenate([np.load(root/dtype/name) for name in result['logit_files']]).reshape(ref.shape)
                    error=error_metrics(logits,ref);quality=result['quality']
                    # Tiny random models have unstable top-1 ties, so logit
                    # semantic error/NLL, not production quality gates, apply.
                    record('whole_model_logits',error['finite'] and error['relative_l2']<=.03 and abs(quality['nll']-refnll)<=.01,family=family,dtype=dtype,**error,quality=quality,official_nll=refnll)
                    record('quality_boundaries',quality['positions']==78 and not result['timing']['eligible_for_inference_timing'],family=family,dtype=dtype,positions=quality['positions'])
                    clean=stream.forward(ids)['timing']
                    record('clean_timing_scope',clean['eligible_for_inference_timing'] and len(clean['layers'])==6 and clean['elapsed_ms']>=clean['resident_layer_sum_ms'],family=family,dtype=dtype)
                    # Instrumentation must not alter Native model results.
                    builder=build_layer(cp.config,40,batch_size=2,output_dtype=dtype,capture=True)
                    natural={k:jnp.asarray(v) for k,v in cp.layer(0).items()};x=jnp.asarray(stream.embeddings(ids))
                    y,traces=builder(x,natural);jax.block_until_ready((y,traces))
                    record('activation_capture',set(traces)==set(SITES) and np.array_equal(np.asarray(y,dtype=np.float32),result['hidden'][0]),family=family,dtype=dtype)
                    del stream
                weights0=cp.layer(0)
                for depth in (0,1,2):
                    policy={s:dict(arm((256,512,512),depth,dtype='bfloat16'),architecture='v6e') for s in SITES}
                    if depth==0:
                        for choice in policy.values():choice['implementation']='cubic'
                    builder=build_layer(cp.config,40,policy,batch_size=2,interpret=True)
                    expected=builder.prepare_weights({k:jnp.asarray(v) for k,v in weights0.items()})
                    actual=pack_host(weights0,builder.metadata)
                    for site in SITES:
                        record('host_weight_layout',np.array_equal(actual[site].view(np.uint16),np.asarray(expected[site]).view(np.uint16)),family=family,depth=depth,site=site)
                profiles={name:dict(policy=native_policy(),compiler_options={}) for name in ('first','second')}
                pool=ModelPool(cp,2,40,profiles,'bfloat16')
                first,_=pool.activate('first');first_quality=first.forward(ids,score=True)['quality']
                second,_=pool.activate('second');second_quality=second.forward(ids,score=True)['quality']
                evicted=not first.host and first.head is None
                restored,_=pool.activate('first');restored_quality=restored.forward(ids,score=True)['quality']
                record('bounded_host_pool',evicted and first_quality==second_quality==restored_quality,family=family)
                pool.close()
                _,shapes,specs=geometry(cp.config,2,512)
                for site in SITES:
                    for algorithm in ('cubic_tuned','s1','s2'):
                        menu=list(candidates(shapes[site],specs[site],'bfloat16',algorithm));chosen=shortlist(menu,12)
                        def categories(rows):return {(r['arm']['implementation'],r['arm']['accumulator'],r['arm']['buffers'],r['metadata']['full_contraction']) for r in rows}
                        offered=[r for r in menu if r['disposition']=='offered']
                        record('bounded_search_coverage',len(chosen)<=12 and categories(chosen)==categories(offered),family=family,site=site,algorithm=algorithm)
                jax.clear_caches()
        except Exception as error:
            record('exception',False,family=family,error_type=type(error).__name__,error=str(error)[-2000:])
    nonfinite=finish_metrics([dict(positions=1,finite=False,nll_sum=float('nan'))])
    record('nonfinite_result',nonfinite['finite'] is False and nonfinite['nll'] is None)
    summary=dict(scope='CPU full-model semantics with generated small weights; no real-model quality or TPU timing claim',passed=sum(r['status']=='passed' for r in rows),failed=sum(r['status']=='failed' for r in rows),versions=dict(jax=jax.__version__,torch=torch.__version__,transformers=transformers.__version__))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    return int(summary['failed']>0)

if __name__=='__main__':raise SystemExit(main())
