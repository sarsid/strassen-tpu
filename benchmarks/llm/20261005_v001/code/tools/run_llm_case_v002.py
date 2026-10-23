"""One matched real-model workload on one v6e, all five arms and both stores.

Prerequisite independent official BF16 checkpoint qualification. Profiles freeze
before held-out scoring or confirmation. Results are emitted atomically only
after all five matched arms have 15 fresh synchronized confirmation samples.
"""
import argparse,gc,hashlib,json,os,random,statistics,time,tarfile
from datetime import datetime,timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output-dtypes',nargs='+',choices=['bfloat16','float32']);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def emit(row):
        row['utc']=datetime.now(timezone.utc).isoformat();text=json.dumps(row,allow_nan=False)
        with (a.output/'events.jsonl').open('a') as f:f.write(text+'\n')
        print(text,flush=True)
    import jax,jax.numpy as jnp,numpy as np
    from strassen_mm.llm_stream_v001 import checkpoint,StreamedModel,error_metrics
    from strassen_mm.llm_host_pool_v001 import ModelPool
    from strassen_mm.model_fused_v005 import build_layer
    from strassen_mm.tuner_llm_v001 import tune
    from strassen_mm.llm_campaign_v002 import identity
    protocol=json.loads(a.protocol.read_text());inputs=json.loads((a.inputs/'summary.json').read_text());model=inputs['model']
    devices=jax.devices()
    if len(devices)!=1 or 'v6' not in devices[0].device_kind.lower():raise RuntimeError('This runner requires one v6e TPU')
    emit(dict(kind='device',device_kind=devices[0].device_kind,jax=jax.__version__))
    cp=checkpoint(a.private/'checkpoint-manifest.json');train=np.load(a.inputs/'train-tokens.npy');test=np.load(a.inputs/'test-tokens.npy');oracle_ids=np.load(a.inputs/'oracle-tokens.npy')
    gates=[]
    for i in range(cp.config['num_hidden_layers']):
        x=jnp.asarray(np.load(a.inputs/'official'/f'layer-{i:03d}-input.npy'),jnp.bfloat16)
        w={k:jnp.asarray(v) for k,v in cp.layer(i).items()};fn=build_layer(cp.config,oracle_ids.shape[1],batch_size=1,layer_index=i)
        y=fn(x,w);jax.block_until_ready(y);errors=error_metrics(y,np.load(a.inputs/'official'/f'layer-{i:03d}-output.npy'))
        row=dict(kind='official_checkpoint_layer_gate',layer=i,errors=errors,passed=errors['finite'] and errors['relative_l2']<=protocol['gates']['native_layer_relative_l2_max']);gates.append(row);emit(row)
        for leaf in jax.tree_util.tree_leaves((x,w,y)):leaf.delete()
        del x,w,y,fn
    (a.output/'official-checkpoint-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
    if not all(row['passed'] for row in gates):raise RuntimeError('Actual-checkpoint Native semantics gate failed; tuning blocked')
    # Bound setup memory to one prepared host model plus mmap checkpoint. No
    # five-copy residency assumption for the 24B/27B/32B models.
    all_results=[]
    for dtype in (a.output_dtypes or protocol['output_dtypes']):
        emit(dict(kind='tuning_started',output_dtype=dtype))
        profiles=tune(cp,train,dtype,a.output/('tuning-'+dtype),emit)
        frozen=a.output/f'profiles-{dtype}.json';frozen.write_text(json.dumps(profiles,sort_keys=True,indent=2)+'\n');profile_hash=hashlib.sha256(frozen.read_bytes()).hexdigest()
        emit(dict(kind='profiles_frozen',output_dtype=dtype,sha256=profile_hash))
        pool=ModelPool(cp,*train.shape,profiles,dtype);baseline_folder=a.private/('native-logits-'+dtype)
        base,setup=pool.activate('native_default');baseline=base.forward(test,score=True,capture=True,save_logits=baseline_folder)
        if not baseline['quality']['finite']:raise RuntimeError('Nonfinite held-out Native outputs')
        qualities={'native_default':dict(**baseline['quality'],delta_nll=0.,mean_kl=0.,top1_agreement=1.,logit_relative_l2=0.,logit_max_abs=0.,logit_rmse=0.,logit_normalized_max=0.)};hidden_errors={}
        for name in protocol['algorithms'][1:]:
            runner,setup=pool.activate(name);result=runner.forward(test,score=True,capture=True,reference_logits=baseline_folder)
            qualities[name]=result['quality'];hidden_errors[name]=[dict(layer=i,**error_metrics(actual,ref)) for i,(actual,ref) in enumerate(zip(result['hidden'],baseline['hidden']))]
            emit(dict(kind='heldout_quality',algorithm=name,output_dtype=dtype,quality=qualities[name],hidden_errors=hidden_errors[name]))
            del result
        del baseline;gc.collect()
        samples={name:[] for name in protocol['algorithms']};order=[];rng=random.Random(protocol['timing']['order_seed'])
        for round_index in range(protocol['timing']['confirmation_rounds']):
            names=list(samples);rng.shuffle(names);order.append(names)
            for name in names:
                runner,setup=pool.activate(name)
                # The same three full warmups after every host-policy switch;
                # compilation/packing never enters measured inference samples.
                for _ in range(protocol['timing']['warmups']):runner.forward(train)
                timing=runner.forward(train)['timing'];assert timing['eligible_for_inference_timing']
                samples[name].append(timing)
                emit(dict(kind='confirmation',output_dtype=dtype,algorithm=name,round=round_index,elapsed_ms=timing['elapsed_ms'],resident_layer_sum_ms=timing['resident_layer_sum_ms'],host_preparation_seconds=setup))
        for name in protocol['algorithms']:
            quality=qualities[name];g=protocol['gates'];passed=quality['finite'] and abs(quality['delta_nll'])<=g['heldout_absolute_nll_delta_max'] and quality['mean_kl']<=g['heldout_mean_kl_max'] and quality['top1_agreement']>=g['heldout_top1_agreement_min']
            key=dict(model=model['id'],revision=model['revision'],batch=train.shape[0],sequence=train.shape[1],algorithm=name,output_dtype=dtype,architecture='v6e')
            speed=[];rngboot=np.random.default_rng(20261003)
            native=np.array([s['elapsed_ms'] for s in samples['native_default']]);current=np.array([s['elapsed_ms'] for s in samples[name]])
            for _ in range(protocol['timing']['bootstrap_samples']):
                indices=rngboot.integers(0,len(native),len(native));speed.append(float(np.median(native[indices])/np.median(current[indices])))
            measurement=dict(samples=samples[name],elapsed_median_ms=float(np.median(current)),
                resident_layer_median_ms=statistics.median(t['resident_layer_sum_ms'] for t in samples[name]),
                speedup_vs_default=float(np.median(native)/np.median(current)),
                speedup_bootstrap_95=list(map(float,np.quantile(speed,[.025,.975]))),
                quality=quality,hidden_errors=hidden_errors.get(name,[]),
                quality_scope='next-token positions in the first fixed held-out WikiText2 test window; compared to qualified Native BF16 model',
                timing_scope='complete streamed prompt forward including recurring H2D and all logits; serialized transfers; prepared host layouts and compilation excluded',
                confirmation_order=order)
            row=dict(entry_id=identity(key),**key,status='completed',quality_gate_passed=bool(passed),
                profile_sha256=profile_hash,profile=profiles[name],measurement=measurement)
            all_results.append(row);emit(dict(kind='comparison_complete',entry_id=row['entry_id'],algorithm=name,output_dtype=dtype,quality_gate_passed=bool(passed),median_ms=row['measurement']['elapsed_median_ms']))
        pool.close();jax.clear_caches()
        result_path=a.output/'completed_entries.jsonl';temporary=result_path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in all_results));temporary.replace(result_path)
        # Durable five-arm group: never mix partial timing rounds across chips.
        package=a.output/('checkpoint-'+str(len(all_results))+'.tar.gz')
        with tarfile.open(package,'w:gz') as archive:
            for path in sorted(a.output.iterdir()):
                if not path.name.endswith('.tar.gz') and path.name!='checkpoint-ready.json':archive.add(path,arcname='main-prefill/'+path.name)
        ready=a.output/'checkpoint-ready.json';temp=ready.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(path=str(package),sha256=hashlib.sha256(package.read_bytes()).hexdigest(),completed=len(all_results))))
        temp.replace(ready)
    summary=dict(status='completed',model=model,batch=train.shape[0],sequence=train.shape[1],main_entries_completed=len(all_results),quality_gates_passed=sum(r['quality_gate_passed'] for r in all_results),scope='one matched complete-model workload; downstream accuracy and KV decode remain separate pending stages')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');emit(dict(kind='case_complete',**summary))

if __name__=='__main__':main()
