"""Cache equivalence and session recovery. No TPU timing claims from CPU fixtures."""
import argparse,contextlib,hashlib,io,json,os,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True)
    import jax,numpy as np,torch,transformers
    import jax.numpy as jnp
    from strassen_mm.llm_stream_v001 import checkpoint,StreamedModel,pack_host
    from strassen_mm.llm_weight_cache_v001 import LayoutCache,CachedCheckpoint,cache_identity,packed_layer,digest
    from strassen_mm.llm_host_pool_v002 import ModelPool
    from strassen_mm.model_fused_v005 import build_layer,native_policy,SITES
    from strassen_mm.tuner_joint_v001 import arm
    from strassen_mm.llm_resident_v001 import measure
    from llm_queue_state_v002 import read,save,plan,merge,ledger,remaining
    from run_llm_queue_v005 import reconcile,finish
    from run_llm_session_controller_v001 import merge_downloaded
    root=Path(__file__).resolve().parents[1];checks=[]
    def check(name,condition,**extra):
        if not condition:raise AssertionError(name)
        row=dict(name=name,status='passed',**extra);checks.append(row);print(json.dumps(row),flush=True)
    torch.set_num_threads(1);torch.manual_seed(1234)
    for family,Config,Model in [('qwen3',transformers.Qwen3Config,transformers.Qwen3ForCausalLM),('mistral',transformers.MistralConfig,transformers.MistralForCausalLM),('gemma3_text',transformers.Gemma3TextConfig,transformers.Gemma3ForCausalLM)]:
        with tempfile.TemporaryDirectory() as temp:
            t=Path(temp);cfg=Config(vocab_size=97,hidden_size=96,intermediate_size=192,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=1,head_dim=32,sliding_window=16 if family=='gemma3_text' else None,query_pre_attn_scalar=32)
            cfg._attn_implementation='eager';official=Model(cfg).to(torch.bfloat16).eval();official.save_pretrained(t/'weights');del official
            config=read(t/'weights/config.json');files=[dict(path=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in (t/'weights').iterdir() if f.suffix=='.safetensors' or f.name=='config.json']
            save(t/'manifest.json',dict(config=config,revision='0'*40,cache_dir=str(t/'weights'),files=files));raw=checkpoint(t/'manifest.json')
            cache=LayoutCache(t/'cache',cache_identity(raw));cp=CachedCheckpoint(raw,cache)
            with patch.object(raw,'layer',wraps=raw.layer) as calls:
                first=cp.layer(0);second=cp.layer(0)
                check(family+' canonical layer built once',calls.call_count==1)
                check(family+' canonical read-only shared mappings',all(first[k] is second[k] and not first[k].flags.writeable for k in first))
            for depth in (0,1,2):
                policy={site:dict(arm((256,512,512),depth,dtype='bfloat16'),architecture='v6e') for site in SITES}
                if depth==0:
                    for value in policy.values():value['implementation']='cubic'
                fn=build_layer(cp.config,16,policy,batch_size=1,interpret=True)
                expected=pack_host(raw.layer(0),fn.metadata);actual=packed_layer(cp,0,fn.metadata)
                check(family+f' depth{depth} exact packed BF16 bits',all(np.array_equal(expected[k].view(np.uint8),actual[k].view(np.uint8)) for k in expected))
                again=packed_layer(cp,0,fn.metadata);check(family+f' depth{depth} shared packed buffers',all(actual[k] is again[k] for k in actual))
            profiles={name:dict(policy=native_policy(),compiler_options={}) for name in ('native_default','native_tuned','cubic_tuned','s1','s2')}
            ids=np.random.default_rng(3).integers(3,97,(1,16),dtype=np.int32)
            for dtype in ('bfloat16','float32'):
                profiles={name:dict(policy=native_policy(dtype),compiler_options={}) for name in profiles}
                original=StreamedModel(raw,1,16,output_dtype=dtype);old=original.forward(ids,score=True,capture=True,save_logits=t/('old-'+dtype));del original
                pool=ModelPool(cp,1,16,profiles,dtype);model,_=pool.activate('native_default');new=model.forward(ids,score=True,capture=True,save_logits=t/('new-'+dtype))
                check(family+' '+dtype+' full hidden/logit/quality bitwise equivalence',old['quality']==new['quality'] and all(np.array_equal(x,y) for x,y in zip(old['hidden'],new['hidden'])) and all(digest(t/('old-'+dtype)/f)==digest(t/('new-'+dtype)/f) for f in old['logit_files']))
                pool.activate('s1');model,seconds=pool.activate('native_default');again=model.forward(ids,score=True)
                check(family+' '+dtype+' reactivation has zero builds',pool.last_preparation['cache_delta']['misses']==0 and again['quality']==old['quality'])
                check(family+' '+dtype+' scoring excluded from inference',not new['timing']['eligible_for_inference_timing'] and model.forward(ids)['timing']['eligible_for_inference_timing'])
                pool.close()
            profiles={name:dict(policy=native_policy('bfloat16'),compiler_options={}) for name in profiles}
            events=[];resident=measure(cp,ids,profiles,'bfloat16',events.append,warmups=1,repeats=2)
            check(family+' resident pass advances all layers/five arms',len(events)==30 and all(len(r['per_layer'])==6 for r in resident.values()))
            check(family+' resident scope explicitly excludes full-forward latency',all('not full-forward latency' in r['scope'] and all(len(x['samples_ms'])==2 and x['finite'] for x in r['per_layer']) for r in resident.values()))
            check(family+' cache contains no bypass',cache.stats['bypasses']==0)
            # Reopened entries validate their checksums before reuse.
            reopened=LayoutCache(t/'cache',cache_identity(raw));check(family+' cache persists across workloads',np.array_equal(CachedCheckpoint(raw,reopened).layer(0)['q'].view(np.uint8),raw.layer(0)['q'].view(np.uint8)))
            # Execute the real measurement main on CPU fixtures with only
            # TPU discovery and candidate search substituted. Both precision
            # checkpoints, the real-weight pilot hook, resident pass, all five
            # confirmation arms and the output schema execute end-to-end.
            import run_llm_case_v003 as runner
            import strassen_mm.tuner_llm_v001 as tuner
            import types
            protocol_fixture=read(root/'configs/llm_v6e_v002.json');protocol_fixture['timing'].update(confirmation_rounds=2,warmups=1,bootstrap_samples=20)
            save(t/'protocol.json',protocol_fixture);private=t/'private';private.mkdir();shutil.copyfile(t/'manifest.json',private/'checkpoint-manifest.json')
            inputs=t/'inputs';(inputs/'official').mkdir(parents=True);np.save(inputs/'train-tokens.npy',ids);np.save(inputs/'test-tokens.npy',ids);np.save(inputs/'oracle-tokens.npy',ids)
            save(inputs/'summary.json',dict(model=dict(id='fixture-'+family,revision='0'*40),oracle=dict(scope='generated fixture')))
            for index,output in enumerate(old['hidden']):
                np.save(inputs/'official'/f'layer-{index:03d}-input.npy',np.asarray(raw.embeddings(ids) if index==0 else old['hidden'][index-1],dtype=np.float32))
                np.save(inputs/'official'/f'layer-{index:03d}-output.npy',output)
            def fake_tune(cp,ids,dtype,output,emit):
                output.mkdir();return {name:dict(policy=native_policy(dtype),compiler_options={}) for name in profiles}
            actual_devices=jax.devices;device_calls=[0]
            def discovery(*args,**kwargs):
                device_calls[0]+=1
                return [types.SimpleNamespace(device_kind='TPU v6 fixture')] if device_calls[0]==1 else actual_devices(*args,**kwargs)
            argv=['case','--private',str(private),'--inputs',str(inputs),'--output',str(t/'measured'),'--protocol',str(t/'protocol.json'),'--pilot']
            with patch.object(sys,'argv',argv),patch.object(jax,'devices',side_effect=discovery),patch.object(tuner,'tune',side_effect=fake_tune),contextlib.redirect_stdout(io.StringIO()):runner.main()
            result=[json.loads(x) for x in (t/'measured/completed_entries.jsonl').read_text().splitlines()]
            check(family+' real measurement main executes both stores/all five arms',len(result)==10 and all(len(r['measurement']['samples'])==2 and r['quality_gate_passed'] for r in result))
            check(family+' pilot and independent resident pass saved',all(read(t/'measured'/('cache-pilot-'+dtype+'.json'))['bitwise_full_model_equal'] for dtype in ('bfloat16','float32')) and all(r['measurement']['resident_pass']['aggregate_ms']>0 for r in result))
            check(family+' second precision checkpoint contains ten unique results',read(t/'measured/checkpoint-ready.json')['completed']==10 and len({r['entry_id'] for r in result})==10)
            one=next(cache.root.glob('*.bin'))
            with one.open('r+b') as f:byte=f.read(1);f.seek(0);f.write(bytes([byte[0]^1]))
            descriptor=read(one.with_suffix('.json'))['descriptor'];corrupt=LayoutCache(t/'cache',cache_identity(raw))
            try:corrupt.get(descriptor,lambda:None)
            except ValueError:check(family+' corrupt cache rejected',True)
            else:check('corrupt cache rejected',False)
            jax.clear_caches()
    with tempfile.TemporaryDirectory() as temp:
        cache=LayoutCache(Path(temp),{'test':'bounded'},max_bytes=32)
        x=cache.get({'kind':'packed','id':1},lambda:np.arange(8,dtype=np.float32))
        y=cache.get({'kind':'packed','id':2},lambda:np.arange(8,dtype=np.float32))
        check('disk bound preserves live arrays and safely bypasses',cache.used==32 and cache.stats['bypasses']==1 and np.array_equal(x,y))
        del x,y
        z=cache.get({'kind':'packed','id':3},lambda:np.arange(8,dtype=np.float32))
        check('unreferenced derived layouts evicted within bound',cache.used==32 and cache.stats['evictions']==1)
    protocol=read(root/'configs/llm_v6e_v002.json');jobs=plan(protocol);selected=[j for j in jobs if j['model']=='qwen3_8b'][:3]
    def rows(job,dtype):return [dict(entry_id=ident,**{k:job[k] for k in ('model','revision','batch','sequence','architecture')},algorithm=alg,output_dtype=dtype,status='completed',measurement={'fixture':True}) for alg,ident in zip(protocol['algorithms'],job['groups'][dtype])]
    with tempfile.TemporaryDirectory() as temp:
        t=Path(temp);dest=t/'ledger.jsonl';j0,j1,j2=selected
        allrows=rows(j0,'bfloat16')+rows(j0,'float32');merge(dest,allrows,j0,'session')
        merge(dest,rows(j1,'bfloat16'),j1,'session')
        for job in selected:job.update(status='reserved',attempts=[dict(run='session',status='reserved',source_commit='new')])
        state=dict(jobs=selected,status='running',active_session=dict(run='session',model='qwen3_8b',job_ids=[j['job_id'] for j in selected],source_commit='new'))
        reconcile(state,ledger(dest),{'current_job':j1})
        check('session progress preserves completed groups and current workload',j0['status']=='completed' and j1['status']=='running' and j2['status']=='reserved')
        receipt=dict(status='controller_failed',release_exit_code=0,remote_summary=dict(current_job=j1,cases=[dict(job_id=j0['job_id'],status='completed')]))
        finish(state,receipt,ledger(dest))
        check('runtime loss resumes only missing precision',remaining(j1,ledger(dest))==['float32'] and j0['status']=='completed' and j1['status']=='queued')
        check('unstarted later workload does not consume failure budget',j2['attempts'][-1]['status']=='not_started')
        state['active_session']=dict(run='again',model='qwen3_8b',job_ids=[j1['job_id']],source_commit='new')
        finish(state,dict(status='failed',release_exit_code=1),ledger(dest));check('release failure blocks another allocation',state['status']=='blocked' and state.get('active_session'))
        finish(state,dict(status='failed',release_exit_code=0,failure_class='shared_setup'),ledger(dest));check('shared setup failure stops entire queue',state['status']=='blocked_shared_setup' and not state.get('active_session'))
        # Actual controller promotion for checkpoint, case and final layouts.
        fake=t/'extracted';(fake/'main-prefill').mkdir(parents=True);(fake/'main-prefill/completed_entries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
        merge_downloaded(fake,dict(job_id=j0['job_id']),{j0['job_id']:j0},'session',dest)
        check('duplicate checkpoint promotion is idempotent',len(ledger(dest))==15)
        target=fake/'session/artifacts/cases'/j0['job_id']/'main-prefill';target.mkdir(parents=True);shutil.copyfile(fake/'main-prefill/completed_entries.jsonl',target/'completed_entries.jsonl')
        merge_downloaded(fake,{}, {j0['job_id']:j0},'session',dest,final=True)
        check('final nested session archive promotes without overwrites',len(ledger(dest))==15)
    # Exercise actual bundled worker, both preparation invocations and launch.
    from llm_bundle_v002 import validate,WORKER
    import importlib.util,types
    sys.path.insert(0,str(root/'runtime'));import parent_replication_remote_v001 as parent
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp).resolve();run=base/'runs/fixture';run.mkdir(parents=True);source=run/'source'
        for folder in ('tools','runtime','configs','src'):shutil.copytree(root/folder,source/folder)
        casejobs=[dict(j,output_dtypes=['bfloat16','float32']) for j in plan(protocol) if j['model']=='gemma3_12b'][:2]
        job=dict(casejobs[0],session_jobs=casejobs);save(source/'job.json',job);save(run/'launch.json',{'fixture':True})
        check('session archive CLIs and pinned jobs validated',validate(source,job)['status']=='passed')
        spec=importlib.util.spec_from_file_location('session_worker',source/WORKER);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.BASE=base
        (base/'.runtime_private/fixture/model').mkdir(parents=True);calls=[]
        def execute(command,log,env,timeout,cwd):
            calls.append(command);log.write_text('external processes replaced\n');return 0
        original=Path.read_text
        def read_text(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return 'MemAvailable: 999999999 kB\nMemTotal: 999999999 kB\n'
            return original(path,*args,**kwargs)
        with patch.object(parent,'execute',side_effect=execute),patch.object(worker.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**15)),patch.object(Path,'read_text',read_text):worker.worker(run)
        check('one setup serves two jobs and precreated Gemma private directory',sum('-m' in c and 'venv' in c for c in calls)==1 and read(run/'status.json')['status']=='completed')
        mains=[c for c in calls if any('tools/run_llm_case_v003.py' in x for x in c)]
        check('all session jobs launch new cached runner; only first gets pilot',len(mains)==2 and '--pilot' in mains[0] and '--pilot' not in mains[1])
        check('case archives precede final session archive',len(list((run/'artifacts/cases').glob('*/case-ready.json')))==2 and (run/'archive-ready.json').exists())
        package=base/'source.tar'
        with tarfile.open(package,'w') as tar:
            for item in source.iterdir():tar.add(item,arcname=item.name)
        argv=['worker','--archive',str(package),'--sha256',digest(package),'--run-id','launch-test']
        with patch.object(sys,'argv',argv),patch.object(worker.subprocess,'Popen',return_value=types.SimpleNamespace(pid=999)) as spawn:worker.main()
        check('real detached launch targets the session worker',spawn.call_args.args[0][1].endswith('/runtime/llm_model_session_v001.py'))
    summary=dict(status='passed',checks=len(checks),scope='CPU fixture equivalence and actual session orchestration with external services replaced; no TPU performance claim')
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
