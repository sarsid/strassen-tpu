"""Offline recovery/ledger tests and independent Gemma oracle extraction checks."""
import argparse,contextlib,io,json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path
from unittest.mock import patch
from llm_queue_state_v001 import plan,remaining,merge,ledger,save,read,failure_action
import run_llm_queue_v002 as supervisor

class StopLoop(BaseException):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1];protocol=read(root/'configs/llm_v6e_v002.json');checks=[]
    def check(name,test):
        if not test:raise AssertionError(name)
        checks.append(name);print(json.dumps(dict(check=name,status='passed')),flush=True)
    jobs=plan(protocol);ids=[i for j in jobs for g in j['groups'].values() for i in g]
    check('700 unique identities across 70 workloads',len(jobs)==70 and len(ids)==700 and len(set(ids))==700)
    def rows(job,dtype):
        result=[]
        for algorithm,i in zip(protocol['algorithms'],job['groups'][dtype]):
            result.append(dict(entry_id=i,**{k:job[k] for k in ('model','revision','batch','sequence','architecture')},algorithm=algorithm,output_dtype=dtype,status='completed',measurement={'fixture':True}))
        return result
    with tempfile.TemporaryDirectory() as temporary:
        dest=Path(temporary)/'ledger.jsonl';job=jobs[0];group=rows(job,'bfloat16')
        merge(dest,group,job,'fixture-a');merge(dest,group,job,'fixture-a')
        check('idempotent group checkpoint and FP32-only resumption',len(ledger(dest))==5 and remaining(job,ledger(dest))==['float32'])
        for name,bad,evidence in [('partial group',group[:4],'fixture-a'),('conflicting run',group,'fixture-b'),('fake identity',[dict(group[0],entry_id='fake')],'fixture-a')]:
            try:merge(dest,bad,job,evidence)
            except ValueError:check('reject '+name,True)
            else:check('reject '+name,False)
        merge(dest,rows(job,'float32'),job,'fixture-b');check('complete workload after independent store resume',not remaining(job,ledger(dest)))
    check('allocation failure backs off globally',failure_action({'status':'allocation_failed'})=='allocation_wait')
    check('release failure prevents next allocation',failure_action({'release_exit_code':1})=='release_blocked')
    # Exercise the real supervisor's filesystem transition logic with process and
    # Colab calls replaced. No TPU APIs, production ledgers or network are used.
    with tempfile.TemporaryDirectory() as temporary:
        fake=Path(temporary);campaign=fake/'results';campaign.mkdir();queue=campaign/'queue_v001';queue.mkdir();(fake/'configs').mkdir()
        save(fake/'configs/llm_v6e_v002.json',protocol)
        with tarfile.open(queue/'source.tar','w') as tar:pass
        state=dict(status='running',jobs=plan(protocol),source_commit='fixture',allocation_failures=0,next_attempt_after=0)
        current=state['jobs'][0];current['status']='running';current['attempts']=[dict(run='case-fixture',pid=1234,status='running',recoveries=0)]
        run=campaign/'case-fixture';run.mkdir();save(run/'allocation.json',dict(session='case-fixture',endpoint='fixture-endpoint'))
        class Child:
            pid=2345
            def poll(self):return None
        def command(argv,**kw):
            output=json.dumps(dict(kind='allocations',assignments=[])) if argv[-1]=='list' else ''
            return subprocess.CompletedProcess(argv,0,stdout=output if kw.get('text') else b'',stderr=b'')
        def tick(alive=None):
            save(queue/'state.json',state)
            with patch.dict(os.environ,STRASSEN_PROJECT_ROOT=str(fake)),patch.object(sys,'argv',['queue','--campaign',str(campaign),'--controller-python',sys.executable]),patch.object(supervisor,'running',return_value=alive),patch.object(supervisor.subprocess,'run',side_effect=command),patch.object(supervisor.subprocess,'Popen',return_value=Child()) as spawn,patch.object(supervisor.time,'sleep',side_effect=StopLoop):
                try:supervisor.main()
                except StopLoop:pass
            state.clear();state.update(read(queue/'state.json'));return spawn
        spawn=tick(1234);check('adopt live controller without launching duplicate',spawn.call_count==0 and state['jobs'][0]['status']=='running')
        spawn=tick();check('dead controller reconnects to same run',spawn.call_count==1 and '--recover' in spawn.call_args.args[0] and state['jobs'][0]['attempts'][0]['recoveries']==1)
        save(run/'completion.json',dict(status='controller_failed',release_exit_code=0))
        tick();check('failed workload automatically requeues',state['jobs'][0]['status']=='queued')
        spawn=tick();check('next attempt launches unattended',spawn.call_count==1 and state['jobs'][0]['status']=='running')
        retry=campaign/state['jobs'][0]['attempts'][-1]['run'];save(retry/'completion.json',dict(status='allocation_failed'))
        tick();check('allocation rejection sets backoff without losing jobs',state['allocation_failures']==1 and state['next_attempt_after']>0 and state['jobs'][1]['status']=='queued')
        spawn=tick();check('backoff does not allocate',spawn.call_count==0)
    # Lost release response and provider expiry must not strand the queue.
    sys.path.insert(0,str(root/'runtime'))
    import release_allocation_v003 as cleanup
    from types import SimpleNamespace as N
    class Store:
        def __init__(self):self.saved=N(endpoint='owned')
        def get(self,name):return self.saved
        def remove(self,name):self.saved=None
    class Client:
        def __init__(self):self.assignments=[N(endpoint='owned',accelerator=N(value='V6E1'))];self.released=[]
        def list_assignments(self):return self.assignments
        def unassign(self,endpoint):self.released.append(endpoint);self.assignments=[]
    client=Client();store=Store()
    first=cleanup.release(client,store,'session','owned');second=cleanup.release(client,store,'session','owned')
    check('repeated release succeeds after verified absence',first['verified_absent'] and second['already_absent'] and client.released==['owned'])
    client=Client();store=Store();client.assignments=[]
    check('expired runtime cleaned up without unassigning another',cleanup.release(client,store,'session','owned')['already_absent'] and not client.released)
    client=Client();store=Store();store.saved=None
    try:cleanup.release(client,store,'session','owned')
    except RuntimeError:check('unowned runtime never released',not client.released)
    else:check('unowned runtime never released',False)
    import ast
    for name in ('tools/run_llm_case_controller_v004.py','tools/run_llm_case_v002.py','tools/prepare_llm_case_v002.py','runtime/llm_case_remote_v002.py','runtime/poll_llm_queue_v001.py','tools/watch_llm_queue_v002.py'):
        ast.parse((root/name).read_text())
    check('all queued execution entrypoints parse',True)
    import torch,transformers
    from safetensors.torch import save_file
    from load_gemma_text_v001 import load_text
    torch.set_num_threads(1);torch.manual_seed(731)
    cfg=transformers.Gemma3TextConfig(vocab_size=97,hidden_size=96,intermediate_size=192,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=1,head_dim=32,sliding_window=16,query_pre_attn_scalar=32,rope_scaling={'rope_type':'linear','factor':8.},final_logit_softcapping=3.)
    cfg._attn_implementation='eager';model=transformers.Gemma3ForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'norm' in name:value.copy_(torch.linspace(-.2,.3,value.numel()).reshape(value.shape))
    tokens=torch.randint(3,97,(2,40))
    with torch.inference_mode():reference=model(tokens,use_cache=False).logits
    for style in ('original','new'):
        with tempfile.TemporaryDirectory() as temporary:
            cache=Path(temporary)
            # Match production from_pretrained(dtype=BF16), which retains FP32
            # RoPE frequencies. model.to(BF16) also rounds nonpersistent buffers.
            reference_path=cache/'official-text';model.save_pretrained(reference_path)
            independent=transformers.Gemma3ForCausalLM.from_pretrained(reference_path,torch_dtype=torch.bfloat16,attn_implementation='eager').eval()
            with torch.inference_mode():reference=independent(tokens,use_cache=False).logits
            save(cache/'config.json',dict(model_type='gemma3',text_config=cfg.to_dict()))
            tensors={}
            for name,value in model.named_parameters():
                target='language_model.'+name if style=='original' else ('model.language_model.'+name[len('model.'):] if name.startswith('model.') else name)
                tensors[target]=value.detach().contiguous().clone()
            tensors['vision_tower.fake.weight']=torch.ones((3,3),dtype=torch.bfloat16)
            save_file(tensors,str(cache/'model.safetensors'));loaded=load_text(cache)
            with torch.inference_mode():actual=loaded(tokens,use_cache=False).logits
            check('Gemma '+style+' serialization exact text logits with local/global attention',torch.equal(reference,actual))
            missing=next(k for k in tensors if 'q_proj' in k);del tensors[missing];save_file(tensors,str(cache/'model.safetensors'))
            try:load_text(cache)
            except ValueError:check('Gemma '+style+' missing tensor rejected',True)
            else:check('Gemma missing tensor',False)
    save(a.output/'summary.json',dict(status='completed',passed=len(checks),failed=0,checks=checks,scope='offline orchestration and synthetic Gemma loader qualification; no TPU performance or real-model quality claim'))
    return 0

if __name__=='__main__':raise SystemExit(main())
