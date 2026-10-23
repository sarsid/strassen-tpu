"""One model's admitted, restartable Native/S1 downstream subset evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import jax
import jax.numpy as jnp
from accuracy_engine_v004 import Engine,MODELS,ROOT
from accuracy_scoring_v001 import ARMS,packed,mmlu_batch,ppl_batch,gsm_batch
from accuracy_analysis_v003 import analyze


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Existing artifact differs: '+path.name)
        return
    temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_text(text);temp.rename(path)


def qualify(engine,pilot,emit):
    if pilot['scope']!='train_generation_timing_pilot' or pilot['model']!=engine.repo or pilot['revision']!=engine.revision:
        raise ValueError('Admission must use separate pinned train inputs')
    train=[dict(token_ids=x[:63]) for x in pilot['token_ids']]
    ids,pos,lens=packed(train,batch=8,width=64);engine.reset(8,128)
    engine.forward({a:ids for a in ARMS},{a:pos for a in ARMS},{a:lens for a in ARMS},
        last_indices={a:lens-1 for a in ARMS},label='admission_prefix')
    following=np.asarray([[x[63]] for x in pilot['token_ids']],np.int32)
    logits,_,_=engine.forward({a:following for a in ARMS},{a:np.full((8,1),63,np.int32) for a in ARMS},
        {a:np.full(8,64,np.int32) for a in ARMS},last_indices={a:np.zeros(8,np.int32) for a in ARMS},label='admission_cached')
    ids,pos,lens=packed([dict(token_ids=x[:64]) for x in pilot['token_ids']],batch=8,width=64)
    engine.reset(8,128)
    full,_,_=engine.forward({a:ids for a in ARMS},{a:pos for a in ARMS},{a:lens for a in ARMS},
        last_indices={a:lens-1 for a in ARMS},label='admission_full_prefix',retain_cache=False)
    if not all(np.all(np.isfinite(x)) for x in [*logits.values(),*full.values()]):raise ValueError('Nonfinite admission output')
    delta=logits['native'].astype(np.float64)-full['native'].astype(np.float64)
    relative=float(np.linalg.norm(delta)/max(np.linalg.norm(full['native'].astype(np.float64)),1e-30))
    result=dict(status='passed' if relative<=.005 else 'failed',native_cached_full_relative_l2=relative,
        native_cached_full_max_abs=float(np.abs(delta).max()),relative_l2_limit=.005,
        native_greedy_agreement=float(np.mean(np.argmax(logits['native'],axis=-1)==np.argmax(full['native'],axis=-1))),
        train_example_ids=pilot['example_ids'],train_input_sha256=pilot['inputs_sha256'],all_logits_finite=True,
        strassen_quality_is_not_an_admission_gate=True)
    emit(dict(kind='runtime_admission',**result))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'summary.json').exists():return
    if jax.default_backend()!='tpu' or len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Expected single v6e')
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    events=a.output/('events-'+attempt+'.jsonl');contracts=[]
    def emit(value):
        value={'utc':datetime.now(timezone.utc).isoformat(),'model':a.model,**value}
        with events.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        print(json.dumps(value,allow_nan=False),flush=True)
        if value['kind']=='engine_contract':contracts.append(value)
    started=time.monotonic()
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    raw=(a.inputs/'inputs.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['inputs_sha256'] or manifest['counts']!={'mmlu':114,'gsm8k':32,'ppl':32}:raise ValueError('Input manifest mismatch')
    rows=[json.loads(x) for x in raw.decode().splitlines()]
    for row in rows:
        if digest({k:v for k,v in row.items() if k not in ('gold','input_sha256')})!=row['input_sha256']:raise ValueError('Input content hash mismatch')
    protocol=ROOT/'protocols/LLM_ACCURACY_QUERY_BLOCK_REPAIR_v009.md'
    evidence=json.loads((ROOT/'plans/llm_accuracy_20261018_v001/admission_evidence_v002.json').read_text())
    if any(x['status']!='completed' or x['exit_code']!=0 for x in evidence['validations']):raise ValueError('Local admission failed')
    source_files=['tools/accuracy_study_v002.py','tools/accuracy_engine_v004.py','tools/accuracy_decoder_v001.py',
        'tools/accuracy_mm_v001.py','tools/accuracy_scoring_v001.py','tools/accuracy_analysis_v003.py',
        'tools/accuracy_query_block_preflight_v001.py','runtime/accuracy_query_block_recovery_v001.py']
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    engine=Engine(a.model,a.cache,emit)
    contract={k:v for k,v in contracts[-1].items() if k not in ('utc','kind')}
    contract.update(source_sha256=source,bundle_sha256=os.environ.get('STRASSEN_SOURCE_SHA256'),
        batch_size=8,max_new_tokens=1024,do_sample=False,local_admission=evidence,
        environment={k:os.environ.get(k) for k in ('XLA_FLAGS','LIBTPU_INIT_ARGS','JAX_PLATFORMS')},
        software={k:md.version(k) for k in ('jax','jaxlib','libtpu','numpy','transformers','tokenizers','scipy')})
    save(a.output/'execution-contract.json',contract);contract_hash=digest(contract)
    emit(dict(kind='preparing_model',test_evaluations_completed=0));engine.prepare()
    admission=qualify(engine,json.loads(a.pilot.read_text()),emit)
    save(a.output/('admission-'+attempt+'.json'),admission)
    if admission['status']!='passed':raise RuntimeError('Runtime admission failed; no test scoring')
    manifest=dict(manifest,admission_status='passed',execution_contract_sha256=contract_hash,
        original_subset_protocol_sha256=manifest['protocol_sha256'],protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    save(a.output/'manifest.json',manifest)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(engine.repo,revision=engine.revision,use_fast=True,token=False)
    gen=json.loads((ROOT/f'plans/llm_accuracy_20261018_v001/metadata/{a.model}.generation_config.json').read_text())
    eos=gen['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    checkpoints=a.output/'batches';checkpoints.mkdir(exist_ok=True)
    collected={arm:[] for arm in ARMS}
    for task in ('mmlu','ppl','gsm8k'):
        task_rows=[row for row in rows if row['task']==task]
        width=128*math.ceil(max(len(x['token_ids']) for x in task_rows)/128)
        capacity=128*math.ceil((max(len(x['token_ids']) for x in task_rows)+1024)/128)
        for first in range(0,len(task_rows),8):
            chunk=task_rows[first:first+8];path=checkpoints/f'{task}-{first//8:03d}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['execution_contract_sha256']!=contract_hash:raise ValueError('Cannot resume changed contract')
                if saved['example_ids']!=[r['example_id'] for r in chunk]:raise ValueError('Cannot resume changed example order')
                result=saved['results']
            else:
                emit(dict(kind='task_batch_started',task=task,batch=first//8+1,batches=math.ceil(len(task_rows)/8),completed_examples=first,total_examples=len(task_rows)))
                if task=='mmlu':result=mmlu_batch(engine,chunk,width=width)
                elif task=='ppl':result=ppl_batch(engine,chunk)
                else:result=gsm_batch(engine,chunk,tokenizer,eos,emit,width=width,capacity=capacity)
                save(path,dict(task=task,example_ids=[r['example_id'] for r in chunk],results=result,execution_contract_sha256=contract_hash))
            for arm in ARMS:
                for row,metrics in zip(chunk,result[arm],strict=True):
                    common={k:row.get(k) for k in ('task','example_id','subject','gold','input_sha256')}
                    collected[arm].append(dict(common,arm=arm,checkpoint_revision=engine.revision,
                        protocol_sha256=manifest['protocol_sha256'],execution_contract_sha256=contract_hash,**metrics))
            emit(dict(kind='task_batch_complete',task=task,batch=first//8+1,completed_examples=min(first+8,len(task_rows)),total_examples=len(task_rows)))
        emit(dict(kind='task_complete',task=task,examples=len(task_rows)))
    for arm in ARMS:
        path=a.output/(arm+'.jsonl');text=''.join(json.dumps(x,allow_nan=False)+'\n' for x in collected[arm])
        if path.exists() and path.read_text()!=text:raise ValueError('Existing paired rows differ')
        if not path.exists():
            temp=path.with_suffix('.tmp');temp.write_text(text);temp.rename(path)
    report=analyze(manifest,a.output/'native.jsonl',a.output/'strassen.jsonl')
    save(a.output/'summary.json',dict(report,wall_seconds=time.monotonic()-started,evaluation_scope=manifest['evaluation_scope']))
    metadata={p.name:json.loads(p.read_text()) for p in (a.cache/'canonical').glob('*.json')}
    save(a.output/'canonical-weight-manifest.json',metadata)
    emit(dict(kind='model_complete',task_evaluations_completed=3,summary='summary.json'))


if __name__=='__main__':main()
