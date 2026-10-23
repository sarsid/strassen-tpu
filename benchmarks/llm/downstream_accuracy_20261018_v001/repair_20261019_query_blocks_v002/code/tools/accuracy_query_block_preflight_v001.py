"""Training-only checks of the actual repaired two-arm accuracy engine."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import numpy as np
import jax
from accuracy_engine_v004 import Engine,MODELS
from accuracy_scoring_v001 import ARMS,packed
from accuracy_study_v002 import qualify


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    for name in ('pilot','output','cache'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if jax.default_backend()!='tpu' or len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Require one v6e TPU')
    def emit(value):
        row={'utc':datetime.now(timezone.utc).isoformat(),'model':a.model,**value}
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    pilot=json.loads(a.pilot.read_text())
    if pilot['scope']!='train_generation_timing_pilot' or pilot['model']!=MODELS[a.model][1] or pilot['revision']!=MODELS[a.model][2]:raise ValueError('Pinned training inputs required')
    engine=Engine(a.model,a.cache,emit);engine.prepare();checks=[]
    first=qualify(engine,pilot,emit);checks.append(dict(prefix_length=64,**first))
    if first['status']!='passed':raise RuntimeError('Original Native admission failed; no test scoring')
    for length in (129,1024):
        if any(len(x)<length for x in pilot['token_ids']):raise ValueError('Incomplete training prefix')
        width=128*math.ceil(length/128);capacity=128*math.ceil((length+16)/128)
        ids,pos,lens=packed([dict(token_ids=x[:length-1]) for x in pilot['token_ids']],batch=8,width=width)
        engine.reset(8,capacity)
        engine.forward({arm:ids for arm in ARMS},{arm:pos for arm in ARMS},{arm:lens for arm in ARMS},last_indices={arm:lens-1 for arm in ARMS},label=f'preflight_{length}_prefix')
        following=np.asarray([[x[length-1]] for x in pilot['token_ids']],np.int32)
        logits,_,_=engine.forward({arm:following for arm in ARMS},{arm:np.full((8,1),length-1,np.int32) for arm in ARMS},{arm:np.full(8,length,np.int32) for arm in ARMS},last_indices={arm:np.zeros(8,np.int32) for arm in ARMS},label=f'preflight_{length}_cached')
        ids,pos,lens=packed([dict(token_ids=x[:length]) for x in pilot['token_ids']],batch=8,width=width)
        engine.reset(8,capacity)
        full,_,_=engine.forward({arm:ids for arm in ARMS},{arm:pos for arm in ARMS},{arm:lens for arm in ARMS},last_indices={arm:lens-1 for arm in ARMS},label=f'preflight_{length}_full',retain_cache=False)
        metrics={}
        finite=all(np.all(np.isfinite(x)) for x in [*logits.values(),*full.values()])
        if not finite:raise RuntimeError('Nonfinite preflight logits')
        for arm in ARMS:
            reference=full[arm].astype(np.float64);delta=logits[arm].astype(np.float64)-reference
            metrics[arm]=dict(relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30)),max_abs=float(np.abs(delta).max()),greedy_agreement=float(np.mean(np.argmax(logits[arm],axis=-1)==np.argmax(full[arm],axis=-1))))
        check=dict(prefix_length=length,status='passed' if metrics['native']['relative_l2']<=.005 else 'failed',relative_l2_limit=.005,all_logits_finite=finite,cached_full_metrics=metrics,strassen_quality_is_not_an_admission_gate=True)
        checks.append(check);emit(dict(kind='extended_training_admission',**check))
        if check['status']!='passed':raise RuntimeError('Long Native admission failed; no test scoring')
    summary=dict(status='passed',scope='training-only actual two-arm engine validation; no downstream scores',model=engine.repo,revision=engine.revision,query_block=16,train_example_ids=pilot['example_ids'],train_input_sha256=pilot['inputs_sha256'],checks=checks,source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()for name in ('accuracy_engine_v004.py','accuracy_study_v002.py','accuracy_query_block_preflight_v001.py')})
    with (a.output/'summary.json').open('x')as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    emit(dict(kind='preflight_complete',status='passed'))

if __name__=='__main__':main()
