"""CPU FP32 control of the existing formulation; never a BF16 gate override."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official


def load_model(source):
    global probe,model
    sys.path.insert(0,str(Path(source)/'tools'))
    import gemma4_numerics_probe_v001 as probe
    model=probe.model
    model.setup()
    return model


@contextmanager
def fp32_boundaries():
    """Lift only BF16 storage boundaries for an explicitly separate control."""
    rounded,bf32=model.native.rounded,model.native.bf32
    model.native.rounded=lambda x:x.astype(model.jnp.float32)
    model.native.bf32=lambda x:x.astype(model.jnp.float32)
    try:yield
    finally:model.native.rounded,model.native.bf32=rounded,bf32


def compare(raw,index,hidden):
    cfg=Gemma4TextConfig(**model.config);cfg._attn_implementation='eager'
    block=probe.official_layer(raw,index,cfg).float()
    x=torch.as_tensor(np.asarray(hidden,dtype=np.float32));batch,sequence,width=x.shape
    window=model.geometry(model.config,index)['sliding_window']
    with torch.no_grad():
        position=official.Gemma4TextRotaryEmbedding(cfg)(x,torch.arange(sequence)[None],cfg.layer_types[index])
        mask=torch.full((sequence,sequence),torch.finfo(torch.float32).min).triu(1)
        if window is not None:
            ids=torch.arange(sequence)
            mask.masked_fill_(ids[None,:]<=ids[:,None]-window,torch.finfo(torch.float32).min)
        expected=block(x,position_embeddings=position,attention_mask=mask[None,None])
    with model.jax.default_device(model.jax.devices('cpu')[0]),fp32_boundaries():
        p={k:model.jnp.asarray(v,model.jnp.float32) for k,v in raw.items()}
        p['cos'],p['sin']=model.native.rotary_tables(model.config,index,model.jnp.arange(sequence)[None])
        fn=model.jax.jit(model.build_layer('native',model.default_profile(batch,sequence),index,batch,sequence))
        candidate=fn(model.jnp.asarray(hidden,model.jnp.float32).reshape(-1,width),p)
        metric=probe.metrics(expected.reshape(-1,width),candidate)
    return dict(layer=index,attention_type=cfg.layer_types[index],**metric)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('This control must run only on CPU')
    args.output.mkdir(parents=True,exist_ok=False);load_model(args.reference_source)
    if hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()!='6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935':raise RuntimeError('Official source mismatch')
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    def emit(row):
        line=json.dumps(row,allow_nan=False)
        with (args.output/'events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    checkpoint=model.Checkpoint(args.cache,emit);rows=[]
    for index in (0,5,53):
        path=args.captures/f'official-activations-layer-{index:02d}.npz'
        with np.load(path,allow_pickle=False) as captured:
            hidden=torch.from_numpy(captured['input'].copy()).view(torch.bfloat16).float().numpy()
        raw=model.raw_layer(checkpoint,index)
        row=compare(raw,index,hidden);row['kind']='fp32_semantics'
        row['capture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row);emit(row)
        del raw,hidden;model.jax.clear_caches();gc.collect()
    summary=dict(scope='Independent CPU FP32 formulation control on real weights and official activations; not BF16 qualification or performance',
        layers=rows,all_pass=None,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_gates_unchanged=True,diagnosis_complete=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
