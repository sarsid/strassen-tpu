"""Acquire exact larger checkpoints and independent official CPU BF16 references.

Official checkpoint bytes stay private. No JAX imports or altered checkpoint
configuration. Gemma's multimodal wrapper receives text only, without images.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import prepare_model_application_v001 as prep

MODELS = [
 ('qwen','Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),
 ('mistral','mistralai/Mistral-7B-v0.3','caa1feb0e54d415e2df31207e5f4e273e33509b1'),
 ('gemma','google/gemma-3-12b-pt','295efb63d01a7017928f273a94ebb86105c9526f'),
]

def tokenize(model_path,corpus_path,out):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    model,directory=prep.verify_manifest(model_path)
    corpus,corpus_dir=prep.verify_manifest(corpus_path)
    text='\n'.join(t for row in corpus['files'] for t in pq.read_table(corpus_dir/row['path'],columns=['text']).column('text').to_pylist())
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_fast=True)
    tokens=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
    if len(tokens)<65536:raise ValueError('Pinned corpus has insufficient tokens')
    ids=np.asarray(tokens[:65536],dtype='<i4').reshape(32,2048)
    np.save(out/'token_ids.npy',ids,allow_pickle=False)
    report=dict(model_id=model['model_id'],model_revision=model['revision'],model_manifest_sha256=prep.digest(model_path),
        corpus_manifest_sha256=prep.digest(corpus_path),dataset_id=corpus['dataset_id'],dataset_revision=corpus['revision'],
        text_sha256=prep.hashlib.sha256(text.encode()).hexdigest(),token_array_path='token_ids.npy',
        token_array_sha256=prep.digest(out/'token_ids.npy'),shape=list(ids.shape),add_special_tokens=False,
        selection='First65536 contiguous tokenizer IDs,32 disjoint2048-token windows; quality uses first16windows.',
        unique_token_ids=int(np.unique(ids).size),full_corpus_token_count=len(tokens))
    prep.save(out/'tokens_manifest.json',report)

def reference(model_path,tokens_path,out):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,Gemma3ForConditionalGeneration
    info,directory=prep.verify_manifest(model_path)
    ids=np.load(tokens_path,allow_pickle=False).reshape(-1)[:64].astype(np.int64).reshape(1,64)
    torch.set_num_threads(1);start=time.monotonic()
    cls=Gemma3ForConditionalGeneration if info['config']['model_type']=='gemma3' else AutoModelForCausalLM
    model=cls.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True)
    model.eval()
    with torch.inference_mode():result=model(torch.from_numpy(ids),use_cache=False,output_hidden_states=True)
    arrays=dict(input_ids=ids.astype('<i4'),logits=result.logits.float().cpu().numpy(),final_hidden=result.hidden_states[-1].float().cpu().numpy())
    files={}
    for name,array in arrays.items():
        np.save(out/(name+'.npy'),array,allow_pickle=False)
        files[name]=dict(path=name+'.npy',sha256=prep.digest(out/(name+'.npy')),shape=list(array.shape))
    prep.save(out/'reference_manifest.json',dict(model_id=info['model_id'],revision=info['revision'],
        model_manifest_sha256=prep.digest(model_path),tokens_sha256=prep.digest(tokens_path),files=files,
        implementation='Official Transformers4.56.2 CPU BF16 eager model, text-only, eval, no cache',
        model_class=type(model).__name__,elapsed_seconds=time.monotonic()-start,
        versions={n:md.version(n) for n in ('torch','transformers','safetensors','numpy')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-key',choices=[x[0] for x in MODELS],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    required={'torch':'2.8.0','transformers':'4.56.2','tokenizers':'0.22.0','huggingface-hub':'0.34.4',
        'safetensors':'0.6.2','numpy':'2.2.6','pyarrow':'21.0.0','sentencepiece':'0.2.1','protobuf':'6.32.1'}
    versions={n:md.version(n) for n in required};prep.save(out/'versions.json',versions)
    if any(versions[n].split('+')[0]!=v for n,v in required.items()):raise ValueError('Pinned oracle mismatch')
    key,model,revision=next(x for x in MODELS if x[0]==a.model_key)
    try:
        acquire=out/'model_acquisition';acquire.mkdir()
        prep.model_action(SimpleNamespace(model_id=model,revision=revision,action='download',use_existing_token=True,
            cache_root=a.cache_root,max_model_bytes=30_000_000_000),acquire)
        corpus=out/'corpus_acquisition';corpus.mkdir()
        prep.corpus_action(SimpleNamespace(cache_root=a.cache_root),corpus)
        tok=out/'tokens';tok.mkdir();tokenize(acquire/'model_manifest.json',corpus/'corpus_manifest.json',tok)
        ref=out/'reference';ref.mkdir();reference(acquire/'model_manifest.json',tok/'token_ids.npy',ref)
        row=dict(model_key=key,model_id=model,revision=revision,status='ready',
            model_manifest=str((acquire/'model_manifest.json').resolve()),
            tokens_manifest=str((tok/'tokens_manifest.json').resolve()),reference_manifest=str((ref/'reference_manifest.json').resolve()))
        row['input_sha256']={n:prep.digest(Path(row[n])) for n in ('model_manifest','tokens_manifest','reference_manifest')}
    except Exception as e:
        row=dict(model_key=key,model_id=model,revision=revision,status='input_preparation_failed',error_type=type(e).__name__)
        # HTTP traces can include signed credentials: only exception type is recorded.
    prep.save(out/'input_bundle.json',row)
    prep.save(out/'artifact_manifest.json',{'sha256':{str(x.relative_to(out)):prep.digest(x) for x in out.rglob('*') if x.is_file()}})
    print(json.dumps({k:row[k] for k in ('model_key','model_id','status')}),flush=True)
    return 0 if row['status']=='ready' else 1
if __name__=='__main__':raise SystemExit(main())
