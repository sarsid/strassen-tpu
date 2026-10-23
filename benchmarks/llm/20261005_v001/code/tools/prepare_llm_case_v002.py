"""Acquire pinned weights/tokens and an independent actual-checkpoint oracle.

Private cache contains checkpoint bytes; artifacts contain hashes and small
semantic-reference activations only. Never archive the private checkpoint.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
CORPUS_REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):value.update(chunk)
    return value.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--sequence',type=int,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);a.private.mkdir(parents=True,exist_ok=True)
    def emit(**row):print(json.dumps(row),flush=True)
    from huggingface_hub import snapshot_download
    import numpy as np,torch,transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert transformers.__version__=='4.56.2' and torch.__version__.split('+')[0]=='2.8.0'
    protocol=json.loads(a.protocol.read_text());model=next(m for m in protocol['models'] if m['id']==a.model)
    emit(kind='checkpoint_download',model=model['repo_id'],revision=model['revision'])
    token_path=a.private/'hf_token';token=token_path.read_text().strip() if token_path.exists() else False
    cache=Path(snapshot_download(model['repo_id'],revision=model['revision'],cache_dir=str(a.private/'hub'),token=token,allow_patterns=['*.json','*.safetensors','*.model','*.txt'],max_workers=4))
    if digest(cache/'config.json')!=model['config_sha256']:raise ValueError('Pinned configuration hash mismatch')
    files=[]
    for path in sorted(cache.iterdir()):
        if path.name=='config.json' or path.suffix=='.safetensors':
            files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=digest(path)))
            emit(kind='checkpoint_file_verified',path=path.name,bytes=path.stat().st_size)
    # Resolve symlinks into a local directory contract without moving content.
    # Snapshot paths point to blobs outside the snapshot: the verified loader
    # rejects escaping paths, so create private hard links to the exact bytes.
    linked=a.private/'checkpoint';linked.mkdir(exist_ok=True)
    for row in files:
        target=linked/row['path']
        if not target.exists():os.link((cache/row['path']).resolve(),target)
    manifest=dict(model_id=model['repo_id'],revision=model['revision'],config=json.loads((cache/'config.json').read_text()),cache_dir=str(linked),files=files)
    (a.private/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'checkpoint-provenance.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='cache_dir'},indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(cache,local_files_only=True,trust_remote_code=False)
    count=a.batch*a.sequence
    for split in ('train','test'):
        dataset=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=CORPUS_REVISION,split=split,cache_dir=str(a.private/'datasets'))
        text='\n\n'.join(row['text'] for row in dataset if row['text'].strip())
        tokens=np.asarray(tokenizer(text,add_special_tokens=False)['input_ids'],np.int32)
        if len(tokens)<count:raise ValueError('Insufficient corpus tokens; no repetition allowed')
        selected=tokens[:count].reshape(a.batch,a.sequence);np.save(a.output/(split+'-tokens.npy'),selected)
        record=dict(dataset='Salesforce/wikitext',revision=CORPUS_REVISION,subset='wikitext-2-raw-v1',split=split,selection='first consecutive tokens after joining nonempty rows with two newlines; no BOS or chat template',text_sha256=hashlib.sha256(text.encode()).hexdigest(),token_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),shape=list(selected.shape),tokenizer_revision=model['revision'])
        (a.output/(split+'-tokens.json')).write_text(json.dumps(record,indent=2)+'\n')
    ids=np.load(a.output/'train-tokens.npy')[:1,:64].copy();np.save(a.output/'oracle-tokens.npy',ids)
    torch.set_num_threads(min(8,os.cpu_count() or 1));emit(kind='official_oracle_loading',tokens=int(ids.size))
    if manifest['config']['model_type']=='gemma3':
        from load_gemma_text_v001 import load_text
        official=load_text(cache)
    else:
        official=AutoModelForCausalLM.from_pretrained(cache,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='eager').eval()
    oracle=a.output/'official';oracle.mkdir();handles=[]
    for i,layer in enumerate(official.model.layers):
        def pre(mod,args,kwargs,i=i):
            value=args[0] if args else kwargs['hidden_states'];np.save(oracle/f'layer-{i:03d}-input.npy',value.detach().float().numpy())
        def post(mod,args,value,i=i):
            y=value[0] if isinstance(value,tuple) else value;np.save(oracle/f'layer-{i:03d}-output.npy',y.detach().float().numpy());emit(kind='official_oracle_layer',layer=i)
        handles += [layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post)]
    with torch.inference_mode():logits=official(torch.tensor(ids,dtype=torch.long),use_cache=False).logits.float().numpy()
    for handle in handles:handle.remove()
    np.save(oracle/'logits.npy',logits)
    receipt=dict(status='completed',model=model,actual_checkpoint=True,oracle=dict(torch=torch.__version__,transformers=transformers.__version__,tokens=int(ids.size),batch=1,layers=len(official.model.layers)),batch=a.batch,sequence=a.sequence,artifacts={str(f.relative_to(a.output)):dict(sha256=digest(f),bytes=f.stat().st_size) for f in sorted(a.output.rglob('*')) if f.is_file()})
    (a.output/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');emit(kind='inputs_ready',model=a.model)

if __name__=='__main__':main()
