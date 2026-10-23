"""Reuse pinned checkpoint, token corpus and independent oracle within a model session."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from prepare_llm_case_v002 import digest,CORPUS_REVISION

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--sequence',type=int,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    model=next(m for m in json.loads(a.protocol.read_text())['models'] if m['id']==a.model)
    template=a.private/'input-template';receipt=template/'cache-receipt.json'
    if not receipt.exists():
        import prepare_llm_case_v002 as original
        original.main()
        # Freeze the full token arrays once so subsequent workload shapes need
        # no tokenizer, dataset loading or independent model re-initialization.
        from transformers import AutoTokenizer
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        token_path=a.private/'hf_token';token=token_path.read_text().strip() if token_path.exists() else False
        cache=Path(snapshot_download(model['repo_id'],revision=model['revision'],cache_dir=str(a.private/'hub'),token=token,local_files_only=True))
        tokenizer=AutoTokenizer.from_pretrained(cache,local_files_only=True,trust_remote_code=False)
        template.mkdir(exist_ok=True);shutil.copytree(a.output/'official',template/'official',dirs_exist_ok=True)
        for name in ('checkpoint-provenance.json','oracle-tokens.npy','summary.json'):shutil.copyfile(a.output/name,template/name)
        for split in ('train','test'):
            dataset=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=CORPUS_REVISION,split=split,cache_dir=str(a.private/'datasets'))
            text='\n\n'.join(row['text'] for row in dataset if row['text'].strip())
            tokens=np.asarray(tokenizer(text,add_special_tokens=False)['input_ids'],np.int32)
            if not np.array_equal(tokens[:a.batch*a.sequence].reshape(a.batch,a.sequence),np.load(a.output/(split+'-tokens.npy'))):raise ValueError('Token cache differs from original preparation')
            np.save(template/(split+'-all.npy'),tokens)
            shutil.copyfile(a.output/(split+'-tokens.json'),template/(split+'-tokens.json'))
        data=dict(model=model,files={str(f.relative_to(template)):dict(sha256=digest(f),bytes=f.stat().st_size) for f in template.rglob('*') if f.is_file() and f.name!='cache-receipt.json'})
        receipt.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(kind='model_input_cache_ready',model=a.model)),flush=True);return
    cached=json.loads(receipt.read_text())
    if cached['model']!=model:raise ValueError('Model-session input identity mismatch')
    for name,row in cached['files'].items():
        path=template/name
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Input cache checksum mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoint-provenance.json','oracle-tokens.npy'):shutil.copyfile(template/name,a.output/name)
    shutil.copytree(template/'official',a.output/'official')
    for split in ('train','test'):
        tokens=np.load(template/(split+'-all.npy'));count=a.batch*a.sequence
        if len(tokens)<count:raise ValueError('Insufficient tokens; no repetition allowed')
        selected=tokens[:count].reshape(a.batch,a.sequence);np.save(a.output/(split+'-tokens.npy'),selected)
        meta=json.loads((template/(split+'-tokens.json')).read_text());meta.update(shape=list(selected.shape),token_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
        (a.output/(split+'-tokens.json')).write_text(json.dumps(meta,indent=2)+'\n')
    summary=json.loads((template/'summary.json').read_text());summary.update(batch=a.batch,sequence=a.sequence,reused_model_inputs=True,cache_receipt_sha256=digest(receipt),artifacts={str(f.relative_to(a.output)):dict(sha256=digest(f),bytes=f.stat().st_size) for f in a.output.rglob('*') if f.is_file()})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(kind='model_inputs_reused',model=a.model,batch=a.batch,sequence=a.sequence)),flush=True)
if __name__=='__main__':main()
