#!/usr/bin/env python3
"""Official model/corpus acquisition and isolated CPU qualification for N9.

All outputs are new directories. Large checkpoint inputs stay in a separate
immutable cache; manifests, token arrays and reference outputs are execution
evidence. This tool never imports JAX or accepts model access terms.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import requests

MODELS = ('Qwen/Qwen3-0.6B', 'mistralai/Mistral-7B-v0.3', 'google/gemma-3-1b-pt')
DATASET = 'Salesforce/wikitext'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
TOOLS_PINS = ['transformers==4.56.2', 'tokenizers==0.22.0', 'huggingface-hub==0.34.4',
              'safetensors==0.6.2', 'numpy==2.2.6', 'pyarrow==21.0.0', 'sentencepiece==0.2.1',
              'accelerate==1.10.1']


def utc(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write('\n')


class AccessBlocked(RuntimeError): pass
class InvalidArtifact(RuntimeError): pass


def token_if_configured(enabled):
    if not enabled: return None
    value = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if value: return value.strip()
    paths = [Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'token',
             Path.home()/'.huggingface/token']
    for path in paths:
        if path.is_file():
            value = path.read_text().strip()
            if value: return value
    return None


class PublicSource:
    def __init__(self, output, use_token=False):
        self.output, self.count = output, 0
        (output/'requests').mkdir()
        self.session = requests.Session(); self.session.trust_env = False
        self.session.headers['Accept-Encoding'] = 'identity'
        token = token_if_configured(use_token)
        if token: self.session.headers['Authorization'] = 'Bearer ' + token
        self.token_configured = bool(token)

    def get(self, canonical, path=None, maximum=4*1024**2, access_probe=False):
        self.count += 1
        item = {'canonical_url':canonical,'started_utc':utc(), 'maximum_bytes':maximum,
                'existing_token_configured':self.token_configured, 'access_probe':access_probe}
        response = None
        try:
            headers = {'Range':'bytes=0-7'} if access_probe else {}
            response = self.session.get(canonical, headers=headers, stream=True,
                                        timeout=(30,120), allow_redirects=True)
            item['http_status'] = response.status_code
            if response.status_code in (401,403):
                raise AccessBlocked('Official repository denied access; no terms acceptance attempted')
            if response.status_code not in ((200,206) if access_probe else (200,)):
                raise InvalidArtifact('Official endpoint returned an unexpected HTTP status')
            if access_probe:
                item['body_bytes_read'] = 0
                return item
            if response.headers.get('Content-Encoding','identity').lower() not in ('','identity'):
                raise InvalidArtifact('Encoded body prevents exact file accounting')
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdigit() or int(length)>maximum):
                raise InvalidArtifact('Content length exceeds frozen acquisition bound')
            total, h = 0, hashlib.sha256()
            with path.open('xb') as f:
                while True:
                    chunk = response.raw.read(min(1024**2, maximum-total+1), decode_content=False)
                    if not chunk: break
                    if total+len(chunk)>maximum: raise InvalidArtifact('Response exceeded acquisition bound')
                    f.write(chunk); h.update(chunk); total += len(chunk)
            if length is not None and total != int(length):
                raise InvalidArtifact('Truncated official file response')
            item.update(bytes=total,sha256=h.hexdigest(),complete=True)
            return item
        except BaseException as error:
            item['error_type'] = type(error).__name__
            if isinstance(error,(AccessBlocked,InvalidArtifact)): item['error']=str(error)
            raise
        finally:
            if response is not None: response.close()
            item['finished_utc']=utc()
            # Signed redirect URLs, request auth and third-party exception text
            # are deliberately never recorded.
            save(self.output/'requests'/f'{self.count:04d}.json',item)


def metadata(source, model, revision, output, dataset=False):
    kind = 'datasets' if dataset else 'models'
    url = f'https://huggingface.co/api/{kind}/{model}/revision/{revision}?blobs=true'
    path = output/'repository.json'
    source.get(url,path)
    meta = json.loads(path.read_text()); sha=meta.get('sha','')
    if not re.fullmatch('[0-9a-f]{40}',sha): raise InvalidArtifact('Repository revision is not an exact commit')
    if revision!='main' and revision!=sha: raise InvalidArtifact('Frozen revision changed')
    return meta,sha


def selected_model_files(meta):
    files={item['rfilename']:item for item in meta['siblings']}
    if 'model.safetensors' in files: weights=['model.safetensors']
    else: weights=sorted(x for x in files if re.fullmatch(r'model-\d+-of-\d+\.safetensors',x))
    if not weights: raise InvalidArtifact('Official Transformers safetensors weights are absent')
    names=['config.json']+weights
    optional=('model.safetensors.index.json','generation_config.json','tokenizer.json',
              'tokenizer_config.json','special_tokens_map.json','tokenizer.model','tokenizer.model.v3',
              'vocab.json','merges.txt','README.md','LICENSE','LICENSE.txt','LICENSE.md','NOTICE')
    names += [x for x in optional if x in files]
    if not any(x.startswith('tokenizer') for x in names):
        raise InvalidArtifact('Official tokenizer files are absent')
    return files,names,weights


def model_action(args,output):
    source=PublicSource(output,args.use_existing_token)
    try:
        meta,revision=metadata(source,args.model_id,args.revision,output)
        files,names,weights=selected_model_files(meta)
        base=f'https://huggingface.co/{args.model_id}/resolve/{revision}/'
        source.get(base+'config.json',output/'config.json')
        source.get(base+weights[0],access_probe=True)
        report={'model_id':args.model_id,'revision':revision,'gated':meta.get('gated',False),
                'official_source':base,'existing_token_configured':source.token_configured,
                'access':'available','files':names,'config':json.loads((output/'config.json').read_text()),
                'download_policy':'Official repository only; no license acceptance or mirrors'}
        if args.action=='probe':
            save(output/'probe.json',report); return report
        cache=args.cache_root/args.model_id.replace('/','--')/revision
        cache.mkdir(parents=True,exist_ok=False)
        records=[]; total=0
        for name in names:
            if '/' in name or '..' in name: raise InvalidArtifact('Unsupported model filename')
            info=files[name]; lfs=info.get('lfs') or {}
            expected=lfs.get('size') or info.get('size')
            bound=expected if isinstance(expected,int) else (args.max_model_bytes if name.endswith('.safetensors') else 16*1024**2)
            if bound>args.max_model_bytes-total: raise InvalidArtifact('Model exceeds registered cache byte budget')
            result=source.get(base+name,cache/name,maximum=bound)
            expected_hash=lfs.get('sha256') or lfs.get('oid')
            if expected_hash and re.fullmatch('[0-9a-f]{64}',expected_hash) and result['sha256']!=expected_hash:
                raise InvalidArtifact('Checkpoint LFS SHA256 differs from official metadata')
            total+=result['bytes']
            records.append({'path':name,'bytes':result['bytes'],'sha256':result['sha256'],
                            'canonical_url':base+name,'official_lfs_sha256':expected_hash})
            print(json.dumps({'kind':'model_file','model_id':args.model_id,'path':name,'bytes':result['bytes']}),flush=True)
        manifest={**report,'cache_dir':str(cache),'files':records,'total_bytes':total,
                  'created_utc':utc(),'input_role':'external reproducible checkpoint; not code or benchmark results'}
        save(cache/'manifest.json',manifest); save(output/'model_manifest.json',manifest)
        return manifest
    finally: source.session.close()


def corpus_action(args,output):
    source=PublicSource(output)
    try:
        meta,revision=metadata(source,DATASET,DATA_REVISION,output,dataset=True)
        names=[x['rfilename'] for x in meta['siblings']
               if x['rfilename'].startswith('wikitext-2-raw-v1/') and '/test-' in x['rfilename']
               and x['rfilename'].endswith('.parquet')]
        if not names: raise InvalidArtifact('Pinned WikiText2 raw test parquet was not found')
        cache=args.cache_root/'corpus'/revision; cache.mkdir(parents=True,exist_ok=False)
        records=[]
        for index,name in enumerate(sorted(names)):
            path=cache/f'test-{index:03d}.parquet'
            url=f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}'
            item=source.get(url,path,maximum=32*1024**2)
            records.append({'path':path.name,'canonical_url':url,'sha256':item['sha256'],'bytes':item['bytes']})
        url=f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/README.md'
        card=source.get(url,cache/'README.md')
        manifest={'dataset_id':DATASET,'revision':revision,'split':'test','configuration':'wikitext-2-raw-v1',
                  'cache_dir':str(cache),'files':records,'card_sha256':card['sha256'],
                  'selection':'First 32768 contiguous tokens per model tokenizer; no outcome-based text selection'}
        save(cache/'manifest.json',manifest); save(output/'corpus_manifest.json',manifest)
        return manifest
    finally: source.session.close()


def model_tools_action(args,output):
    if args.venv_dir.exists(): raise FileExistsError('Isolated model-tools environment already exists')
    commands=[[sys.executable,'-m','venv',str(args.venv_dir)],
              [str(args.venv_dir/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check',
               '--index-url','https://download.pytorch.org/whl/cpu','torch==2.8.0'],
              [str(args.venv_dir/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check',*TOOLS_PINS]]
    for index,command in enumerate(commands):
        save(output/f'command-{index}.json',{'argv':command})
        with (output/f'install-{index}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    return {'venv_dir':str(args.venv_dir),'python':str(args.venv_dir/'bin/python'),
            'pins':['torch==2.8.0',*TOOLS_PINS],'core_jax_environment_modified':False}


def verify_manifest(path):
    value=json.loads(path.read_text()); directory=Path(value['cache_dir'])
    for item in value['files']:
        file=directory/item['path']
        if not file.resolve().is_relative_to(directory.resolve()) or digest(file)!=item['sha256']:
            raise InvalidArtifact('External input file failed manifest hash verification')
    return value,directory


def tokenize_action(args,output):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    model,model_dir=verify_manifest(args.model_manifest)
    corpus,corpus_dir=verify_manifest(args.corpus_manifest)
    texts=[]
    for item in corpus['files']:
        texts.extend(pq.read_table(corpus_dir/item['path'],columns=['text']).column('text').to_pylist())
    text='\n'.join(texts)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
    ids=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
    if len(ids)<32768: raise InvalidArtifact('Corpus produced fewer than the registered token count')
    tokens=np.asarray(ids[:32768],dtype='<i4').reshape(32,1024)
    with (output/'token_ids.npy').open('xb') as f: np.save(f,tokens,allow_pickle=False)
    manifest={'model_id':model['model_id'],'model_revision':model['revision'],
              'model_manifest_sha256':digest(args.model_manifest),
              'dataset_id':corpus['dataset_id'],'dataset_revision':corpus['revision'],
              'corpus_manifest_sha256':digest(args.corpus_manifest),
              'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
              'token_array_path':'token_ids.npy','token_array_sha256':digest(output/'token_ids.npy'),
              'raw_token_bytes_sha256':hashlib.sha256(tokens.tobytes()).hexdigest(),
              'shape':[32,1024],'scored_positions':32*1023,'unique_token_ids':int(np.unique(tokens).size),
              'full_corpus_token_count':len(ids),'add_special_tokens':False,
              'selection':'first 32768 contiguous tokens, nonoverlapping 1024-token windows',
              'versions':{n:md.version(n) for n in ('transformers','tokenizers','numpy','pyarrow')}}
    save(output/'tokens_manifest.json',manifest); return manifest


def reference_action(args,output):
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    model_info,model_dir=verify_manifest(args.model_manifest)
    tokens=np.load(args.tokens,allow_pickle=False)
    ids=tokens.reshape(-1)[:64].astype(np.int64).reshape(1,64)
    torch.set_num_threads(1)
    start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True)
    model.eval()
    with torch.inference_mode():
        value=model(torch.from_numpy(ids),use_cache=False,output_hidden_states=True)
    arrays={'input_ids':ids.astype('<i4'),'logits':value.logits.float().cpu().numpy(),
            'final_hidden':value.hidden_states[-1].float().cpu().numpy()}
    files={}
    for name,array in arrays.items():
        with (output/(name+'.npy')).open('xb') as f: np.save(f,array,allow_pickle=False)
        files[name]={'path':name+'.npy','sha256':digest(output/(name+'.npy')),'shape':list(array.shape)}
    report={'model_id':model_info['model_id'],'revision':model_info['revision'],
            'model_manifest_sha256':digest(args.model_manifest),'tokens_sha256':digest(args.tokens),
            'implementation':'official Transformers AutoModelForCausalLM, eager attention, CPU BF16, eval(), no KV cache',
            'model_class':type(model).__name__,'files':files,'elapsed_seconds':time.monotonic()-start,
            'versions':{n:md.version(n) for n in ('torch','transformers','safetensors','numpy')},
            'role':'Independent short-sequence implementation qualification; not the 32768-token quality study'}
    save(output/'reference_manifest.json',report); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('probe','download','corpus','model-tools','tokenize','reference'):
        p=sub.add_parser(action); p.add_argument('--output-dir',required=True,type=Path)
        if action in ('probe','download'):
            p.add_argument('--model-id',required=True,choices=MODELS); p.add_argument('--revision',default='main')
            p.add_argument('--use-existing-token',action='store_true')
        if action in ('download','corpus'): p.add_argument('--cache-root',required=True,type=Path)
        if action=='download': p.add_argument('--max-model-bytes',type=int,default=18_000_000_000)
        if action=='model-tools': p.add_argument('--venv-dir',required=True,type=Path)
        if action in ('tokenize','reference'): p.add_argument('--model-manifest',required=True,type=Path)
        if action=='tokenize': p.add_argument('--corpus-manifest',required=True,type=Path)
        if action=='reference': p.add_argument('--tokens',required=True,type=Path)
    args=parser.parse_args(); output=args.output_dir; output.mkdir(parents=True,exist_ok=False)
    status={'started_utc':utc(),'action':args.action,'status':'running'}; code=1
    try:
        if args.action in ('probe','download'):
            if args.revision!='main' and not re.fullmatch('[0-9a-f]{40}',args.revision):
                raise InvalidArtifact('Revision must be main or an exact 40-hex commit')
            result=model_action(args,output)
        else:
            result={'corpus':corpus_action,'model-tools':model_tools_action,
                    'tokenize':tokenize_action,'reference':reference_action}[args.action](args,output)
        save(output/'result.json',result); status['status']='completed'; code=0
    except BaseException as error:
        status.update(status='blocked_access' if isinstance(error,AccessBlocked) else 'failed',
                      error_type=type(error).__name__)
        if isinstance(error,(AccessBlocked,InvalidArtifact)): status['error']=str(error)
    finally:
        status['finished_utc']=utc(); save(output/'status.json',status)
        save(output/'artifact_manifest.json',{'files':{str(p.relative_to(output)):
             {'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'kind':'application_preparation','output_dir':str(output),**status}),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
