"""Acquire pinned original N9 models, corpus, tokens and independent CPU oracle.

Run in a dedicated pinned CPU tools venv. No JAX import, no device work, no auth
logging. Each model is independent; a denied or failed input gets an explicit
record without substituting a different model. Cache payloads stay outside the
retrieved artifact tree. No existing paths are overwritten.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
from types import SimpleNamespace
import prepare_model_application_v001 as prep

MODELS = [
    ('qwen', 'Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
    ('mistral', 'mistralai/Mistral-7B-v0.3', 'caa1feb0e54d415e2df31207e5f4e273e33509b1'),
    ('gemma', 'google/gemma-3-1b-pt', 'fcf18a2a879aab110ca39f8bffbccd5d49d8eb29'),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--model-key',choices=[x[0] for x in MODELS],required=True)
    args=p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    required={'torch':'2.8.0','transformers':'4.56.2','tokenizers':'0.22.0','huggingface-hub':'0.34.4',
              'safetensors':'0.6.2','numpy':'2.2.6','pyarrow':'21.0.0','sentencepiece':'0.2.1','protobuf':'6.32.1'}
    versions={k:md.version(k) for k in required}
    if any(versions[k].split('+')[0]!=v for k,v in required.items()):
        raise ValueError('Independent oracle dependencies differ from frozen pins')
    prep.save(out/'versions.json',versions)
    key,model,revision=next(x for x in MODELS if x[0]==args.model_key)
    try:
        acquisition=out/'model_acquisition';acquisition.mkdir()
        prep.model_action(SimpleNamespace(model_id=model,revision=revision,action='download',
            use_existing_token=True,cache_root=args.cache_root,max_model_bytes=18_000_000_000),acquisition)
        corpus=out/'corpus_acquisition';corpus.mkdir()
        # Dedicated model-specific cache makes each preparation retry independent.
        prep.corpus_action(SimpleNamespace(cache_root=args.cache_root),corpus)
        tokens=out/'tokens';tokens.mkdir()
        prep.tokenize_action(SimpleNamespace(model_manifest=acquisition/'model_manifest.json',
            corpus_manifest=corpus/'corpus_manifest.json'),tokens)
        reference=out/'reference';reference.mkdir()
        prep.reference_action(SimpleNamespace(model_manifest=acquisition/'model_manifest.json',
            tokens=tokens/'token_ids.npy'),reference)
        row=dict(model_key=key,model_id=model,revision=revision,status='ready',
            model_manifest=str((acquisition/'model_manifest.json').resolve()),
            tokens_manifest=str((tokens/'tokens_manifest.json').resolve()),
            reference_manifest=str((reference/'reference_manifest.json').resolve()))
        row['input_sha256']={n:prep.digest(Path(row[n])) for n in ('model_manifest','tokens_manifest','reference_manifest')}
    except Exception as exc:
        row=dict(model_key=key,model_id=model,revision=revision,status='input_preparation_failed',error_type=type(exc).__name__)
        # Third-party exception messages may contain signed download redirects.
    prep.save(out/'input_bundle.json',row)
    prep.save(out/'artifact_manifest.json',{'sha256':{str(x.relative_to(out)):prep.digest(x) for x in out.rglob('*') if x.is_file()}})
    print(json.dumps(row if row['status']!='ready' else {k:row[k] for k in ('model_key','model_id','status')}),flush=True)
    return 0 if row['status']=='ready' else 1

if __name__=='__main__':raise SystemExit(main())
