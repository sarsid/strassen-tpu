"""Acquire pinned official Gemma weights and deterministic qualification tokens.

Gated payloads stay in ignored private cache; only provenance/hashes and public
corpus token IDs are archived. Never print auth material or signed redirect URLs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import prepare_model_application_v001 as prep

MODEL = 'google/gemma-3-1b-pt'
REVISION = 'fcf18a2a879aab110ca39f8bffbccd5d49d8eb29'


def main():
    root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir()
    acquisition = out / 'model_acquisition'
    acquisition.mkdir()
    cache_root = root / '.runtime_private/gemma-checkpoint-v001'
    cache = cache_root / MODEL.replace('/', '--') / REVISION
    cache.mkdir(parents=True, exist_ok=False)
    source = prep.PublicSource(acquisition, use_token=True)
    try:
        meta, revision = prep.metadata(source, MODEL, REVISION, acquisition)
        files, names, weights = prep.selected_model_files(meta)
        if 'added_tokens.json' in files:
            names.append('added_tokens.json')
        records = []
        total = 0
        for name in names:
            info = files[name]
            lfs = info.get('lfs') or {}
            expected = lfs.get('size') or info.get('size')
            if '/' in name or '..' in name or not isinstance(expected, int) or expected <= 0:
                raise prep.InvalidArtifact('Unexpected official file descriptor')
            if total + expected > 3 * 1024 ** 3:
                raise prep.InvalidArtifact('Pinned checkpoint exceeds acquisition budget')
            canonical = f'https://huggingface.co/{MODEL}/resolve/{revision}/{name}'
            item = source.get(canonical, cache / name, maximum=expected)
            expected_hash = lfs.get('sha256') or lfs.get('oid')
            if expected_hash and item['sha256'] != expected_hash:
                raise prep.InvalidArtifact('Downloaded file differs from official LFS hash')
            records.append({'path': name, 'bytes': item['bytes'], 'sha256': item['sha256'],
                            'canonical_url': canonical, 'official_lfs_sha256': expected_hash})
            total += item['bytes']
            print(json.dumps({'event': 'model_file_verified', 'path': name, 'bytes': item['bytes']}), flush=True)
        config = json.loads((cache / 'config.json').read_text())
        manifest = {'model_id': MODEL, 'revision': revision, 'cache_dir': str(cache),
                    'config': config, 'files': records, 'total_bytes': total,
                    'input_role': 'external reproducible official checkpoint',
                    'created_utc': datetime.now(timezone.utc).isoformat()}
        prep.save(out / 'model_manifest.json', manifest)
        prep.save(cache / 'manifest.json', manifest)
    finally:
        source.session.close()

    corpus_out = out / 'corpus_acquisition'
    corpus_out.mkdir()
    corpus = prep.corpus_action(SimpleNamespace(cache_root=cache_root), corpus_out)
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    texts = []
    for item in corpus['files']:
        texts.extend(pq.read_table(Path(corpus['cache_dir']) / item['path'], columns=['text']).column('text').to_pylist())
    text = '\n'.join(texts)
    tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False, use_fast=True)
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)['input_ids']
    if len(ids) < 32768:
        raise prep.InvalidArtifact('Official corpus has insufficient qualification tokens')
    tokens = np.asarray(ids[:32768], dtype='<i4')
    with (out / 'token_ids.npy').open('xb') as handle:
        np.save(handle, tokens, allow_pickle=False)
    prep.save(out / 'token_manifest.json', {
        'model_id': MODEL, 'revision': REVISION, 'tokens_sha256': prep.digest(out / 'token_ids.npy'),
        'token_count': int(tokens.size), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'model_manifest_sha256': prep.digest(out / 'model_manifest.json'),
        'dataset_id': corpus['dataset_id'], 'dataset_revision': corpus['revision'],
        'selection': 'first32768 token IDs of official WikiText2 raw test; no special tokens',
        'add_special_tokens': False, 'tokenizer_class': type(tokenizer).__name__,
    })
    prep.save(out / 'summary.json', {'status': 'completed', 'model_id': MODEL, 'revision': REVISION,
              'checkpoint_bytes': total, 'token_count': int(tokens.size),
              'full_corpus_tokens': len(ids), 'performance_measurements': False})
    print(json.dumps({'event': 'inputs_ready', 'model_id': MODEL, 'tokens': int(tokens.size)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
