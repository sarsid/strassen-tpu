"""Pinned parent computation; only public checkpoint transport is replaced.

Run each stage in a fresh process. Cached HTTP bytes have the same URL/range
and are hash checked on reuse. No credentials are sent for this public model.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'third_party/strassen_tpu_public_95be1fb_v001'


def configure(output):
    for path in (PARENT, PARENT / 'experiments/qwen3'):
        sys.path.insert(0, str(path))
    run = importlib.import_module('run')
    profile = run.PROFILES['v6e']
    for key in ('product_tile', 'cubic_tile', 'qk_tile', 'site_tiles',
                'extended_sites', 'strassen_limit_mib', 'cubic_limit_mib', 'scoped_vmem_kib'):
        os.environ['QWEN3_' + key.upper()] = profile[key]
    os.environ.update(QWEN3_MODEL='32b', QWEN3_SEQUENCE='1024', QWEN3_FUSED_QK='1',
                      QWEN3_OUTPUT_SUFFIX='', QWEN3_STREAM_PRODUCT_AWARE='1',
                      STRASSEN_OUTPUT_DIR=str(output), HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    return run


def install_transport(layer, cache, emit):
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    layer._hub_token = lambda: None

    def get(self, url, byte_range=None):
        if not url.startswith(layer.BASE + '/'):
            raise ValueError('Checkpoint request escaped the frozen public revision')
        request_key = json.dumps([url, byte_range], separators=(',', ':'))
        key = hashlib.sha256(request_key.encode()).hexdigest()
        data = cache / (key + '.bin')
        meta = cache / (key + '.json')
        started = time.monotonic()
        expected = None
        if byte_range:
            match = re.fullmatch(r'bytes=(\d+)-(\d+)', byte_range)
            if not match:
                raise ValueError('Invalid byte range')
            first, last = map(int, match.groups())
            expected = last - first + 1
            if expected <= 0 or expected > 2 * 1024**3:
                raise ValueError('Invalid or excessive checkpoint range')
        hit = data.exists() and meta.exists()
        if hit:
            info = json.loads(meta.read_text())
            payload = data.read_bytes()
            if info['key'] != request_key or hashlib.sha256(payload).hexdigest() != info['sha256']:
                raise ValueError('Checkpoint cache integrity failure')
        else:
            last_error = None
            for attempt in range(5):
                try:
                    # Range-specific URL prevents a proxy reusing a different range.
                    fetch_url = url + ('?replication_range=' + str(first) + '-' + str(last) if byte_range else '')
                    with self.session.get(fetch_url, headers={'Range': byte_range} if byte_range else {},
                                          stream=True, timeout=(30, 180)) as response:
                        if response.status_code != (206 if byte_range else 200):
                            raise RuntimeError('HTTP status ' + str(response.status_code))
                        if byte_range:
                            got = response.headers.get('Content-Range', '')
                            if not got.startswith(f'bytes {first}-{last}/'):
                                raise RuntimeError('Content-Range does not match requested interval')
                        partial = cache / (key + '.partial')
                        count = 0
                        with partial.open('wb') as target:
                            for chunk in response.iter_content(4 * 1024**2):
                                count += len(chunk)
                                if count > (expected if expected is not None else 8 * 1024**2):
                                    raise RuntimeError('Checkpoint response exceeds requested size')
                                target.write(chunk)
                        if expected is not None and count != expected:
                            raise RuntimeError('Incomplete checkpoint range')
                    payload = partial.read_bytes()
                    info = dict(key=request_key, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                    partial.replace(data)
                    meta.write_text(json.dumps(info) + '\n')
                    break
                except (requests.RequestException, OSError, RuntimeError) as error:
                    # Exception strings may contain signed redirected URLs: omit them.
                    last_error = type(error).__name__
                    emit(dict(kind='transport_retry', cache_key=key, attempt=attempt + 1, error_type=last_error))
                    time.sleep(min(2**attempt, 16))
            else:
                raise RuntimeError('Public checkpoint transport failed: ' + str(last_error))
        if expected is not None and len(payload) != expected:
            raise ValueError('Cached interval length mismatch')
        emit(dict(kind='checkpoint_transport', cache_key=key, range=byte_range,
                  cache_hit=hit, bytes=len(payload), sha256=info['sha256'], seconds=time.monotonic()-started))
        answer = requests.Response()
        answer.status_code = 206 if byte_range else 200
        answer._content = payload
        return answer

    layer.ShardedSafetensors._get = get


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['layer', 'streamed', 'quality'], required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cache', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def emit(value):
        line = json.dumps(value, sort_keys=True)
        with (args.output / 'events.jsonl').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)
    run = configure(args.output)
    module = importlib.import_module(run.STAGES[args.stage])
    layer = importlib.import_module('benchmark_qwen3_32b_layer')
    install_transport(layer, args.cache, emit)
    import jax
    devices = jax.devices()
    if len(devices) != 1 or 'v6' not in devices[0].device_kind.lower():
        raise RuntimeError('Expected one v6e device')
    emit(dict(kind='replication_environment', stage=args.stage, python=sys.version,
              device=devices[0].device_kind, versions={n: md.version(n) for n in ('jax', 'jaxlib', 'libtpu', 'transformers', 'tokenizers', 'datasets')},
              libtpu_init_args=os.environ.get('LIBTPU_INIT_ARGS'),
              parent_commit='95be1fb088656a89813b04492e1d77c66b36ccf9',
              changes='validated cached public HTTP transport only; parent computation unmodified',
              effective_down_tile=[2048,1024,512] if args.stage == 'layer' else [2048,2560,1024]))
    # Parent owns its original outputs, but mirror every record into this stage's
    # unified ledger for live status and safe incremental downloading.
    original_emit = module.emit
    def both(value):
        original_emit(value)
        emit(value)
    module.emit = both
    layer.emit = both
    started = time.monotonic()
    try:
        module.main()
        status = dict(status='completed', wall_seconds=time.monotonic()-started)
    except BaseException as error:
        status = dict(status='failed', error_type=type(error).__name__, wall_seconds=time.monotonic()-started)
        raise
    finally:
        (args.output/'completion.json').write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__':
    main()
