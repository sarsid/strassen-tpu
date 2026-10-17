"""Run unmodified parent modules with published or frozen retuned settings."""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import time

import parent_stage_v001 as transport

GATE_ENV = dict(QWEN3_TILE_BM='512,1024,2048,4096',
                QWEN3_TILE_BN='1024,2048,2560,5120',
                QWEN3_TILE_BK='512,1024,2560,5120',
                QWEN3_STRASSEN_LIMIT_MIB='100', QWEN3_CUBIC_LIMIT_MIB='104',
                QWEN3_SCOPED_VMEM_KIB='131072')


def verify_parent():
    provenance = json.loads((transport.PARENT/'provenance.json').read_text())
    for name, info in provenance['files'].items():
        data = (transport.PARENT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != info['sha256'] or len(data) != info['bytes']:
            raise ValueError('Modified parent file: ' + name)
    return provenance


def configure(stage, output, profile=None):
    # A new process is mandatory: parent modules bind environment at import.
    for key in list(os.environ):
        if key.startswith('QWEN3_'):
            del os.environ[key]
    run = transport.configure(output)
    if stage == 'gate-tune':
        os.environ.update(GATE_ENV)
        return 'benchmark_qwen3_scaling_tile_tune'
    if stage == 'site-tune':
        return 'benchmark_qwen3_site_tile_tune'
    if profile is not None:
        if set(profile) != {'product_tile', 'cubic_tile', 'site_tiles'}:
            raise ValueError('Retuning may override only gate/up and O/down tile settings')
        os.environ.update({'QWEN3_'+k.upper(): v for k, v in profile.items()})
    return run.STAGES[stage]


def chosen(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    selections = [r for r in records if r.get('kind') == 'chosen']
    verdicts = [r for r in records if r.get('kind') == 'verdict']
    if len(selections) != 1 or len(verdicts) != 1 or verdicts[0].get('passes') is not True:
        raise ValueError('Tuning incomplete or failed: ' + str(path))
    return selections[0]


def freeze_profile(gate_path, site_path, output):
    gate, sites = chosen(gate_path)['winners'], chosen(site_path)['winners']
    for tile in [gate['product_strassen']['tile'], gate['cubic']['tile'],
                 sites['o']['tile'], sites['down']['tile']]:
        if len(tile) != 3 or any(type(v) is not int or v <= 0 for v in tile):
            raise ValueError('Invalid selected tile')
    join = lambda values: ','.join(map(str, values))
    profile = dict(product_tile=join(gate['product_strassen']['tile']),
                   cubic_tile=join(gate['cubic']['tile']),
                   site_tiles=';'.join(s+':'+join(sites[s]['tile']) for s in ['o','down']))
    with output.open('x') as f:
        json.dump(dict(profile=profile, source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [gate_path, site_path]},
                  scope='parent gate/up cubic and Strassen plus O/down; published fused-QK retained'), f, indent=2)
        f.write('\n')
    return profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['layer','streamed','quality','gate-tune','site-tune'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def emit(value):
        line = json.dumps(value, sort_keys=True)
        with (a.output/'events.jsonl').open('a') as f:
            f.write(line+'\n')
        print(line, flush=True)
    provenance = verify_parent()
    profile = json.loads(a.profile.read_text())['profile'] if a.profile else None
    module_name = configure(a.stage, a.output, profile)
    module = importlib.import_module(module_name)
    layer = importlib.import_module('benchmark_qwen3_32b_layer')
    transport.install_transport(layer, a.cache, emit)
    import jax
    if len(jax.devices()) != 1 or 'v6' not in jax.devices()[0].device_kind.lower():
        raise RuntimeError('Expected one v6e TPU')
    versions = {name: md.version(name) for name in ['jax','jaxlib','libtpu','transformers','tokenizers','datasets']}
    if any(versions[k] != v for k,v in {'jax':'0.7.2','jaxlib':'0.7.2','libtpu':'0.0.21.1'}.items()):
        raise RuntimeError('Parent software stack does not match')
    emit(dict(kind='exact_parent_environment', stage=a.stage,
              parent_commit=provenance['commit'], verified_files=len(provenance['files']),
              versions=versions, python=sys.version, device=jax.devices()[0].device_kind,
              environment={k:v for k,v in os.environ.items() if k.startswith('QWEN3_') or k=='LIBTPU_INIT_ARGS'},
              profile='retuned' if a.profile else 'published',
              modifications='verified public HTTP cache and duplicate logging only; computation unchanged'))
    original = module.emit
    def both(value):
        original(value)
        emit(value)
    module.emit = both
    layer.emit = both
    started = time.monotonic()
    status = dict(status='failed')
    try:
        module.main()
        status['status'] = 'completed'
    finally:
        status['wall_seconds'] = time.monotonic()-started
        (a.output/'completion.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__ == '__main__':
    main()
