"""Promote exact parent S1 without rewriting any measured tuning decisions."""
import argparse
import hashlib
import json
from pathlib import Path
from strassen_mm.parent_fused_adapter_v002 import exact_spec, COMPILER_OPTIONS


def adopt(source):
    old = json.loads(source.read_text())
    if (old['model'], old['batch'], old['sequence'], old['native_compiler_options']) != (
            'Qwen3-32B', 8, 1024, COMPILER_OPTIONS):
        raise ValueError('Extension profile does not match the adopted workload/options')
    profiles = {
        dtype: {family: dict(backend='extension', family=family, output_dtype=dtype, policy=policy)
                for family, policy in entries.items()}
        for dtype, entries in old['profiles'].items()
    }
    profiles['bfloat16']['s1'] = exact_spec()
    matched_cubic = profiles['bfloat16']['cubic']
    profiles['bfloat16']['cubic'] = exact_spec('cubic')
    return dict(version='parent-adopted-v001', model='Qwen3-32B', device='v6e', batch=8, sequence=1024,
                default_dtype='bfloat16', default_family='s1',
                native_compiler_options=dict(COMPILER_OPTIONS), profiles=profiles,
                native=exact_spec('native'), parent_cubic_control=exact_spec('cubic'),
                matched_cubic_control=matched_cubic,
                extension_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                selection='User-selected adoption of reproduced parent S1; no new tuning or timing',
                cubic_control='Exact parent cubic by default; use matched_cubic_control to isolate '
                              'gain over our previously tuned equally fused cubic',
                precision='Parent S1 uses DEFAULT fused normalization. Extensions retain HIGHEST; '
                          'custom FP32 stores feed BF16 model consumers.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = adopt(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
