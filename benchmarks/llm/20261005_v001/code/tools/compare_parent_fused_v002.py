"""Evaluate adopted parent S1 by default; extensions and controls are explicit.

Default execution retains the original parent functions and parameter trees.
Use --cubic-control matched to compare with our equally fused cubic, or select
--family s2 / --dtype float32 for the preserved research extensions. Each such
comparison records its actual backend; none is labelled an exact reproduction.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import parent_stage_v002 as setup
from compare_parent_fused_v001 import JaxOptionsProxy
from strassen_mm.parent_fused_adapter_v002 import build, COMPILER_OPTIONS, verify_sources

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / 'configs/qwen3_32b_v6e_parent_adopted_v001.json'


def resolve(profile, dtype, family, cubic_control):
    if profile['version'] != 'parent-adopted-v001':
        raise ValueError('Expected an explicit adopted-parent profile')
    if profile['native_compiler_options'] != COMPILER_OPTIONS:
        raise ValueError('Adopted parent requires its qualified 48 MiB Native setting')
    cubic = (profile['matched_cubic_control'] if cubic_control == 'matched' and dtype == 'bfloat16'
             else profile['profiles'][dtype]['cubic'])
    if cubic_control == 'parent':
        if dtype != 'bfloat16':
            raise ValueError('The exact parent cubic is BF16; select --cubic-control matched for FP32')
        cubic = profile['parent_cubic_control']
    return dict(regular_xla=profile['native'], gated_cubic=cubic,
                gated_strassen=profile['profiles'][dtype][family])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--family', choices=['s1', 's2'], default='s1')
    p.add_argument('--cubic-control', choices=['parent', 'matched'], default='parent')
    p.add_argument('--stage', choices=['streamed', 'quality'], default='streamed')
    p.add_argument('--dry-run', action='store_true')
    return p


def main():
    p = parser(); args = p.parse_args()
    profile = json.loads(args.profile.read_text())
    specs = resolve(profile, args.dtype, args.family, args.cubic_control)
    verify_sources()
    direct = all(spec['backend'] == 'parent_exact' for spec in specs.values())
    plan = dict(version='parent-adopted-evaluation-v001', stage=args.stage,
                family=args.family, custom_store_dtype=args.dtype, consumer_dtype='bfloat16',
                cubic_control=args.cubic_control, specs=specs,
                parameter_path='original parent' if direct else 'prepared extension bundle',
                profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                compiler_options=profile['native_compiler_options'],
                scope='Sum of resident layer compute; excludes checkpoint I/O and weight preparation/transfer')
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if args.output is None or args.cache is None:
        p.error('--output and --cache are required for execution')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def emit(value):
        if value.get('kind') == 'tokens':
            expected = ('4aa8e3628d16c9a26876aedb8c99525c6a15ee601c88d93deadae15855a84d74'
                        if args.stage == 'quality' else
                        '70b86198317db1dee0c351568cf420d78242aa74e0b9a6cbb14925eaa55ca2a5')
            if value['sha256'] != expected:
                raise RuntimeError('Frozen evaluation tokens differ')
        if value.get('kind') == 'task':
            for result in value['results'].values():
                result['native_perplexity'] = math.exp(result['native_loss'])
                result['candidate_perplexity'] = math.exp(result['candidate_loss'])
        if value.get('kind') == 'metadata':
            value = dict(kind='metadata', parent_harness_metadata=value, evaluation=plan,
                         builders={name: fn.metadata for name, fn in builders.items()},
                         note='Parent cubic has fewer fusions than S1; its margin is not pure Strassen savings')
        line = json.dumps(value, sort_keys=True)
        with (args.output / 'events.jsonl').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)

    status = dict(status='failed', stage=args.stage)
    try:
        setup.configure(args.output)
        stream = importlib.import_module('benchmark_qwen3_32b_streamed_inference')
        layer = importlib.import_module('benchmark_qwen3_32b_layer')
        module = stream if args.stage == 'streamed' else importlib.import_module('benchmark_qwen3_streamed_natural_gate')
        setup.install_transport(layer, args.cache, emit)
        import jax
        versions = {n: md.version(n) for n in ('jax', 'jaxlib', 'libtpu')}
        if versions != dict(jax='0.11.2', jaxlib='0.11.2', libtpu='0.0.48'):
            raise RuntimeError('Unexpected current stack')
        if len(jax.devices()) != 1 or 'v6' not in jax.devices()[0].device_kind.lower():
            raise RuntimeError('Expected one v6e')
        emit(dict(kind='environment', versions=versions, device=jax.devices()[0].device_kind))
        builders = {name: build(layer, spec, stream=stream) for name, spec in specs.items()}
        original_load = stream.load_layer
        if direct:
            # No wrapper around the model function and no new parameter bundle.
            def make(name):
                return builders[name]
        else:
            def load(checkpoint, index, cos, sin):
                params = original_load(checkpoint, index, cos, sin)
                cache = {}
                packed = {name: fn.prepare_weights(params, cache) for name, fn in builders.items()}
                jax.block_until_ready(packed)
                return packed

            def make(name):
                def fn(x, packed):
                    return builders[name](x, packed[name])
                fn.replication_compiler_options = profile['native_compiler_options']
                return fn
            stream.load_layer = load
        stream.make_product_layer = make
        stream.jax = JaxOptionsProxy(jax)
        if module is not stream:
            module.jax = JaxOptionsProxy(jax)
        module.emit = stream.emit = layer.emit = emit
        module.main()
        status = dict(status='completed', stage=args.stage, family=args.family, dtype=args.dtype)
    except BaseException as error:
        status['error_type'] = type(error).__name__
        raise
    finally:
        (args.output / 'summary.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
