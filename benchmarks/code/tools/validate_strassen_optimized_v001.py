#!/usr/bin/env python3
"""Archive CPU correctness and TPU-targeted lowering, never TPU performance.

Run from an immutable source snapshot. The output directory must be new.
Requires the qualified local JAX 0.7.2 runtime; no TPU allocation is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def child(out, name, command):
    with (out / (name + '.log')).open('x') as log:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True,
                       stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'scope': 'CPU correctness and local TPU-targeted lowering only',
               'tpu_hardware_used': False, 'tpu_device_compilation_tested': False,
               'performance_measured': False, 'passed': False}
    try:
        import jax
        import jax.numpy as jnp
        from jax import export
        from strassen_mm import strassen_optimized as opt
        from strassen_mm import strassen_schedule_v001 as schedule
        write(out / 'environment.json', {'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'jax': jax.__version__,
              'devices': [str(d) for d in jax.devices()]})
        if jax.__version__ != '0.7.2' or any(d.platform != 'cpu' for d in jax.devices()):
            raise RuntimeError('This qualification protocol requires JAX 0.7.2 on CPU')

        ledger = json.loads((ROOT / 'reports/strassen_optimized_ledger_v001.json').read_text())
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ledger['baseline_sha256']}
        write(out / 'baseline_preservation.json', {'expected': ledger['baseline_sha256'],
              'actual': hashes, 'unchanged': hashes == ledger['baseline_sha256']})
        if hashes != ledger['baseline_sha256']:
            raise RuntimeError('Frozen baseline source changed')

        suite = unittest.TestSuite()
        for pattern in ('test_strassen_optimized_v001.py', 'test_strassen_schedule_v001.py',
                        'test_benchmark_strassen_optimized_v001.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
        with (out / 'unit_tests.log').open('x') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        summary['unit_tests'] = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped), 'passed': result.wasSuccessful()}
        if not result.wasSuccessful():
            raise RuntimeError('Unit qualification failed; inspect unit_tests.log')

        ir_dir = out / 'tpu_lowered_ir'
        ir_dir.mkdir()
        lowering = []
        for m, n, k in ((16, 256, 256), (17, 257, 259), (3, 129, 5)):
            for variant in opt.VARIANTS:
                for traversal in opt.TRAVERSALS:
                    fn = opt.make_matmul((m, n, k), (16, 256, 256), variant=variant, traversal=traversal)
                    exported = export.export(jax.jit(fn), platforms=['tpu'])(
                        jax.ShapeDtypeStruct((m, k), jnp.bfloat16),
                        jax.ShapeDtypeStruct((k, n), jnp.bfloat16))
                    name = f'{variant}_{traversal}_{m}_{n}_{k}.mlir'
                    with (ir_dir / name).open('x') as stream:
                        stream.write(exported.mlir_module())
                    lowering.append({'shape_mnk': [m, n, k], 'variant': variant,
                        'traversal': traversal, 'status': 'lowered', 'platforms': list(exported.platforms),
                        'ir_file': f'tpu_lowered_ir/{name}',
                        'ir_sha256': hashlib.sha256((ir_dir / name).read_bytes()).hexdigest(),
                        'device_compilation_or_execution': False})
        write(out / 'tpu_lowering.json', lowering)
        summary['tpu_lowering_cases_passed'] = len(lowering)
        write(out / 'schedule_enumeration.json', schedule.enumeration_report(include_orders=True))
        write(out / 'example_metadata.json', [opt.describe((4097, 4097, 4097), (2048, 2048, 512),
              variant=variant) for variant in opt.VARIANTS])

        for variant in ('optimized', 'peeled_edges'):
            child(out, 'cli_' + variant, ['strassen_optimized.py', '--m', '17', '--n', '257',
                  '--k', '259', '--tile', '16', '256', '256', '--variant', variant, '--interpret-correctness'])
            record = json.loads((out / ('cli_' + variant + '.log')).read_text())
            if not record['correctness_pass'] or record['timing_reported']:
                raise RuntimeError('CLI qualification failed')
        child(out, 'cpu_ablation', ['tools/benchmark_strassen_optimized_v001.py',
              '--output-dir', str(out / 'cpu_ablation'), '--interpret-correctness'])
        smoke = json.loads((out / 'cpu_ablation/summary.json').read_text())
        rows = json.loads((out / 'cpu_ablation/case_results.json').read_text())
        events = [json.loads(line) for line in (out / 'cpu_ablation/results.jsonl').read_text().splitlines()]
        if not smoke['completed'] or len(rows) != 36 or any(
                row['status'] != 'ok' or row['eligible_for_speedup_claim'] or row['timing']['sample_count']
                for row in rows) or any(e.get('event') == 'sample' for e in events):
            raise RuntimeError('CPU ablation qualification failed')
        summary['cpu_ablation_scoped_cases_passed'] = len(rows)
        child(out, 'hardware_plan', ['tools/benchmark_strassen_optimized_v001.py',
              '--output-dir', str(out / 'hardware_plan'), '--plan-only'])
        summary['hardware_plan'] = '8 shapes × 9 arms × 2 scopes; plan only, not executed'
        summary['passed'] = True
    except Exception as error:
        summary['error'] = {'type': type(error).__name__, 'message': str(error),
                            'traceback': traceback.format_exc()}
    write(out / 'validation_summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
