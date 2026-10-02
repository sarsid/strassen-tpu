"""Combine verified per-shape cohorts without mixing allocation timing samples."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from report_joint_v001 import build
from run_large_real_v002 import verify_phase


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--prior-cohort', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    view = args.output_dir.parent / 'combined-input-view'
    config = view / 'source/configs/joint_v001'
    config.mkdir(parents=True, exist_ok=False)
    unchanged = [f'configs/joint_v001/{name}.json' for name in ('campaign', 'shapes', 'distributions')]
    unchanged += ['src/strassen_mm/' + name for name in (
        'kernels_joint_v001.py', 'kernels_fullk_v002.py', 'kernels_v6e_v001.py',
        'kernels_two_level_v001.py', 'kernels_v001.py', 'tuner_joint_v001.py', 'benchmark_joint_v001.py')]
    hashes = {}
    for relative in unchanged:
        old = (args.prior_cohort / 'source' / relative).read_bytes()
        new = (args.cohort / 'source' / relative).read_bytes()
        if old != new:
            raise ValueError('Continuation changed the tuning contract: ' + relative)
        hashes[relative] = hashlib.sha256(old).hexdigest()
    for name in ('campaign.json', 'shapes.json', 'distributions.json'):
        shutil.copyfile(args.prior_cohort / 'source/configs/joint_v001' / name, config / name)
    provenance = []
    for index in range(1, 7):
        source = args.prior_cohort if index <= 3 else args.cohort
        for stage in ('screen', 'confirm'):
            phase = f'joint-{index:02d}-{stage}'
            run, _ = verify_phase(source, phase)
            receipt = source / (phase + '-finished.json')
            shutil.copyfile(receipt, view / receipt.name)
            provenance.append(dict(phase=phase, cohort=str(source), run=str(run),
                                   receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
    if build(view, args.output_dir) != 0:
        raise RuntimeError('Combined report is incomplete')
    evidence = dict(unchanged_source_sha256=hashes, phases=provenance,
        timing_rule='Every paired comparison stays within its shape and original allocation. No cross-allocation timing pool.')
    (args.output_dir / 'continuation_provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    note = ('# Completed across two allocations\n\n'
            'The first three shapes come from the September 24 cohort; the remaining '
            'three come from this continuation. Each comparison uses candidates and Native '
            'measured together on the same allocation. Timing samples are never pooled '
            'across allocations. The original tuning contract is unchanged.\n\n')
    for name in ('RESULTS.md', 'TUNER_DESIGN.md'):
        path = args.output_dir / name
        path.write_text(note + path.read_text())
    subprocess.run([sys.executable, str(Path(__file__).with_name('explain_joint_v002.py')),
        '--results', str(args.output_dir / 'results.json'),
        '--decisions', str(args.output_dir / 'candidate_decisions.json'),
        '--output-dir', str(args.output_dir / 'explanations')], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
