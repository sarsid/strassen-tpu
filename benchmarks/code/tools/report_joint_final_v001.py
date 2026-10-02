"""Report six complete joint-tuning shapes from four verified allocations."""
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
    for name in ('cohort', 'prior-cohort', 'tall-cohort', 'k12288-cohort', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    sources = [args.prior_cohort] * 3 + [args.tall_cohort, args.k12288_cohort, args.cohort]
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
        for source in (args.tall_cohort, args.k12288_cohort, args.cohort):
            if (source / 'source' / relative).read_bytes() != old:
                raise ValueError('Continuation changed the tuning contract: ' + relative)
        hashes[relative] = hashlib.sha256(old).hexdigest()
    for name in ('campaign.json', 'shapes.json', 'distributions.json'):
        shutil.copyfile(args.prior_cohort / 'source/configs/joint_v001' / name, config / name)
    provenance = []
    for index, source in enumerate(sources, 1):
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
        timing_rule='Every paired comparison stays within its shape and original allocation. No cross-allocation timing pool.',
        excluded_partial_attempt=str(args.k12288_cohort / 'joint-06-screen-finished.json'))
    (args.output_dir / 'continuation_provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    note = ('# Completed across four allocations\n\n'
            'The first three shapes come from September 24; tall, K12288 and the '
            '16384 cube each come from a separate September 26 continuation. '
            'Failed or partial attempts remain archived and do not contribute timing '
            'samples to this report. Each comparison uses candidates and Native '
            'measured together on the same allocation. The original tuning contract '
            'is unchanged.\n\n')
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
