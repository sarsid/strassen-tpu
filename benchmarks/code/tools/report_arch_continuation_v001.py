"""Report complete shape comparisons, including recovered acknowledgment loss."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_arch_continuation_v001 import PRIOR, RECOVERY, unchanged
from reconcile_arch_launch_v001 import verify as verify_recovery
from report_arch_study_v002 import build
from run_large_real_v002 import verify_phase
from run_region_cohort_v001 import write, sha


def make_view(cohort, view, partial=False):
    unchanged(cohort / 'source')
    cfg = view / 'source/configs/arch_v6e_168_v001'
    cfg.mkdir(parents=True, exist_ok=False)
    for name in ('campaign', 'shapes', 'distributions'):
        shutil.copyfile(PRIOR / f'source/configs/arch_v6e_168_v001/{name}.json', cfg / (name + '.json'))
    recovered, _ = verify_recovery(RECOVERY)
    provenance = []
    for index in range(168):
        source = PRIOR if index in list(range(27)) + [130] else cohort
        if partial and index not in list(range(27)) + [130]:
            continue
        for stage in ('screen', 'confirm'):
            phase = f'arch-{index+1:03d}-{stage}'
            if index == 26 and stage == 'confirm':
                run = recovered
                evidence = RECOVERY / 'reconciliation.json'
            else:
                run, _ = verify_phase(source, phase)
                evidence = source / (phase + '-finished.json')
            write(view / (phase + '-finished.json'), dict(status='completed', run=str(run),
                evidence=str(evidence), evidence_sha256=sha(evidence)))
            provenance.append(dict(phase=phase, run=str(run), evidence=str(evidence), sha256=sha(evidence)))
    write(view / 'provenance.json', dict(phases=provenance,
        timing_rule='Every comparison uses the original paired observations; no reruns, reselection or pooling across allocations.'))
    return provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    view = a.output_dir.parent / 'combined-input-view'
    provenance = make_view(a.cohort, view)
    code = build(view, a.output_dir)
    write(a.output_dir / 'continuation_provenance.json', dict(phases=provenance))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
