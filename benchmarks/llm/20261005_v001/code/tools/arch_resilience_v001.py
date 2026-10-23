"""Whole-comparison checkpoints for the architecture studies; no sample pooling."""
import json
import os
from pathlib import Path
from run_large_real_v002 import verify_phase
from run_region_cohort_v001 import read, sha, write
from reconcile_arch_launch_v001 import verify as verify_recovery

ROOT = Path(os.environ.get('STRASSEN_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
BASE = ROOT / 'runs/20260927-arch-v6e-168-v001'
RECOVERY = ROOT / 'runs/20260927-arch-recovery-diagnosis-v001'
RESULTS = {
    'v6e': 'results/v6e/168_shapes_joint_fp32_bf16_20260927_v001',
    'v5e': 'results/v5e/168_shapes_legacy_fp32_bf16_20260927_v001',
}


def units(hardware):
    if hardware == 'v6e':
        order = read(BASE / 'source/configs/arch_v6e_168_v001/campaign.json')['execution_shape_order']
        return [f'arch-{i+1:03d}' for i in order]
    if hardware == 'v5e':
        return [f'{dtype}-{i:02d}' for dtype in ('fp32', 'bf16') for i in range(1, 13)]
    raise ValueError('Unsupported architecture')


def check_contract(source):
    """The experiment implementation stays frozen while orchestration evolves."""
    old = BASE / 'source'
    paths = list((old / 'src').rglob('*.py'))
    for name in ('arch_v6e_168_v001', 'arch_v5e_168_fp32_v001', 'arch_v5e_168_bf16_v001'):
        paths.extend((old / 'configs' / name).glob('*.json'))
    for before in paths:
        relative = before.relative_to(old)
        if (source / relative).read_bytes() != before.read_bytes():
            raise ValueError('Measurement contract changed: ' + str(relative))


def collect(history, hardware):
    """Keep the earliest wholly completed screen/confirm pair for each unit."""
    selected = {}
    for entry in history['cohorts']:
        if entry['hardware'] != hardware:
            continue
        cohort = Path(entry['path'])
        check_contract(cohort / 'source')
        for unit in units(hardware):
            if unit in selected:
                continue
            phases = {}
            for part in ('screen', 'confirm'):
                phase = unit + '-' + part
                recovery = history.get('recoveries', {}).get(str(cohort) + ':' + phase)
                if recovery or (cohort == BASE and phase == 'arch-027-confirm'):
                    folder = Path(recovery) if recovery else RECOVERY
                    run, summary = verify_recovery(folder)
                    evidence = folder / 'reconciliation.json'
                else:
                    evidence = cohort / (phase + '-finished.json')
                    if not evidence.exists() or read(evidence)['status'] != 'completed':
                        break
                    run, summary = verify_phase(cohort, phase)
                environment = read(run / 'artifacts/environment.json')['identity']
                phases[part] = dict(phase=phase, run=str(run), cohort=str(cohort),
                    evidence=str(evidence), evidence_sha256=sha(evidence), identity=environment)
            if len(phases) == 2:
                if phases['screen']['identity'] != phases['confirm']['identity']:
                    raise ValueError('Cannot join screening and confirmation across device identities')
                selected[unit] = phases
    return selected


def missing(history, hardware):
    complete = collect(history, hardware)
    return [unit for unit in units(hardware) if unit not in complete]
