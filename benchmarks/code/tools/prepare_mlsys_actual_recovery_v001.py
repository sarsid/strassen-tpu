"""Freeze an actual-model-only continuation; no launch, network call or commit.

The owning handoff controller must prove the original final phase idle and
retire its controller before launching this plan on the same allocation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tarfile

from run_region_cohort_v001 import read, sha, utc, verify_frozen, write

ROOT = Path(os.environ.get('STRASSEN_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
CAMPAIGN = 'configs/mlsys_real_models_v001/campaign.json'
PREPARER = 'strassen_mm.benchmark_mlsys_prepare_v002'


def build_plan(cohort: Path, original_plan: dict, original_link: dict | None = None) -> dict:
    """Pure plan builder: retain original model protocol and logical machine."""
    cohort = Path(cohort)
    required = ('cohort_id', 'session', 'allocation_id', 'remote_identity',
                'controller_python', 'analysis_python', 'private_root')
    if any(not isinstance(original_plan.get(k), str) or not original_plan[k] for k in required):
        raise ValueError('Original plan lacks allocation, identity, interpreter or private-root fields')
    if cohort.name == original_plan['cohort_id']:
        raise ValueError('Recovery requires a new cohort')
    private = '/content/Strassen_MM_Focus/.runtime_private/' + original_plan['cohort_id']
    if original_plan['private_root'] != private or not original_plan['allocation_id'].startswith('tpu-v5e1-'):
        raise ValueError('Expected original private root and single-chip v5e allocation')
    entries = original_plan['stages']; stages_by_id = {s['id']: s for s in entries}
    if len(stages_by_id) != len(entries):
        raise ValueError('Duplicate original stage IDs')
    link = deepcopy(original_link or {'cohort_id': original_plan['cohort_id']})
    original = Path(link.get('cohort_absolute_path', ROOT / 'runs' / original_plan['cohort_id']))
    if not original.is_absolute() or original.name != original_plan['cohort_id']:
        raise ValueError('Canonical absolute original cohort path required')
    link['cohort_absolute_path'] = str(original)
    venv = private + '/cpu-tools-v002'
    tools = deepcopy(stages_by_id['MODEL-tools'])
    tools.update(id='MODEL-tools-v002', label='Prepare repaired isolated CPU model tools',
                 module=PREPARER, continue_on_failure=True,
                 runner_args=['--action', 'tools', '--private-root', private, '--venv-dir', venv])
    stages = [tools]
    for model in ('qwen', 'mistral', 'gemma'):
        prep = deepcopy(stages_by_id[model + '-inputs'])
        prep.update(module=PREPARER, requires=['MODEL-tools-v002'], continue_on_failure=True,
                    runner_args=['--action', 'inputs', '--model-key', model,
                                 '--private-root', private, '--venv-dir', venv])
        actual = deepcopy(stages_by_id[model + '-actual'])
        if actual['module'] != 'strassen_mm.benchmark_mlsys_models_v001':
            raise ValueError('Unexpected actual-model benchmark module')
        actual.update(requires=[prep['id']], continue_on_failure=True,
                      input_stages=[dict(stage=prep['id'], flag='--model-input', file='prepared/input_bundle.json')])
        stages.extend([prep, actual])
    if any(s.get('campaign') != CAMPAIGN or s.get('operation') != 'phase' for s in stages):
        raise ValueError('Recovery must retain the original real-model campaign')
    stages[:0] = [
        dict(id='original-final-audit', label='Audit the original final 168-shape confirmation',
             operation='command', kind='analysis', timeout_seconds=1200,
             command=['{analysis_python}', '{source}/tools/audit_mlsys_stage_v001.py',
                      '--cohort', str(original), '--stage', 'MAIN-12-confirm']),
        dict(id='original-report', label='Report the original completed synthetic-shape study',
             operation='command', kind='analysis', timeout_seconds=1200,
             requires=['original-final-audit'],
             command=['{analysis_python}', '{source}/tools/report_mlsys_campaign_v001.py',
                      '--cohort', str(original), '--output-dir', '{cohort}/operations/original-report/artifacts']),
    ]
    tools['requires'] = ['original-final-audit', 'original-report']
    stages.extend(deepcopy(stages_by_id[name]) for name in ('release', 'report'))
    plan = {k: original_plan[k] for k in required}
    plan.update(schema_version=1, cohort_id=cohort.name, stages=stages,
                recovery_of=link,
                decisions=[
                    'Actual-model continuation after the original 168-shape study; no synthetic measurements are repeated or pooled.',
                    'Audit the original final confirmation and report its sealed evidence locally before any remote model preparation.',
                    'Same logical v5e allocation and frozen identity; no allocation replacement, environment reset or automatic retry.',
                    'Original real-model configs, checkpoint revisions, corpus, numerical gates and confirmation rules remain unchanged.',
                    'Reuse the approved private-root credential without copying it into code, artifacts or the source archive.',
                    'Fresh cpu-tools-v002 uses ensurepip-free bootstrap; the failed cpu-tools target remains intact.',
                    'Preparation is excluded from measured-case progress. Each inputs stage requires successful repaired tools.',
                    'Known-idle model failures remain explicit and allow later models; uncertain remote execution stops advancement.',
                    'Native qualification gates full-model claims independently of real-projection MM eligibility.',
                    'Release the exact allocation after all available actual-model stages, then report sealed evidence.',
                ])
    return plan


def freeze(args):
    original = args.original_cohort.resolve(); cohort = args.cohort.resolve()
    if not original.is_relative_to(ROOT / 'runs') or not cohort.is_relative_to(ROOT / 'runs') or cohort == original:
        raise ValueError('Distinct project run directories required')
    if cohort.exists():
        raise FileExistsError('Recovery cohort already exists')
    frozen_original = verify_frozen(original); original_plan = read(original / 'plan.json')
    if original_plan != read(original / 'source/plan.json'):
        raise ValueError('Original plan differs from frozen source')
    for key in ('cohort_id', 'session', 'allocation_id'):
        if original_plan[key] != frozen_original[key]:
            raise ValueError('Original plan identity differs: ' + key)
    link = dict(cohort_id=original.name, cohort_path=str(original.relative_to(ROOT)), cohort_absolute_path=str(original),
                baseline_commit=frozen_original['baseline_commit'], archive_sha256=frozen_original['archive_sha256'],
                frozen_sha256=sha(original / 'frozen.json'), plan_sha256=sha(original / 'plan.json'))
    plan = build_plan(cohort, original_plan, link)
    for field in ('controller_python', 'analysis_python'):
        if getattr(args, field, None):
            plan[field] = str(Path(getattr(args, field)).resolve())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cohort.mkdir(parents=True, exist_ok=False); source = cohort / 'source'; source.mkdir()
    with (cohort / 'baseline-source.tar').open('xb') as out:
        subprocess.run(['git', 'archive', head, '--', 'src', 'tools', 'runtime', 'configs', 'status', 'plans', 'tests'],
                       cwd=ROOT, stdout=out, check=True)
    with tarfile.open(cohort / 'baseline-source.tar') as packed:
        packed.extractall(source, filter='data')
    for required in ('src/strassen_mm/benchmark_mlsys_prepare_v002.py', 'tools/install_mlsys_model_tools_v001.py',
                     'tools/prepare_mlsys_actual_recovery_v001.py', 'tools/run_mlsys_unattended_v001.py'):
        if not (source / required).is_file():
            raise ValueError('Required recovery source is not committed: ' + required)
    preserved = [name for name in frozen_original['source_sha256']
                 if name.startswith('configs/mlsys_real_models_v001/')]
    preserved += ['src/strassen_mm/benchmark_mlsys_models_v001.py', 'tools/prepare_mlsys_models_v001.py',
                  'tools/run_mlsys_unattended_v001.py', 'tools/audit_mlsys_stage_v001.py',
                  'tools/audit_mlsys_shapes_v001.py', 'tools/report_mlsys_campaign_v001.py']
    for name in preserved:
        if sha(source / name) != frozen_original['source_sha256'][name]:
            raise ValueError('Original actual-model protocol/source changed: ' + name)
    write(cohort / 'plan.json', plan); write(source / 'plan.json', plan)
    write(cohort / 'cohort.json', dict(cohort_id=cohort.name, session=plan['session'], allocation_id=plan['allocation_id'],
          source_commit=head, created_utc=utc(), recovery_of=link))
    with tarfile.open(cohort / 'source.tar', 'x') as packed:
        for child in sorted(source.iterdir()):
            packed.add(child, arcname=child.name)
    write(cohort / 'frozen.json', dict(cohort_id=cohort.name, session=plan['session'], allocation_id=plan['allocation_id'],
          baseline_commit=head, archive_sha256=sha(cohort / 'source.tar'), recovery_of=link,
          source_sha256={str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}))
    write(cohort / 'progress.json', dict(schema_version=1, campaign_id=cohort.name, title='Actual-model recovery on the original v5e',
          cohort_path=str(cohort.relative_to(ROOT)), state='preparing', updated_utc=utc(), heartbeat_utc=utc(),
          detail='Frozen continuation; launch requires verified original-worker quiescence and controller handoff.',
          stages=[], selected_results=[]))
    verify_frozen(cohort)
    print(json.dumps(dict(status='frozen', cohort=str(cohort), stage_count=len(plan['stages']),
                         allocation_id=plan['allocation_id'], source_archive_sha256=sha(cohort / 'source.tar'))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze'])
    parser.add_argument('--original-cohort', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--controller-python'); parser.add_argument('--analysis-python')
    freeze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
