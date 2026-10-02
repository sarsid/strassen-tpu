"""Continue untouched v6e measurements after reconciling a lost acknowledgment."""
import json
from pathlib import Path
import prepare_arch_v6e_v002 as original
from run_large_real_v002 import verify_phase
from reconcile_arch_launch_v001 import verify as verify_recovery

campaign = original.campaign
PRIOR = campaign.ROOT / 'runs/20260927-arch-v6e-168-v001'
RECOVERY = campaign.ROOT / 'runs/20260927-arch-recovery-diagnosis-v001'


def unchanged(source):
    old = PRIOR / 'source'
    names = [str(p.relative_to(old)) for p in (old / 'src').rglob('*.py')]
    names += [f'configs/arch_v6e_168_v001/{n}.json' for n in ('campaign', 'shapes', 'distributions')]
    for name in names:
        if (source / name).read_bytes() != (old / name).read_bytes():
            raise ValueError('Continuation changed measurement code or policy: ' + name)
    return names


def build(args):
    prior = json.loads((PRIOR / 'plan.json').read_text())
    if args.endpoint != prior['allocation_id'] or args.session != prior['session'] or args.identity != prior['remote_identity']:
        raise ValueError('This continuation is only for the same existing allocation')
    unchanged(campaign.ROOT)
    recovered, _ = verify_recovery(RECOVERY)
    completed = list(range(27)) + [130]
    for index in completed:
        verify_phase(PRIOR, f'arch-{index+1:03d}-screen')
        if index != 26:
            verify_phase(PRIOR, f'arch-{index+1:03d}-confirm')
    plan = original.build(args)
    skipped = {'arch-smoke', 'export-arch-smoke'}
    for index in completed:
        for stage in ('screen', 'confirm'):
            phase = f'arch-{index+1:03d}-{stage}'
            skipped.update((phase, 'export-' + phase))
    plan['stages'] = [s for s in plan['stages'] if s['id'] not in skipped]
    plan['stages'][0]['requires'] = []
    for stage in plan['stages']:
        if stage['id'] == 'report':
            stage['command'] = ['{analysis_python}', '{source}/tools/report_arch_continuation_v001.py',
                '--cohort', '{cohort}', '--output-dir', '{cohort}/operations/report/artifacts']
    plan.update(transport='tools/run_phase_v006.py', continuation_of=str(PRIOR),
        recovered_confirmation=str(RECOVERY), completed_shape_indices=completed,
        title='v6e 168-shape study: 28 saved; continuing the remaining 140 shapes')
    plan['decisions'] += [
        'Same allocation; all measurement source, kernels, candidates, seeds and precision contracts are unchanged.',
        'Recover the successful arch-027-confirm archive; preserve the original failed local launch receipt.',
        'Read existing worker state after an unacknowledged launch; never issue a duplicate launch.',
        'Final report verifies and combines all original, recovered and continuation phases.']
    return plan


campaign.build_plan = build
if __name__ == '__main__':
    campaign.main()
