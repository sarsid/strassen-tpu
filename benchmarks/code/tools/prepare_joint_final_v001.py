"""Run only the final joint-tuning shape, keeping all earlier attempts intact."""
import json
import prepare_joint_v001 as original
from run_large_real_v002 import verify_phase

campaign = original.campaign
PRIOR = campaign.ROOT / 'runs/20260924-joint-v001'
TALL = campaign.ROOT / 'runs/20260926-joint-continuation-v001'
K12288 = campaign.ROOT / 'runs/20260926-joint-recovery-v001'


def build(args):
    prior_sources = [PRIOR] * 3 + [TALL, K12288]
    for index, source in enumerate(prior_sources, 1):
        for stage in ('screen', 'confirm'):
            verify_phase(source, f'joint-{index:02d}-{stage}')
    plan = original.build(args)
    keep = {'joint-smoke', 'joint-06-screen', 'joint-06-confirm', 'report', 'release'}
    plan['stages'] = [s for s in plan['stages'] if s['id'] in keep]
    for stage in plan['stages']:
        if stage['id'] == 'joint-06-screen':
            stage['requires'] = ['joint-smoke']
        if stage['id'] == 'report':
            stage['command'] = ['{analysis_python}', '{source}/tools/report_joint_final_v001.py',
                '--cohort', '{cohort}', '--prior-cohort', str(PRIOR), '--tall-cohort', str(TALL),
                '--k12288-cohort', str(K12288), '--output-dir', '{cohort}/operations/report/artifacts']
    plan.update(title='Final joint v6e shape: 16384 cube, FP32 and BF16 output',
                continuation_of=[str(PRIOR), str(TALL), str(K12288)],
                completed_shape_indices=[0, 1, 2, 3, 4], resumed_shape_indices=[5])
    plan['decisions'] += [
        'User requested the final shape after laptop sleep and allocation loss; keep the Mac awake, lid open and online.',
        'Verify all ten completed phase archives. Retain the incomplete cube attempt separately and rerun its entire screen and confirmation.',
        'Keep kernels, candidate menus, seeds, output contracts, error gates and confirmation rules unchanged.',
        'Combine complete per-shape comparisons from four cohorts; never pool timings across allocations.']
    return plan


campaign.build_plan = build
if __name__ == '__main__':
    campaign.main()
