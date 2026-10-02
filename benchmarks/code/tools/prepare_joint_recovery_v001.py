"""Recover only the final two joint-tuning shapes after a lost allocation."""
import json
import prepare_joint_v001 as original
from run_large_real_v002 import verify_phase

campaign = original.campaign
PRIOR = campaign.ROOT / 'runs/20260924-joint-v001'
TALL = campaign.ROOT / 'runs/20260926-joint-continuation-v001'
ABSENCE = campaign.ROOT / 'runs/20260926-joint-recovery-runtime-v001/prior-allocation-absence.json'


def build(args):
    proof = json.loads(ABSENCE.read_text())
    assert proof['result']['assignments'] == []
    assert proof['prior_endpoint'] == json.loads((TALL / 'plan.json').read_text())['allocation_id']
    for index in range(1, 5):
        source = PRIOR if index <= 3 else TALL
        for stage in ('screen', 'confirm'):
            verify_phase(source, f'joint-{index:02d}-{stage}')
    plan = original.build(args)
    keep = {'joint-smoke', 'report', 'release'}
    keep.update(f'joint-{i:02d}-{stage}' for i in (5, 6) for stage in ('screen', 'confirm'))
    plan['stages'] = [s for s in plan['stages'] if s['id'] in keep]
    for stage in plan['stages']:
        if stage['id'] == 'joint-05-screen':
            stage['requires'] = ['joint-smoke']
        if stage['id'] == 'report':
            stage['command'] = ['{analysis_python}', '{source}/tools/report_joint_recovery_v001.py',
                '--cohort', '{cohort}', '--prior-cohort', str(PRIOR), '--tall-cohort', str(TALL),
                '--output-dir', '{cohort}/operations/report/artifacts']
    plan.update(title='Joint v6e tuning recovery: K12288 and 16384 cube',
                continuation_of=[str(PRIOR), str(TALL)], prior_allocation_absence=str(ABSENCE),
                completed_shape_indices=[0, 1, 2, 3], resumed_shape_indices=[4, 5])
    plan['decisions'] += [
        'Fresh allocation query verified the prior runtime absent after an unacknowledged launch; preserve that failed attempt.',
        'Resume only shapes 5 and 6; verify all eight completed screen/confirmation archives before freezing.',
        'Rerun compiled smoke, retaining all original kernels, candidate menus, seeds, errors and confirmation rules.',
        'Combine whole per-shape comparisons from three cohorts, never timing samples from different allocations.']
    return plan


campaign.build_plan = build
if __name__ == '__main__':
    campaign.main()
