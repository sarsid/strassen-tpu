"""Resume only the three unrun joint-tuning shapes in a fresh immutable cohort."""
import json
from pathlib import Path
import prepare_joint_v001 as original

campaign = original.campaign
PRIOR = campaign.ROOT / 'runs/20260924-joint-v001'


def build(args):
    closeout = json.loads((PRIOR / 'user-closeout-v001.json').read_text())
    assert closeout['allocation_released'] is True
    assert set(closeout['missing_shapes']) == {'tall', 'k12288', 'large_cube'}
    plan = original.build(args)
    keep = {'joint-smoke', 'report', 'release'}
    keep.update(f'joint-{i:02d}-{stage}' for i in (4, 5, 6) for stage in ('screen', 'confirm'))
    plan['stages'] = [s for s in plan['stages'] if s['id'] in keep]
    for stage in plan['stages']:
        if stage['id'] == 'joint-04-screen':
            stage['requires'] = ['joint-smoke']
        if stage['id'] == 'report':
            stage['command'] = ['{analysis_python}', '{source}/tools/report_joint_continuation_v001.py',
                '--cohort', '{cohort}', '--prior-cohort', str(PRIOR),
                '--output-dir', '{cohort}/operations/report/artifacts']
    plan.update(title='Joint v6e tuning continuation: tall, K12288 and 16384 cube',
                continuation_of=str(PRIOR), completed_shape_indices=[0, 1, 2], resumed_shape_indices=[3, 4, 5])
    plan['decisions'] += [
        'Resume only the three unrun shapes after the user-requested quota shutdown; do not modify the earlier cohort.',
        'Rerun compiled smoke on the new allocation before performance. Keep kernels, candidate menus, seeds and error gates unchanged.',
        'The combined report takes complete per-shape comparisons from their own allocation; never pool paired timings across allocations.']
    return plan


campaign.build_plan = build
if __name__ == '__main__':
    campaign.main()
