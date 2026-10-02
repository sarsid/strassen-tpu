"""Continue only unconfirmed shapes on a fresh, explicitly separate allocation."""
import json
from pathlib import Path
from run_region_cohort_v001 import verify_frozen,sha
import prepare_v6e_suite_v001 as prior
campaign=prior.campaign

def build(args):
    original=campaign.ROOT/'runs/20260923-v6e-suite-v001'
    frozen=verify_frozen(original)
    if args.endpoint==frozen['allocation_id']:raise ValueError('This recovery requires the explicitly recorded new allocation')
    preserved=[n for n in frozen['source_sha256']if n.startswith('src/')or n.startswith('configs/v6e_suite_v001/')]
    for name in preserved:
        if sha(campaign.ROOT/name)!=frozen['source_sha256'][name]:raise ValueError('Executed scientific source changed: '+name)
    completed=[]
    for i in range(1,10):
        for suffix in ('screen','confirm'):
            stage=f'suite-{i:02d}-{suffix}';receipt=json.loads((original/(stage+'-finished.json')).read_text())
            if receipt['status']!='completed':raise ValueError('Original prerequisite incomplete: '+stage)
            run=Path(receipt['run'])
            if sha(run/'completion.json')!=receipt['completion_sha256']:raise ValueError('Receipt hash mismatch')
            if not json.loads((run/'artifacts/summary.json').read_text())['completed']:raise ValueError('Phase incomplete')
            completed.append(stage)
    plan=prior.build(args)
    stages=[s for s in plan['stages']if s['id']=='suite-smoke'or any(s['id'].startswith(f'suite-{i:02d}-')for i in (10,11,12))]
    next(s for s in stages if s['id']=='suite-10-screen')['requires']=['suite-smoke']
    stages += [dict(id='combined-report',label='Verify and report all 168 shapes with explicit allocation provenance',operation='command',kind='analysis',timeout_seconds=3600,continue_on_failure=True,
        command=['{analysis_python}','{source}/tools/report_v6e_suite_recovery_v001.py','--original',str(original),'--cohort','{cohort}',
                 '--output-dir','{cohort}/operations/combined-report/artifacts']),
        next(s for s in plan['stages']if s['id']=='release')]
    plan.update(title='v6e recovery: remaining 42 of 168 shapes',stages=stages,
        recovery_of=dict(cohort=str(original),allocation_id=frozen['allocation_id'],source_archive_sha256=frozen['archive_sha256'],
                         confirmed_shapes=126,completed_stages=completed,unavailable_partial_stage='suite-10-screen'),
        decisions=plan['decisions']+[
            'Previous allocation absent after laptop disconnection; exact release reason unknown. Preserve sealed 126 shapes.',
            'Repeat only the unavailable batch-10 screen and subsequent unstarted phases: 42 shapes. Fresh qualification and paired baselines on new allocation.',
            'All kernels and scientific configurations byte-identical to original source. No pooled screen/confirmation measurements across allocations.',
            'Combined report joins distinct completed shape rows and labels the allocation of each; incomplete old batch-10 measurements remain excluded.'])
    return plan

campaign.build_plan=build
if __name__=='__main__':campaign.main()
