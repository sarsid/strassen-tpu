"""Freeze a Mistral-only v5e test of MM-to-model latency translation."""
import prepare_llm_tradeoff_v002 as campaign

original=campaign.build_plan


def build(args):
    if args.hardware!='v5e':raise ValueError('This comparison is v5e only')
    args.models=['mistral']
    plan=original(args)
    by_id={s['id']:s for s in plan['stages']}
    # Establish resident feasibility and latency before the longer held-out scoring.
    plan['stages']=[by_id[k] for k in ('MODEL-tools','mistral-inputs','mistral-tune',
        'mistral-resident','mistral-quality','report','release')]
    report=by_id['report']
    report['command']=[x.replace('report_llm_tradeoff_v001.py','report_llm_tradeoff_v002.py') for x in report['command']]
    plan['title']='Mistral-7B: v5e MM-to-model speed and accuracy'
    plan['decisions']=[
        'User prioritized fully resident Mistral-7B on v5e; Qwen deferred until more v5e memory is available.',
        'Exact official BF16 checkpoint, batch 1, 2048-token prompts; no quantization or streamed fallback.',
        'Primary comparison: tuned composed Native versus composed Strassen 1/2, sharing all surrounding operations.',
        'Secondary practical comparison: fastest measured Native control, including whole-layer JIT.',
        'Fresh per-projection tuning and three-window MM confirmation on the same allocation as the model.',
        'Tuning windows 0..3 and held-out quality windows 16..31; 32752 actual next-token targets.',
        'Resident timing includes embedding, all decoder layers, final normalization/head, dispatch and synchronization.',
        'Checkpoint reads, layout conversion, weight transfers and compilation occur before resident timing.',
        'Unchanged official Native qualification and model-quality gates; a failed gate blocks a speed-quality claim.',
        'All arms use two warmups and nine randomized paired rounds over three held-out prompts.',
        'Report isolated MM, full resident prompt and full scoring timings, paired intervals and numerical/prediction errors.',
        'A favorable result is a hypothesis to test, not a selection condition; preserve losses, failures and controls.',
        'One owned allocation; archive results and release automatically. No v6e optimization in this experiment.']
    return plan


campaign.build_plan=build
if __name__=='__main__':campaign.main()
