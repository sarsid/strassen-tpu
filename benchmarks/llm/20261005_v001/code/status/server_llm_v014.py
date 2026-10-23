"""Show the active model plan, including pending architecture qualification."""
from collections import Counter
from pathlib import Path
import server_llm_v013 as previous

base = previous.base
base.PAGE = Path(__file__).with_name('llm_dashboard_v012.html')


def snapshot(run):
    data = previous.snapshot(run)
    state = base.read(run.parent / 'queue_v001/state.json')
    planned = base.lines(run.parent / state['active_plan'])
    ids = {r['entry_id'] for r in planned}
    all_rows = [r for r in base.lines(run.parent / 'completed_entries.jsonl')
                if r.get('status') == 'completed' and r.get('measurement')]
    active = [r for r in all_rows if r['entry_id'] in ids]
    data['main'] = dict(measured=len(active), expected=len(planned))
    data['scope'].update(historical_comparisons=len(all_rows)-len(active),
                         revision=state['scope_revision'])
    models = {m['id']: m for m in data['models']}
    if state.get('additional_model_metadata'):
        for item in base.read(run.parent / state['additional_model_metadata']):
            config = item['config'].get('text_config', item['config'])
            models[item['id']] = dict(id=item['id'], name='Gemma 4-31B', family='Gemma 4',
                hidden=config['hidden_size'], layers=config['num_hidden_layers'],
                metadata_ready=True, pilot=False, revision=item['revision'])
    expected, measured = Counter(r['model'] for r in planned), Counter(r['model'] for r in active)
    data['models'] = [models[ident] for ident in state['model_order']]
    for model in data['models']:
        model.update(expected=expected[model['id']], measured=measured[model['id']],
            awaiting_qualification=any(j['model'] == model['id'] and
                j.get('qualification_status') == 'awaiting_qualification' for j in state['jobs']))
    for stage in data['stages']:
        if stage['id'] == 'main-prefill':
            stage['detail'] = f'{len(active)} / {len(planned)} active comparisons'
    data['queue'].update(total_workloads=len(state['jobs']),
        next=next((j for j in state['jobs'] if j['status'] in
                  ('reserved', 'queued', 'awaiting_qualification')), None),
        deferred_workloads=len(state.get('deferred_jobs', [])))
    if state['status'] == 'blocked_model_qualification':data['queue']['stale'] = False
    return data


base.snapshot = snapshot
if __name__ == '__main__':base.main()
