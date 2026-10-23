"""Show active BF16 scope separately from preserved historical FP32 evidence."""
from collections import Counter
from pathlib import Path
import server_llm_v012 as previous

base = previous.base
base.PAGE = Path(__file__).with_name('llm_dashboard_v011.html')


def snapshot(run):
    data = previous.snapshot(run)
    state = base.read(run.parent / 'queue_v001/state.json')
    dtypes = state.get('active_output_dtypes', ['bfloat16', 'float32'])
    planned = [r for r in base.lines(run.parent / 'planned_grid_v001/planned_entries.jsonl') if r['output_dtype'] in dtypes]
    ids = {r['entry_id'] for r in planned}
    all_rows = [r for r in base.lines(run.parent / 'completed_entries.jsonl') if r.get('status') == 'completed' and r.get('measurement')]
    active = [r for r in all_rows if r['entry_id'] in ids]
    data['scope'] = dict(output_dtypes=dtypes, historical_comparisons=len(all_rows)-len(active),
                         original_expected=700, revision=state.get('scope_revision'))
    data['main'] = dict(measured=len(active), expected=len(planned))
    expected, measured = Counter(r['model'] for r in planned), Counter(r['model'] for r in active)
    for model in data['models']:
        model.update(expected=expected[model['id']], measured=measured[model['id']])
    for stage in data['stages']:
        if stage['id'] == 'main-prefill':stage['detail'] = f'{len(active)} / {len(planned)} active comparisons'
    progress = data['progress']
    progress['tracks'] = [t for t in progress['tracks'] if t['output_dtype'] in dtypes]
    progress['observed_rounds'] = sum(t['observed_rounds'] for t in progress['tracks'])
    progress['expected_rounds'] = sum(t['expected_rounds'] for t in progress['tracks'])
    progress['recent_measurements'] = [e for e in progress['recent_measurements'] if e['output_dtype'] in dtypes]
    # Do not let a full tail of old FP32 samples hide recovered BF16 samples.
    watched = run.parent / data['run_id']
    remote = base.read(watched / 'latest.json')
    session = data.get('queue', {}).get('session')
    events = previous.history.read(watched, remote, progress['job_id'] if session else None)
    progress['recent_measurements'] = [{k:e[k] for k in ('utc','algorithm','output_dtype','round','elapsed_ms','resident_layer_sum_ms','host_preparation_seconds','host_cache') if k in e}
                                      for e in reversed([e for e in events if e['kind']=='confirmation' and e.get('output_dtype') in dtypes][-20:])]
    return data


base.snapshot = snapshot
if __name__ == '__main__':base.main()
