"""Add persistent queue status to the existing read-only evidence dashboard."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import server_llm_v008 as base

previous=base.snapshot
base.PAGE=Path(__file__).with_name('llm_dashboard_v009.html')
def snapshot(run):
    data=previous(run);state=base.read(run.parent/'queue_v001/state.json')
    if state:
        jobs=state.get('jobs',[])
        updated=state.get('updated_utc',state.get('created_utc'))
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds() if updated else 999999
        data['queue']=dict(status=state['status'],counts=dict(Counter(j['status'] for j in jobs)),next=next((j for j in jobs if j['status'] in ('reserved','queued')),None),updated_utc=updated,stale=age>120 and state['status'] not in ('completed','finished_with_failures','paused','blocked_shared_setup'),reason=state.get('reason'))
        order=state.get('model_order',[])
        if order:data['models'].sort(key=lambda m:order.index(m['id']) if m['id'] in order else len(order))
        data['queue']['model_order']=order
        latest=base.read(run.parent/data['run_id']/'latest.json')
        remote=latest.get('status.json',{});data['queue']['session']=bool(state.get('active_session',{}).get('legacy') is False)
        data['queue']['current_job']=remote.get('current_job')
        events=[]
        for path,text in latest.get('tails',{}).items():
            if not path.endswith('events.jsonl'):continue
            for line in text.splitlines():
                try:events.append(__import__('json').loads(line))
                except ValueError:pass
        for event in events[-3:]:
            kind=event.get('kind');detail=None
            if kind=='confirmation':detail=f"{event['algorithm']} · {event['output_dtype']} · round {event['round']+1}/15 · forward {event['elapsed_ms']:.1f} ms · host preparation {event['host_preparation_seconds']:.2f} s"
            elif kind=='resident_layer':detail=f"{event['algorithm']} · {event['output_dtype']} · layer {event['layer']+1} · resident median {event['median_ms']:.3f} ms"
            elif kind=='cache_pilot':detail='Cached/original full-model equality: '+str(event.get('bitwise_full_model_equal'))
            if detail:data['events'].insert(0,dict(time=event.get('utc'),title='Latest measurement',detail=detail,tone='active'))
    return data
base.snapshot=snapshot
if __name__=='__main__':base.main()
