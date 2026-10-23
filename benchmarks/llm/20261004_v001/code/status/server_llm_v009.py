"""Add persistent queue status to the existing read-only evidence dashboard."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import server_llm_v008 as base

previous=base.snapshot
base.PAGE=Path(__file__).with_name('llm_dashboard_v007.html')
def snapshot(run):
    data=previous(run);state=base.read(run.parent/'queue_v001/state.json')
    if state:
        jobs=state.get('jobs',[])
        updated=state.get('updated_utc',state.get('created_utc'))
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds() if updated else 999999
        data['queue']=dict(status=state['status'],counts=dict(Counter(j['status'] for j in jobs)),next=next((j for j in jobs if j['status']=='queued'),None),updated_utc=updated,stale=age>120 and state['status'] not in ('completed','finished_with_failures'),reason=state.get('reason'))
    return data
base.snapshot=snapshot
if __name__=='__main__':base.main()
