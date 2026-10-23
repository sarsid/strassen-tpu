"""Fine-grained, read-only dashboard for legacy workloads and model sessions."""
from datetime import datetime, timezone
from pathlib import Path
import server_llm_v011 as previous
from llm_progress_v001 import EventHistory, summarize

base = previous.base
base.PAGE = Path(__file__).with_name('llm_dashboard_v010.html')
history = EventHistory()


def snapshot(run):
    data = previous.snapshot(run)
    watched = run.parent / data['run_id']
    latest = base.read(watched / 'latest.json')
    remote = latest.get('status.json', {})
    state = base.read(run.parent / 'queue_v001/state.json')
    session = state.get('active_session', {}).get('legacy') is False
    job = remote.get('current_job') or base.read(watched / 'job.json')
    # Model-session job.json contains a job list; never borrow another case.
    if 'job_id' not in job:
        job = {}
    model = next((m for m in data['models'] if m['id'] == job.get('model')), {})
    events = history.read(watched, latest, job.get('job_id') if session else None)
    data['progress'] = summarize(events, job, model, remote, session=session)
    data['generated_utc'] = datetime.now(timezone.utc).isoformat()
    data['snapshot_utc'] = datetime.fromtimestamp((watched / 'latest.json').stat().st_mtime, timezone.utc).isoformat() if (watched / 'latest.json').exists() else None
    return data


base.snapshot = snapshot
if __name__ == '__main__':
    base.main()
