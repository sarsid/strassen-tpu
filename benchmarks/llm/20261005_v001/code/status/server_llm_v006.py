"""Evidence-driven, read-only LLM dashboard. No TPU or experiment control."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

PAGE = Path(__file__).with_name('llm_dashboard_v005.html')

def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default

def lines(path):
    result = []
    try:
        for line in path.read_text().splitlines():
            try: result.append(json.loads(line))
            except ValueError: pass
    except OSError: pass
    return result

def snapshot(run):
    campaign = run.parent
    pointer=read(campaign/'active-run.json')
    watched=campaign/pointer.get('run_name',run.name)
    if watched.parent!=campaign or not watched.is_dir():watched=run
    latest = read(watched/'latest.json')
    completion = read(watched/'completion.json')
    remote = completion.get('remote_summary') or latest.get('status.json', {})
    artifacts = run/'remote'/run.name/'artifacts'
    kernels = read(artifacts/'kernel-qualification/summary.json')
    semantics = read(artifacts/'model-semantics/summary.json')
    pilot = read(artifacts/'real-qwen8-pilot/summary.json')
    profiles = read(artifacts/'real-qwen8-pilot/profiles.json')
    meta = read(campaign/'inputs_v002/models.json', [])
    entries = lines(campaign/'planned_grid_v001/planned_entries.jsonl')
    measured = [r for r in lines(campaign/'completed_entries.jsonl') if r.get('status') == 'completed' and r.get('measurement')]
    counts = Counter(r.get('model') for r in measured)
    expected = Counter(r.get('model') for r in entries)
    stage_rows = remote.get('stages', [])
    stage_codes = {s['name']:s.get('returncode') for s in stage_rows}
    if kernels.get('failed')==0:stage_codes['kernel-qualification']=0
    if semantics.get('failed')==0:stage_codes['model-semantics']=0
    if pilot.get('status')=='completed':stage_codes['real-qwen8-pilot']=0
    terminal = completion.get('status') in ('completed','failed','controller_failed','partial_or_failed','allocation_failed')
    modified = (watched/'latest.json').stat().st_mtime if (watched/'latest.json').exists() else None
    age = max(0,datetime.now(timezone.utc).timestamp()-modified) if modified else None
    stale = not terminal and (age is None or age>120)
    released = any(r.get('kind')=='allocation_released' and r.get('verified_absent') is True for r in lines(watched/'control/release.log'))
    state = completion.get('status') or remote.get('status','preparing')
    stages = []
    def phase(key,title,detail,done=False):
        code = stage_codes.get(key)
        state = 'done' if done or code==0 else 'failed' if code is not None else 'active' if remote.get('active')==key and not terminal else 'pending'
        stages.append(dict(id=key,title=title,detail=detail,state=state))
    phase('inputs','Model inputs',str(sum(m.get('status')=='metadata_ready' for m in meta))+' of '+str(len(meta))+' revisions pinned',bool(meta) and all(m.get('status')=='metadata_ready' for m in meta))
    phase('kernel-qualification','Kernel checks',f"{kernels.get('passed','—')} / {kernels.get('cases','—')} passed" if kernels else 'DEFAULT precision and fusion')
    phase('model-semantics','Model semantics',f"{semantics.get('passed','—')} small-fixture checks passed" if semantics else 'Official reference checks')
    phase('real-qwen8-pilot','Real-weight pilot','Qwen3-8B · one layer' if pilot else 'Layer timing and error')
    main_code=stage_codes.pop('main-prefill',None)
    phase('main-prefill','Full-model study',str(len(measured))+' / '+str(len(entries))+' comparisons',bool(entries) and len(measured)==len(entries))
    phase('heldout-quality','Held-out quality','Perplexity · KL · task accuracy')
    journal = lines(run.parents[3]/'status/work_progress_v001.jsonl')
    activity = next((r for r in reversed(journal) if r.get('kind')=='activity' and r.get('id')=='current'), {})
    events = []
    if activity:
        events.append(dict(time=activity.get('utc'),title=activity.get('title','Current work'),detail=activity.get('detail',''),tone='active' if activity.get('state')=='working' else 'neutral'))
    if terminal:
        events.append(dict(time=completion.get('finished_utc'),title='TPU released' if released else 'Run finished',detail='Result archive downloaded and verified.' if completion.get('result_sha256') else 'See run details for the saved outcome.',tone='done' if released else 'neutral'))
    if pilot:
        events.append(dict(time=remote.get('finished_utc'),title='Real-weight pilot complete',detail='Five algorithms compared across both store tracks. Full-model evaluation remains pending.',tone='done'))
    for key,summary,label in [('model-semantics',semantics,'Model semantics passed'),('kernel-qualification',kernels,'Kernel checks passed')]:
        if summary:
            events.append(dict(time=None,title=label if summary.get('failed')==0 else label.replace('passed','needs attention'),detail=f"{summary.get('passed',0)} passed · {summary.get('failed',0)} failed"+(' · small official-model fixtures' if key=='model-semantics' else ' · independent arithmetic reference'),tone='done' if summary.get('failed')==0 else 'failed'))
    old = read(campaign/'pilot_20261003_v001/remote/pilot_20261003_v001/artifacts/kernel-qualification/summary.json')
    if old:
        events.append(dict(time=None,title='Earlier reference mismatch retained',detail=f"{old.get('failed',0)} normalization checks failed against the original reference. The revised reference models DEFAULT operand rounding; rerun outcomes appear above.",tone='warning'))
    if not terminal and remote.get('active'):
        events.insert(0,dict(time=None,title=remote['active'].replace('-',' ').capitalize(),detail='Latest reported remote stage.'+(' Snapshot is stale; running state is unverified.' if stale else ''),tone='warning' if stale else 'active'))
    raw = '\n\n'.join(k+'\n'+v for k,v in latest.get('tails',{}).items() if v.strip())
    if not raw:
        p=artifacts/'real-qwen8-pilot/events.jsonl'
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-16000));raw=f.read().decode('utf-8','replace')
    models=[]
    for m in meta:
        config=m.get('config',{});text=config.get('text_config',config)
        models.append(dict(id=m['id'],name=m['repo_id'].split('/')[-1].replace('Mistral-Small-24B-Base-2501','Mistral Small 24B').replace('gemma-3','Gemma 3').replace('-pt',''),repo_id=m['repo_id'],revision=m.get('revision'),family='Gemma' if 'gemma' in m['id'] else 'Mistral' if 'mistral' in m['id'] else 'Qwen',layers=text.get('num_hidden_layers'),hidden=text.get('hidden_size'),metadata_ready=m.get('status')=='metadata_ready',pilot=pilot.get('status')=='completed' and pilot.get('model_id')==m['repo_id'],measured=counts[m['id']],expected=expected[m['id']]))
    return dict(local_activity=activity,run_id=watched.name,state=state,terminal=terminal,stale=stale,released=released,active=remote.get('active'),finished_utc=completion.get('finished_utc'),snapshot_age_seconds=age,source_commit=read(watched/'frozen.json').get('source_commit'),archive_sha256=completion.get('result_sha256'),main=dict(measured=len(measured),expected=len(entries)),stages=stages,kernels=kernels,semantics=semantics,pilot=pilot,profiles=profiles,models=models,events=events,raw_log=raw,raw_status=completion or remote)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8789);a=p.parse_args();run=a.run.resolve()
    evidence={
        '/evidence/pilot.json':run/'remote'/run.name/'artifacts/real-qwen8-pilot/summary.json',
        '/evidence/profiles.json':run/'remote'/run.name/'artifacts/real-qwen8-pilot/profiles.json',
        '/evidence/protocol.md':run.parents[3]/'docs/LLM_CAMPAIGN_v001.md',
        '/evidence/report.md':run.parent/'PILOT_RESULTS.md',
    }
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':data=PAGE.read_bytes();kind='text/html; charset=utf-8'
            elif path=='/state':data=json.dumps(snapshot(run),allow_nan=False).encode();kind='application/json'
            elif path in evidence and evidence[path].is_file():data=evidence[path].read_bytes();kind='application/json' if path.endswith('.json') else 'text/plain; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
