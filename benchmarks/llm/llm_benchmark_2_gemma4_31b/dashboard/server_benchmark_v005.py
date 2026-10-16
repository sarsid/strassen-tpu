"""Read-only LLM benchmark dashboard; remote freshness and experiment state differ."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('STRASSEN_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
UI = Path(__file__).with_name('benchmark_v003.html')
POINTER = ROOT/'status/llm_benchmark_active_v001.json'
STAGES = ['tune','qualify','confirm','quality','prefill','archive']


def read(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def age(value):
    try: return max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(value.replace('Z','+00:00'))).total_seconds())
    except (ValueError,TypeError,AttributeError): return None


def snapshot():
    p=read(POINTER,{})
    rel=p.get('run','')
    run=(ROOT/rel).resolve()
    if not rel or not run.is_relative_to(ROOT/'runs'):
        return dict(status='not_started',title='LLM benchmark',detail='No active experiment is selected.',batches=[],events=[])
    local=read(run/'progress.json',{})
    latest=read(run/'latest.json',{})
    remote=latest.get('status.json',{})
    progress=latest.get('progress.json',{})
    completion=read(run/'completion.json',{})
    stage_rows=remote.get('stages',[])
    allocation_log=run/'allocation-events.jsonl'
    if not remote and not completion and allocation_log.exists():
        history=[]
        for line in allocation_log.read_text().splitlines():
            try:history.append(json.loads(line))
            except ValueError:pass
        if history and history[-1].get('kind') in ('provisioning_request','provisioning_response','provisioning_wait','preflight_assignments'):
            latest_request=history[-1]
            local=dict(local,status='waiting',stage='allocation',stage_label='Waiting for Colab v6e',test='Allocation request in progress',allocation='requested',utc=latest_request.get('utc'),detail='One tracked request; no TPU computation has started.',next_step='Obtain v6e allocation, then qualify the official checkpoint.')
    cards=[]
    for batch in p.get('batches',[]):
        cells={s:dict(status='pending') for s in STAGES}
        for row in stage_rows:
            if row.get('batch')==batch and row.get('stage') in cells:
                cells[row['stage']]=dict(status='complete' if row.get('returncode')==0 else 'failed',attempt=row.get('attempt',0),seconds=row.get('seconds'))
        for row in stage_rows:
            if row.get('batch')==batch and row.get('name') and row.get('stage') in cells:
                report=read(run/'checkpoints'/row['name']/'summary.json',{})
                if report.get('all_pass') is False:cells[row['stage']]['status']='failed_gate'
        for row in remote.get('batches',[]):
            if row.get('batch')==batch: cells['archive']=dict(status=row.get('status','unknown'))
        if progress.get('batch')==batch and progress.get('stage') in cells and remote.get('status')=='running':
            cells[progress['stage']]=dict(status='running',detail=progress.get('test'))
        cards.append(dict(batch=batch,tokens=batch*p.get('sequence',1024),stages=cells))
    fresh=age(latest.get('observed_utc') or local.get('utc'))
    state=completion.get('status') or remote.get('status') or local.get('status','preparing')
    stale=state in ('running','waiting') and (fresh is None or fresh>120)
    current=progress if remote else local
    if state not in ('running','waiting','preparing'):
        current=dict(current,stage_label='Finished' if state=='completed' else 'Stopped: '+state,test=completion.get('error') or remote.get('error') or 'See saved stage results and accuracy gates.')
    log=run/'dashboard-events.jsonl'
    events=[]
    if log.exists():
        for line in log.read_text().splitlines()[-100:]:
            try:events.append(json.loads(line))
            except ValueError:pass
    return dict(title=p.get('title','LLM benchmark'),model=p.get('model'),sequence=p.get('sequence'),
        status=state,stale=stale,heartbeat_age=fresh,current=current,batches=cards,
        completed=sum(c['stages']['archive']['status']=='completed' for c in cards),
        total=len(cards),events=events[-40:],qualification=remote.get('qualification',local.get('qualification','not_started')),
        updated_utc=latest.get('observed_utc') or local.get('utc'),allocation=local.get('allocation','not_requested'),
        released=completion.get('release_exit_code')==0,run=rel,
        links=p.get('links',[]),raw=dict(remote=remote,progress=progress,local=local,completion=completion))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?',1)[0]
        if path=='/':data=UI.read_bytes();mime='text/html; charset=utf-8'
        elif path=='/api/status':data=json.dumps(snapshot(),allow_nan=False).encode();mime='application/json'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8790);a=ap.parse_args()
    print(f'LLM benchmark dashboard: http://127.0.0.1:{a.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
