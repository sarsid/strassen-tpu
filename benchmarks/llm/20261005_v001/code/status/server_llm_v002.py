"""Small read-only loopback view of this campaign's actual evidence."""
import argparse, json
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
PAGE='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v6e LLM experiment</title><style>body{font:16px system-ui;background:#101720;color:#e6edf3;margin:40px auto;padding:0 24px;max-width:1080px}h1{font-size:30px}h2{font-size:20px;color:#aacbed}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1a2633;padding:18px;border-radius:10px;max-height:480px;overflow:auto}.muted{color:#a3b4c4}.big{font-size:28px}a{color:#80c4ff}</style><h1>v6e LLM experiment</h1><p>Qwen3 8B · 14B · 32B &nbsp; Mistral 7B · 24B &nbsp; Gemma3 12B · 27B</p><p>Default Native · Tuned Native · Tuned Cubic · S1 · S2<br>Separate BF16 / FP32 stores · DEFAULT precision</p><p class="big">Main prefill grid: <b>0 / 700</b></p><p class="muted">Qualification and pilot results do not count as full-model measurements.</p><h2 id="stage">Loading evidence…</h2><p id="fresh"></p><pre id="stages"></pre><h2>Latest execution log</h2><pre id="log"></pre><script>async function update(){try{let d=await(await fetch('/state',{cache:'no-store'})).json();document.querySelector('#stage').textContent=d.stage;document.querySelector('#fresh').textContent=d.freshness;document.querySelector('#stages').textContent=JSON.stringify(d.status,null,2);document.querySelector('#log').textContent=d.log;}catch(e){document.querySelector('#fresh').textContent='Local feed unavailable — execution state unknown';}}update();setInterval(update,10000);</script>'''
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8789);a=p.parse_args();run=a.run.resolve()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':data=PAGE.encode();kind='text/html; charset=utf-8'
            elif self.path=='/state':
                latest=run/'latest.json';done=run/'completion.json';status={};log='No remote measurements yet.';stage='Preparing';fresh='No remote snapshot received'
                if latest.exists():
                    value=json.loads(latest.read_text());status=value.get('status.json',{});stage=status.get('active',status.get('status','starting'));log='\n\n'.join(k+'\n'+v for k,v in value.get('tails',{}).items())
                    age=max(0,datetime.now(timezone.utc).timestamp()-latest.stat().st_mtime);fresh=f'Last remote snapshot {age:.0f}s ago'+(' — stale; running state unverified' if age>120 and not done.exists() else '')
                if done.exists():status=json.loads(done.read_text());stage='Qualification and pilot complete; main grid not started' if status.get('status')=='completed' else status.get('status','finished')
                data=json.dumps(dict(stage=stage,status=status,log=log,freshness=fresh)).encode();kind='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
