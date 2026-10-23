"""Finish an operator-interrupted session archive with fast gzip compression.

Only applies after the measurement has stopped. Existing per-case archives and
scientific files are untouched. The controller still verifies the final SHA256
before releasing its owned allocation. No runtime or measurement is restarted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tarfile
import time


def argv(pid):
    try:return [b.decode() for b in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if b]
    except FileNotFoundError:return []


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid owned run')
    r=Path('/content/Strassen_MM_Focus/runs')/a.run_id
    if (r/'archive-ready.json').exists():
        print(json.dumps(dict(kind='paused_archive',status='already_ready')),flush=True);return
    stopped=json.loads((r/'artifacts/operator-session-stop.json').read_text())
    status=json.loads((r/'status.json').read_text())
    if stopped['status']!='archival_requested' or status.get('error_type')!='KeyboardInterrupt' or status['status']!='failed':
        raise RuntimeError('Session was not interrupted by the operator')
    expected=json.loads((r/'launch.json').read_text())['command'];pid=stopped['owner_pid']
    if argv(pid)!=expected or os.getpgid(pid)!=pid:raise RuntimeError('Owner mismatch')
    os.kill(pid,signal.SIGSTOP)
    for proc in Path('/proc').glob('[0-9]*'):
        command=argv(int(proc.name))
        if any(str(r/'source/tools'/name) in command for name in ('run_llm_case_v003.py','prepare_llm_case_v003.py')):
            raise RuntimeError('Measurement remains alive; owner left frozen')
    archive=r.with_name(r.name+'-operator-paused-fast.tar.gz')
    if archive.exists():raise FileExistsError('Preserve existing fast-archive attempt')
    started=time.monotonic()
    with tarfile.open(archive,'w:gz',compresslevel=1) as package:
        package.add(r/'artifacts',arcname=r.name+'/artifacts',filter=lambda info:None if info.name.endswith('.tar.gz') else info)
        package.add(r/'launch.json',arcname=r.name+'/launch.json')
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    if argv(pid)!=expected:raise RuntimeError('Owner changed during archive')
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    ready=dict(path=str(archive),sha256=h.hexdigest(),bytes=archive.stat().st_size)
    tmp=r/'archive-ready.json.tmp';tmp.write_text(json.dumps(ready,indent=2)+'\n');tmp.replace(r/'archive-ready.json')
    receipt=dict(kind='paused_archive',status='ready',owner_pid=pid,compression_level=1,
        elapsed_seconds=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**ready)
    (r/'operator-fast-archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
