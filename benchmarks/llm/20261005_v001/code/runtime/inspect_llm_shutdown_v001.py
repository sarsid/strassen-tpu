"""Read only owned-session archive progress during an operator shutdown."""
import argparse
import json
from pathlib import Path
import re
import time

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid run')
r=Path('/content/Strassen_MM_Focus/runs')/a.run_id
expected=json.loads((r/'launch.json').read_text())['command']
processes=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv=[b.decode() for b in (proc/'cmdline').read_bytes().split(b'\0') if b]
        if argv!=expected:continue
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        processes.append(dict(pid=int(proc.name),state=fields[0],utime_ticks=int(fields[11]),stime_ticks=int(fields[12])))
    except FileNotFoundError:continue
files={}
for f in (r.with_suffix('.tar.gz'),r/'archive-ready.json',r/'status.json'):
    if f.exists():
        s=f.stat();files[f.name]=dict(bytes=s.st_size,mtime=s.st_mtime,age_seconds=time.time()-s.st_mtime)
print(json.dumps(dict(kind='shutdown_archive_progress',utc_epoch=time.time(),run_id=a.run_id,processes=processes,files=files)),flush=True)
