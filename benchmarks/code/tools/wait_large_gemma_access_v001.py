"""Wait only for an explicit authorization-and-staging receipt; never read a token."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);a=p.parse_args()
end=time.monotonic()+14000
while time.monotonic()<end:
    path=a.cohort/'gemma-access-authorized.json'
    if path.is_file():
        d=json.loads(path.read_text());plan=json.loads((a.cohort/'plan.json').read_text())
        if d.get('authorized') is not True or d.get('staged') is not True or d.get('endpoint')!=plan['allocation_id']:
            raise RuntimeError('Invalid Gemma access receipt')
        print(json.dumps({'authorized_gemma_access_ready':True}));break
    time.sleep(10)
else:raise TimeoutError('Gemma access authorization/staging still pending')
