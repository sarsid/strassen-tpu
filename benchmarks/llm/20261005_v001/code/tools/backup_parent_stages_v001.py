"""Copy finished stage evidence locally while the existing controller keeps running."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--controller-python',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve()
    if not run.is_relative_to(root/'runs'):raise ValueError('Expected project run directory')
    allocation=json.loads((run/'allocation.json').read_text());backup=run/'stage-backups';backup.mkdir(exist_ok=False)
    manifest={};attempts={};deadline=time.monotonic()+7200
    controller=[a.controller_python,str(root/'runtime/colab_control_v003.py')]
    while time.monotonic()<deadline:
        try:snapshot=json.loads((run/'latest.json').read_text())
        except (FileNotFoundError,ValueError):time.sleep(5);continue
        for stage in snapshot.get('status.json',{}).get('stages',[]):
            name=stage['name']
            if name.endswith('setup'):continue
            if not name.replace('-','').isalnum():raise ValueError('Unexpected stage name')
            files=['events.jsonl','summary.json']+(['profile.json'] if name in ('tune','refine') else [])
            target=backup/name;target.mkdir(exist_ok=True)
            for filename in files:
                key=name+'/'+filename
                if key in manifest:continue
                local=run/'remote'/run.name/'artifacts'/name/filename
                attempt=attempts.get(key,0)
                if attempt>=3 and not local.exists():continue
                candidate=target/(filename+'.attempt-'+str(attempt));attempts[key]=attempt+1
                try:
                    if local.exists():shutil.copyfile(local,candidate)
                    else:
                        with (target/(filename+'.attempt-'+str(attempt)+'.log')).open('x') as log:
                            outcome=subprocess.run(controller+['download','--session',allocation['session'],
                                '--expect-endpoint',allocation['endpoint'],'--remote',
                                '/content/Strassen_MM_Focus/runs/'+run.name+'/artifacts/'+key,'--local',str(candidate)],
                                stdout=log,stderr=subprocess.STDOUT,timeout=90)
                        if outcome.returncode:continue
                    content=candidate.read_text()
                    if filename.endswith('jsonl'):
                        parsed=[json.loads(line) for line in content.splitlines() if line.strip()]
                        if not parsed:raise ValueError('Empty stage ledger')
                    else:json.loads(content)
                    shutil.copyfile(candidate,target/filename)
                    manifest[key]=dict(sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),bytes=candidate.stat().st_size,
                                       stage_returncode=stage['returncode'],source='completed stage; final archive hash comparison follows')
                    temp=backup/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(backup/'manifest.json')
                    print(json.dumps(dict(kind='stage_backup',file=key,**manifest[key])),flush=True)
                except Exception as error:print(json.dumps(dict(kind='backup_attempt_failed',file=key,error_type=type(error).__name__)),flush=True)
        if (run/'completion.json').exists():return
        time.sleep(45)


if __name__=='__main__':main()
