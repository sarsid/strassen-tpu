"""Detached local watchdog and sleep guard for the authorized queue only."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from llm_queue_state_v002 import read,save,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--controller-python',required=True);a=p.parse_args()
    queue=a.campaign.resolve()/'queue_v001';lock=(queue/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    save(queue/'watchdog-process.json',dict(pid=os.getpid(),caffeinate_pid=guard.pid,utc=utc()))
    try:
        while True:
            state=read(queue/'state.json')
            if state['status'] in ('completed','finished_with_failures','paused','blocked_shared_setup'):break
            command=[a.controller_python,str(queue/state.get('source_dir','source')/'tools/run_llm_queue_v004.py'),'--campaign',str(a.campaign.resolve()),'--controller-python',a.controller_python]
            with (queue/'supervisor.log').open('ab') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                save(queue/'supervisor-process.json',dict(pid=child.pid,utc=utc(),command=command));code=child.wait()
            print('Supervisor exit '+str(code)+'; checking saved state',flush=True)
            time.sleep(30)
    finally:guard.terminate()

if __name__=='__main__':main()
