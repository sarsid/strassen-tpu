"""Stop only the identified pilot after a complete Q/BF16 selection group."""
import argparse,json,os,re,signal
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',a.run_id):raise ValueError('Invalid run ID')
    run=Path('/content/Strassen_MM_Focus/runs')/a.run_id
    ledger=run/'artifacts/tune/events.jsonl'
    records=[]
    for line in ledger.read_text().splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    selected={r['family'] for r in records if r.get('kind')=='selected' and r.get('site')=='q' and r.get('dtype')=='bfloat16'}
    if selected!={'cubic','s1','s2'}:
        print(json.dumps(dict(kind='pilot_group_not_complete',selected=sorted(selected))));return
    expected=str(run/'source/tools/tune_parent_fused_v002.py')
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:command=(path/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if len(command)>1 and command[1]==expected and str(run/'artifacts/tune') in command:matches.append((int(path.name),command))
    if len(matches)!=1:raise RuntimeError('Expected exactly one matching pilot process')
    pid,command=matches[0]
    if os.getpgid(pid)!=pid:raise RuntimeError('Pilot is not the expected isolated process group')
    receipt=dict(kind='planned_pilot_stop',pid=pid,command=command,completed_group='q:bfloat16',
                 reason='Measured compilation cost would exceed original tuning timeout; preserve pilot and run v003 bounded menu',
                 candidate_outcomes=sum(r.get('kind') in ('candidate','candidate_failure') for r in records))
    (run/'artifacts/operator-stop.json').write_text(json.dumps(receipt,indent=2)+'\n')
    os.killpg(pid,signal.SIGTERM)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
