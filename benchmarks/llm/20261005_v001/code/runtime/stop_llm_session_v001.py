"""Operator stop of one owned session; retain worker finally for archival.

Freeze its owner before terminating owned phase children, then interrupt the
owner into its archival finally block. Never signal an unrelated process.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import time


def command(pid):
    try:
        return [x.decode() for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:
        return []


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', args.run_id):
        raise ValueError('Invalid owned run')
    run = Path('/content/Strassen_MM_Focus/runs') / args.run_id
    expected = json.loads((run / 'launch.json').read_text())['command']
    if expected[1:] != [str(run / 'source/runtime/llm_model_session_v001.py'), '--worker', str(run)]:
        raise ValueError('Unexpected session owner command')
    owners = [int(p.name) for p in Path('/proc').glob('[0-9]*') if command(int(p.name)) == expected]
    if len(owners) != 1:
        if (run / 'archive-ready.json').exists():
            print(json.dumps(dict(kind='operator_session_stop', status='already_archived', run_id=args.run_id)), flush=True)
            return
        raise RuntimeError('Cannot identify exactly one owned session process')
    owner = owners[0]
    if os.getpgid(owner) != owner or command(owner) != expected:
        raise RuntimeError('Owner identity changed')
    receipt = dict(kind='operator_session_stop', run_id=args.run_id, owner_pid=owner,
        utc=datetime.now(timezone.utc).isoformat(), reason='User requested pause and release to reconsider LLM strategy',
        status='stop_requested', children=[])
    path = run / 'artifacts/operator-session-stop.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    os.kill(owner, signal.SIGSTOP)
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat = p.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if int(stat[1]) != owner or stat[0] == 'Z':
            continue
        pid = int(p.name)
        argv = command(pid)
        allowed = [str(run / 'source/tools' / name) for name in ('run_llm_case_v003.py', 'prepare_llm_case_v003.py')]
        if len(argv) < 2 or argv[1] not in allowed or os.getpgid(pid) != pid:
            raise RuntimeError('Unexpected session child; owner remains frozen')
        if command(pid) != argv:
            raise RuntimeError('Child identity changed; owner remains frozen')
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.2)
        forced = alive(pid)
        if forced:
            if command(pid) != argv:
                raise RuntimeError('Child identity changed')
            os.killpg(pid, signal.SIGKILL)
        receipt['children'].append(dict(pid=pid, script=Path(argv[1]).name, forced_kill=forced))
    # The worker catches KeyboardInterrupt and runs its existing archive path.
    if command(owner) != expected:
        raise RuntimeError('Owner identity changed')
    receipt.update(status='archival_requested', owner_preserved_for_archival=True)
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    os.kill(owner, signal.SIGINT)
    os.kill(owner, signal.SIGCONT)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
