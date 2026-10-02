"""Stop one explicitly named benchmark child; its outer worker archives evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expect-endpoint', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Invalid run ID')
    run = Path('/content/Strassen_MM_Focus/runs') / args.run_id
    plan = json.loads((run / 'launch.json').read_text())
    if plan['allocation_id'] != args.expect_endpoint:
        raise ValueError('Allocation identity mismatch')
    start = json.loads((run / 'benchmark-start.json').read_text())
    pid = start['pid']
    proc = Path('/proc') / str(pid)
    def matches():
        try:
            argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
            cwd = (proc / 'cwd').resolve(strict=True)
        except FileNotFoundError:
            return False
        if not argv:
            return False
        if argv != start['command'] or argv != plan['command']:
            raise RuntimeError('Recorded benchmark PID no longer matches its command')
        if cwd != Path(plan['source_root']).resolve() or not cwd.is_relative_to(run / 'source'):
            raise RuntimeError('Benchmark does not belong to the named frozen run')
        if os.getpgid(pid) != pid:
            raise RuntimeError('Benchmark is not the expected isolated process-group leader')
        return True
    signals = []
    if matches():
        os.killpg(pid, signal.SIGTERM)
        signals.append('SIGTERM')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and matches():
            time.sleep(.2)
        if matches():
            os.killpg(pid, signal.SIGKILL)
            signals.append('SIGKILL')
    print(json.dumps(dict(kind='operator_stop', run_id=args.run_id, benchmark_pid=pid,
                         endpoint=args.expect_endpoint, signals=signals,
                         outer_worker_preserved=True,
                         utc=datetime.now(timezone.utc).isoformat())), flush=True)


if __name__ == '__main__':
    main()
