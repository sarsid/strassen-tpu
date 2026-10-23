"""Stop only this ablation's inner benchmark after its first completed group.

SIGINT lets the runner seal all results, including not-run cases. The adapter
and outer worker are left alive to archive the outcome and release ownership.
"""
import json
import os
from pathlib import Path
import signal
import time

RUN = '20260920T154332Z-SOPT-screen-v5e-v004-d03e5b'
root = Path('/content/Strassen_MM_Focus/runs') / RUN
deadline = time.monotonic() + 110
while time.monotonic() < deadline:
    path = root / 'artifacts/results.jsonl'
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
    completed = [r for r in records if r.get('event') == 'group_complete']
    if completed:
        matches = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            except (OSError,UnicodeDecodeError):continue
            if any(v.endswith('/tools/benchmark_strassen_optimized_v001.py') for v in argv) and str(root / 'artifacts') in argv:
                matches.append(int(proc.name))
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one owned inner benchmark: ' + str(matches))
        record = {'reason': 'Preserve matched large-tile VMEM failures; retry smaller tiles in a new run',
                  'signal': 'SIGINT', 'pid': matches[0], 'completed_groups': len(completed), 'run_id': RUN}
        with (root / 'requested-stop.json').open('x') as stream:json.dump(record,stream,indent=2)
        os.kill(matches[0], signal.SIGINT)
        print(json.dumps(record), flush=True)
        break
    time.sleep(1)
else:
    print(json.dumps({'run_id': RUN, 'signal_sent': False, 'reason': 'First group still incomplete'}), flush=True)
