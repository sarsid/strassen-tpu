"""Stop only a legacy FP32 measurement process after its BF16 group is saved.

The owning worker remains alive to archive partial diagnostics, and its local
controller downloads the archive and releases its owned TPU normally.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal


def matches(command, script, output):
    return (str(script) in command and '--output' in command
            and command.index('--output') + 1 < len(command)
            and command[command.index('--output') + 1] == str(output))


def main():
    p = argparse.ArgumentParser();p.add_argument('--run-id', required=True);a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', a.run_id):
        raise ValueError('Invalid run ID')
    run = Path('/content/Strassen_MM_Focus/runs') / a.run_id
    output = run / 'artifacts/main-prefill'
    job = json.loads((run / 'source/job.json').read_text())
    rows = [json.loads(x) for x in (output / 'completed_entries.jsonl').read_text().splitlines()]
    expected = set(job['groups']['bfloat16'])
    if {r['entry_id'] for r in rows if r['output_dtype'] == 'bfloat16' and r['status'] == 'completed'} != expected or len(expected) != 5:
        raise RuntimeError('Complete BF16 group is not present; refusing stop')
    script = run / 'source/tools/run_llm_case_v002.py'
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().decode().rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if matches(command, script, output):
            found.append(int(path.parent.name))
    if len(found) > 1:
        raise RuntimeError('More than one matching measurement process')
    if not found:
        print(json.dumps(dict(kind='precision_stop', status='already_stopped', run_id=a.run_id)));return
    events = [json.loads(x) for x in (output / 'events.jsonl').read_text().splitlines() if x.strip()]
    latest = next((e for e in reversed(events) if e.get('output_dtype')), {})
    if latest.get('output_dtype') != 'float32':
        raise RuntimeError('Latest observed phase is not FP32; refusing stop')
    pid = found[0]
    if os.getpgid(pid) != pid:
        raise RuntimeError('Measurement does not own its process group')
    # Recheck command ownership immediately before signaling.
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if not matches(command, script, output):
        raise RuntimeError('Process identity changed')
    receipt = dict(kind='precision_stop', status='stop_requested', run_id=a.run_id, pid=pid,
                   utc=datetime.now(timezone.utc).isoformat(), stopped_output_dtype='float32',
                   reason='User narrowed remaining LLM campaign to BF16 output',
                   preserved_bf16_ids=sorted(expected), last_event=latest)
    (output / 'operator-precision-stop.json').write_text(json.dumps(receipt, indent=2) + '\n')
    os.killpg(pid, signal.SIGTERM)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
