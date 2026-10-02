"""Read-only progress observation for the active Mistral input-preparation worker."""
import json
from pathlib import Path
import time

RUN = b'20260919T080427Z-N9-tokenize-mistral-v003-v5e-v004-1b9bce'
rows = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        command = (path / 'cmdline').read_bytes()
        if RUN not in command or b'prepare_model_application_v001.py' not in command:
            continue
        samples = []
        for index in range(2):
            status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            io = dict(line.split(':', 1) for line in (path / 'io').read_text().splitlines() if ':' in line)
            samples.append({'state': status.get('State', '').strip(), 'rss': status.get('VmRSS', '').strip(),
                'threads': status.get('Threads', '').strip(), 'user_jiffies': int(stat[11]),
                'system_jiffies': int(stat[12]), 'read_bytes': int(io.get('read_bytes', 0)),
                'read_characters': int(io.get('rchar', 0)), 'wait_channel': (path / 'wchan').read_text().strip()})
            if not index:
                time.sleep(1)
        rows.append({'pid': int(path.name), 'one_second_samples': samples})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
print(json.dumps({'kind': 'read_only_input_worker_observation', 'run_id': RUN.decode(), 'workers': rows}), flush=True)
