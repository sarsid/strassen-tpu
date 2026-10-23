"""Read available host, disk and TPU memory in an isolated JAX subprocess."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

mem = {line.split(':')[0]: int(line.split()[1])*1024
       for line in Path('/proc/meminfo').read_text().splitlines()
       if line.startswith(('MemTotal:', 'MemAvailable:'))}
probe = subprocess.run([sys.executable, '-c',
    'import jax,json; print(json.dumps([dict(kind=d.device_kind, memory=d.memory_stats()) for d in jax.devices()]))'],
    capture_output=True, text=True, timeout=120, check=True)
print(json.dumps(dict(host_memory_bytes=mem, disk_free_bytes=shutil.disk_usage('/content').free,
                      cpu_count=os.cpu_count(), devices=json.loads(probe.stdout.strip().splitlines()[-1]))))
