"""Read host capacity without reading credentials or importing JAX."""
from pathlib import Path
import os,json,shutil
root=Path('/content/Strassen_MM_Focus/.runtime_private/20260922-large-real-v001')
assert root.is_dir()
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:','MemAvailable:'))}
row=dict(host_memory_bytes=mem,disk_free_bytes=shutil.disk_usage(root).free,cpu_count=os.cpu_count())
with (root/'host_preflight.json').open('x') as f:json.dump(row,f)
print(json.dumps(row))
