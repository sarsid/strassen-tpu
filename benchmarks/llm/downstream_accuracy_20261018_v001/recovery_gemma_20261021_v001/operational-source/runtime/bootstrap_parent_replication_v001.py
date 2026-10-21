"""Create the upload root; no package installation or TPU initialization."""
import json, shutil
from pathlib import Path
root=Path('/content/Strassen_MM_Focus')
root.mkdir(exist_ok=True)
print(json.dumps(dict(kind='upload_root_ready',free_bytes=shutil.disk_usage(root).free)),flush=True)
