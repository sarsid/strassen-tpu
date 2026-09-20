"""Install an isolated plotting environment; never alter TPU dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
target = a.root.resolve() / '.venv-plots-v001'
if target.exists():
    raise FileExistsError(target)
a.output_dir.mkdir(parents=True, exist_ok=False)
subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
python = target / 'bin/python'
subprocess.run([str(python), '-m', 'pip', 'install', 'matplotlib==3.10.6'], check=True)
inventory = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
(a.output_dir / 'requirements.txt').write_text(inventory)
(a.output_dir / 'environment.json').write_text(json.dumps({
    'python': str(python), 'purpose': 'Local rendering of archived scientific results only',
    'matplotlib_pin': '3.10.6', 'no_tpu_execution': True}, indent=2))
print(inventory)
