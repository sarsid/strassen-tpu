"""Run the bounded local Qwen14 checks as separate CPU-only processes."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for name in ('check_qwen14_geometry_v001.py','check_qwen14_batch_v001.py','check_qwen14_launch_v001.py','check_qwen14_controller_v001.py'):
    print('CHECK '+name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools'/name)],check=True,cwd=ROOT)
print('ALL QWEN14 LOCAL CHECKS PASSED',flush=True)
