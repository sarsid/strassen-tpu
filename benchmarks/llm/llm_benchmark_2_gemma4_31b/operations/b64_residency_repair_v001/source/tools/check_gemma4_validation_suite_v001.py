"""Local checks for the approved validation split; no TPU or network."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
checks=['tests/test_gemma4_validation_v001.py','tests/test_gemma4_fp32_semantics_v001.py',
        'tools/check_gemma4_controller_v004.py','tools/check_gemma4_launch_v004.py',
        'tests/test_gemma4_retained_supervisor_v003.py']
for path in checks:
    print('CHECK '+path,flush=True)
    subprocess.run([sys.executable,str(ROOT/path)],check=True,cwd=ROOT)
print('All approved-validation checks passed',flush=True)
