"""CPU-only checks before the retained-TPU B64 residency retry."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ['tests/test_gemma4_error_metrics_v001.py','tests/test_gemma4_residency_v001.py',
             'tests/test_gemma4_b64_retry_v002.py','tests/test_gemma4_retry_observer_v002.py']:
    print('CHECK '+name,flush=True)
    subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,check=True)
print('All residency-repair checks passed',flush=True)
