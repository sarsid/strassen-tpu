"""Install an isolated CPU validation environment; benchmark environments untouched."""
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import venv

root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
out.mkdir()
target = root / '.runtime_private/gemma-validation-v001'
if target.exists():
    raise FileExistsError('Validation environment already exists; use a new version')
venv.EnvBuilder(with_pip=True).create(target)
python = target / 'bin/python'
requirements = Path('plans/gemma_support_v001/requirements.txt').resolve()
subprocess.run([str(python), '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
                '--index-url', 'https://pypi.org/simple', '-r', str(requirements)], check=True)
freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
with (out / 'pip-freeze.txt').open('x') as handle:
    handle.write(freeze)
with (out / 'environment.json').open('x') as handle:
    json.dump({'python': str(python), 'platform': platform.platform(), 'host_python': sys.version,
               'scope': 'CPU correctness only; no TPU performance comparisons'}, handle, indent=2)
subprocess.run([str(python), '-c', 'import jax, torch, transformers; print({"jax":jax.__version__,"torch":torch.__version__,"transformers":transformers.__version__,"devices":[str(d) for d in jax.devices()]})'], check=True)
