"""Lightweight read-only Python bootstrap inspection; never installs or imports JAX."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

FAILED=Path('/content/Strassen_MM_Focus/.runtime_private/20260921-mlsys-main-v5e-v001/cpu-tools')
rows=[]
for interpreter in (Path(sys.executable),FAILED/'bin/python'):
    for module in ('pip','ensurepip'):
        if not interpreter.exists():continue
        command=[str(interpreter),'-I','-m',module,'--version']
        result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=15)
        rows.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
paths={name:(spec.origin if (spec:=importlib.util.find_spec(name)) else None)for name in ('pip','ensurepip','venv')}
cfg=FAILED/'pyvenv.cfg'
print(json.dumps(dict(kind='read_only_cpu_bootstrap_inspection',python=sys.executable,version=sys.version,
                     system_module_origins=paths,failed_environment_config=cfg.read_text() if cfg.exists() else None,
                     checks=rows,installation_performed=False),sort_keys=True))
