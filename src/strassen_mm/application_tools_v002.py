"""Standard archived-run adapter for the isolated ensurepip recovery installer."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from . import benchmark_v001 as base


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','phase','output-dir','expected-identity','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=5400)
    p.add_argument('--venv-dir',default='/content/Strassen_MM_Focus/model-tools-v002')
    args=p.parse_args(argv);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    journal=base.Journal(out,args.phase);started=time.monotonic();status='failed';error=None;record=None
    try:
        if 'jax' in sys.modules:raise RuntimeError('Model tools setup must not import JAX')
        expected=json.loads(Path(args.expected_identity).read_text());expected=expected.get('identity',expected)
        current={'colab_endpoint':args.allocation_id,'hostname':socket.gethostname(),
                 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'versions':{name:importlib.metadata.version(name) for name in expected['versions']}}
        base.exclusive_json(out/'environment.json',{'identity':current,'qualification':'Host/core packages only; no JAX import'})
        for name in ('colab_endpoint','hostname','boot_id','versions'):
            if current[name]!=expected[name]:raise RuntimeError('Model tools setup cohort mismatch: '+name)
        journal.emit('identity_check',status='matched')
        tool=Path(__file__).resolve().parents[2]/'tools/install_model_tools_v002.py'
        command=[sys.executable,str(tool),'--output-dir',str(out/'preparation-00'),
                 '--venv-dir',args.venv_dir,'--max-wall-seconds',str(args.max_wall_seconds)]
        base.exclusive_json(out/'command-00.json',{'argv':command})
        with (out/'preparation-00.log').open('xb') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.max_wall_seconds+10,check=False)
        record={'exit_code':child.returncode,'status':json.loads((out/'preparation-00/status.json').read_text())}
        journal.emit('preparation_result',**record)
        if child.returncode or record['status']['status']!='completed':raise RuntimeError('Isolated model-tools recovery failed')
        after={name:importlib.metadata.version(name) for name in expected['versions']}
        if after!=current['versions']:raise RuntimeError('Global core package versions changed')
        status='completed'
    except BaseException as problem:
        error={'type':type(problem).__name__,'message':str(problem)}
        journal.emit('run_error',**error,traceback=traceback.format_exc())
    finally:
        summary={'phase':args.phase,'status':status,'completed':status=='completed','record':record,
                 'error':error,'elapsed_seconds':time.monotonic()-started,'jax_imported':'jax' in sys.modules}
        journal.emit('run_complete',**summary);journal.close();base.exclusive_json(out/'summary.json',summary)
        base.exclusive_json(out/'artifact_manifest.json',{'sha256':{str(path.relative_to(out)):base.digest_file(path)
              for path in sorted(out.rglob('*')) if path.is_file()},'sealed_utc':base.utc_now()})
        print(json.dumps({'phase':args.phase,'status':status,'error':error}),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
