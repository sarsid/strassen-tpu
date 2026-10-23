"""Archived application-input preparation without importing or modifying JAX.

The outer frozen launcher owns process timeouts, source and result sealing.
This adapter journals each preparation subprocess and always emits a summary.
Large reproducible model files remain in a separate immutable remote cache.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

MODELS=('Qwen/Qwen3-0.6B','mistralai/Mistral-7B-v0.3','google/gemma-3-1b-pt')

def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')

def main():
    p=argparse.ArgumentParser()
    for name in ('campaign','phase','output-dir','expected-identity','allocation-id'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=int,default=3600)
    p.add_argument('--action',required=True,choices=('probe','download','corpus','model-tools','tokenize','reference'))
    p.add_argument('--model-id',choices=MODELS)
    p.add_argument('--revision')
    p.add_argument('--cache-root',default='/content/Strassen_MM_Focus/models')
    p.add_argument('--venv-dir',default='/content/Strassen_MM_Focus/model-tools-v001')
    p.add_argument('--model-manifest')
    p.add_argument('--corpus-manifest')
    p.add_argument('--tokens')
    args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    expected=json.loads(Path(args.expected_identity).read_text());expected=expected.get('identity',expected)
    current={'colab_endpoint':args.allocation_id,'hostname':socket.gethostname(),
             'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'versions':{n:importlib.metadata.version(n) for n in expected['versions']}}
    save(out/'environment.json',{'identity':current,'qualification':'Host/core-package identity only; no JAX import'})
    for key in ('colab_endpoint','hostname','boot_id','versions'):
        if current[key]!=expected[key]: raise RuntimeError('Preparation cohort identity mismatch: '+key)
    save(out/'arguments.json',vars(args))
    tool=Path(__file__).resolve().parents[2]/'tools/prepare_model_application_v001.py'
    models=MODELS if args.action=='probe' and args.model_id is None else (args.model_id,)
    records=[]; started=time.monotonic()
    for index,model in enumerate(models):
        target=out/f'preparation-{index:02d}'
        python=str(Path(args.venv_dir)/'bin/python') if args.action in ('tokenize','reference') else sys.executable
        command=[python,str(tool),args.action,'--output-dir',str(target)]
        if args.action in ('probe','download'):
            if model is None: p.error('model-id required for download')
            command+=['--model-id',model,'--use-existing-token']
            if args.revision: command+=['--revision',args.revision]
        if args.action in ('download','corpus'): command+=['--cache-root',args.cache_root]
        if args.action=='model-tools': command+=['--venv-dir',args.venv_dir]
        if args.action in ('tokenize','reference'):
            if not args.model_manifest: p.error('model-manifest required')
            command+=['--model-manifest',args.model_manifest]
        if args.action=='tokenize': command+=['--corpus-manifest',args.corpus_manifest]
        if args.action=='reference': command+=['--tokens',args.tokens]
        record={'action':args.action,'model_id':model,'command':command}
        save(out/f'command-{index:02d}.json',record)
        try:
            with (out/f'preparation-{index:02d}.log').open('x') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                     timeout=max(1,args.max_wall_seconds-(time.monotonic()-started)),check=False)
            record['exit_code']=child.returncode
            status=target/'status.json'
            record['status']=json.loads(status.read_text()) if status.is_file() else {'status':'failed_no_status'}
        except BaseException as error:
            record['status']={'status':'failed','error_type':type(error).__name__}
        records.append(record)
        print(json.dumps({'event':'preparation_result',**record}),flush=True)
    failed=[r for r in records if r['status'].get('status') not in ('completed','blocked_access')]
    summary={'phase':args.phase,'action':args.action,'records':records,'status':'failed' if failed else 'completed',
             'blocked_access_is_not_successful_model':True,'elapsed_seconds':time.monotonic()-started}
    save(out/'summary.json',summary)
    return 1 if failed else 0

if __name__=='__main__': raise SystemExit(main())
