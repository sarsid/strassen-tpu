"""Prepare isolated CPU model tooling/inputs while the TPU measurement lock is held."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from strassen_mm import benchmark_v001 as base

PINS=['transformers==4.56.2','tokenizers==0.22.0','huggingface-hub==0.34.4',
      'safetensors==0.6.2','numpy==2.2.6','pyarrow==21.0.0','sentencepiece==0.2.1',
      'accelerate==1.10.1','protobuf==6.32.1']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-identity',type=Path,required=True)
    p.add_argument('--allocation-id',required=True);p.add_argument('--max-wall-seconds',type=float,default=3600)
    p.add_argument('--action',choices=['tools','inputs'],required=True);p.add_argument('--model-key',choices=['qwen','mistral','gemma'])
    p.add_argument('--private-root',type=Path,required=True)
    args=p.parse_args();root=args.private_root.resolve()
    if not root.is_relative_to('/content/Strassen_MM_Focus/.runtime_private'):raise ValueError('Private model cache path required')
    args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir;journal=base.Journal(out,args.phase)
    cfg=json.loads(args.campaign.read_text());error=None;started=time.monotonic();summary={}
    try:
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from strassen_mm.kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,cfg,enable_qualified_mosaic_v7_compat());base.exclusive_json(out/'environment.json',env)
        journal.emit('identity_check',**base.verify_identity(env,args.expected_identity,cfg))
        base.snapshot_sources(out,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        root.mkdir(parents=True,exist_ok=True);venv=root/'cpu-tools';commands=[]
        if args.action=='tools':
            if venv.exists():raise FileExistsError('CPU tools environment already exists; no overwrite')
            commands=[[sys.executable,'-m','venv',str(venv)],
                      [str(venv/'bin/python'),'-m','pip','install','--disable-pip-version-check','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'],
                      [str(venv/'bin/python'),'-m','pip','install','--disable-pip-version-check',*PINS]]
        else:
            if not args.model_key:raise ValueError('Model key required')
            source=Path(__file__).resolve().parents[2]
            commands=[[str(venv/'bin/python'),str(source/'tools/prepare_mlsys_models_v001.py'),'--model-key',args.model_key,
                       '--output-dir',str(out/'prepared'),'--cache-root',str(root/(args.model_key+'-cache'))]]
        base.exclusive_json(out/'commands.json',commands)
        child_env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        token_file=root/'hf_token'
        if token_file.is_file():child_env['HF_TOKEN']=token_file.read_text().strip()
        for i,command in enumerate(commands):
            remaining=args.max_wall_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Preparation budget exhausted')
            journal.emit('preparation_step',index=i+1,total=len(commands),action=args.action,model_key=args.model_key)
            with (out/f'command-{i:02d}.log').open('x') as log:
                done=subprocess.run(command,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            if done.returncode:raise RuntimeError('Preparation subprocess failed; inspect preserved sanitized acquisition records')
        if args.action=='inputs':
            bundle=json.loads((out/'prepared/input_bundle.json').read_text())
            if bundle['status']!='ready':raise RuntimeError('Model inputs unavailable')
            summary['input_bundle']='prepared/input_bundle.json'
    except BaseException as exc:
        # Do not archive third-party exception strings, which may contain URLs/tokens.
        error=dict(type=type(exc).__name__);journal.emit('preparation_error',**error)
    summary.update(completed=error is None,error=error,action=args.action,model_key=args.model_key,
                   interpretation='CPU input/tool preparation; no TPU performance measurements',case_status_counts={})
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(main())
