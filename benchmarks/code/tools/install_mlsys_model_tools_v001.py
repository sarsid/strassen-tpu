"""Fresh isolated MLSys CPU tools, bypassing ensurepip without global installs.

Only pure command/probe helpers are reused from the preserved v003 installer.
Its historical-inventory checks and main() are never invoked. Run this installer
through benchmark_mlsys_prepare_v002, which owns and cleans its process group.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from install_model_tools_v003 import (PINS, PREFIX_PROBE, FINAL_PROBE,
    core_versions, digest, installation_commands, save, utc)

PRIVATE_BASE=Path('/content/Strassen_MM_Focus/.runtime_private')


def validate_private_root(path):
    root=Path(path)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('An absolute private project root is required')
    root=root.resolve()
    if not root.is_relative_to(PRIVATE_BASE) or root==PRIVATE_BASE:
        raise ValueError('A dedicated directory beneath .runtime_private is required')
    return root


def validate_new_target(path,private_root):
    root=validate_private_root(private_root);target=Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError('Preserve the existing or partial environment; choose a new versioned target')
    if not target.is_absolute() or '..' in target.parts or target.name!='cpu-tools-v002':
        raise ValueError('Expected a fresh cpu-tools-v002 environment')
    target=target.resolve()
    if not target.is_relative_to(root):raise ValueError('Environment must be inside its private project root')
    return target


def install_environment(source=None):
    """Installer needs no HF credentials, inherited pip policy or global paths."""
    original=os.environ if source is None else source
    env={key:value for key,value in original.items()
         if not key.startswith('PIP_') and key not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV',
              'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
    env.update(PIP_CONFIG_FILE=os.devnull,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env


def main(argv=None):
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--venv-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=3600)
    parser.add_argument('--managed-process-group',action='store_true')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+args.max_wall_seconds
    before=core_versions();commands=[];error=None;status='failed';descendants_uncertain=False
    env=install_environment()

    def run(label,command,limit=1800):
        nonlocal descendants_uncertain
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('CPU-tools bootstrap wall budget exhausted')
        index=len(commands);record=dict(label=label,argv=command,started_utc=utc(),timeout_seconds=min(limit,remaining))
        save(out/f'command-{index:02d}.json',record)
        try:
            with (out/f'command-{index:02d}.stdout.log').open('xb') as stdout,(out/f'command-{index:02d}.stderr.log').open('xb') as stderr:
                child=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     timeout=record['timeout_seconds'])
            record['returncode']=child.returncode
            if child.returncode:raise RuntimeError('CPU-tools command failed: '+label)
        except BaseException as exc:
            if isinstance(exc,subprocess.TimeoutExpired):descendants_uncertain=True
            record['error_type']=type(exc).__name__;raise
        finally:
            record['finished_utc']=utc();save(out/f'command-{index:02d}.result.json',record);commands.append(record)

    try:
        if args.max_wall_seconds<=0:raise ValueError('Positive wall budget required')
        if not args.managed_process_group or os.getpgrp()!=os.getpid():
            raise RuntimeError('Adapter-owned process group is required')
        if 'jax' in sys.modules:raise RuntimeError('CPU-tools installer must not initialize JAX')
        target=validate_new_target(args.venv_dir,args.private_root)
        version=re.match(r'^(\d+)\.(\d+)',before.get('pip') or '')
        if not version or tuple(map(int,version.groups()))<(22,3):
            raise RuntimeError('Existing global pip22.3+ required; global pip will not be installed or upgraded')
        save(out/'before.json',dict(global_python=sys.executable,core_versions=before,target=str(target),
             pins=['torch==2.8.0',*PINS],strategy='venv --without-pip; existing pip --isolated --python TARGET',
             documentation='https://pip.pypa.io/en/stable/topics/python-option/',
             helper_sha256=digest(Path(__file__).with_name('install_model_tools_v003.py'))))
        planned=installation_commands(sys.executable,target,out);save(out/'planned_commands.json',planned)
        run('create_without_pip',planned[0],120)
        run('verify_prefix',[str(target/'bin/python'),'-I','-c',PREFIX_PROBE,str(target),str(out/'prefix_probe.json')],60)
        if not re.search(r'^include-system-site-packages\s*=\s*false\s*$',(target/'pyvenv.cfg').read_text(),re.I|re.M):
            raise RuntimeError('Environment exposes system packages')
        for label,command in zip(('cpu_torch','pinned_tools','pip_check','pip_freeze'),planned[1:]):run(label,command)
        run('verify_imports',[str(target/'bin/python'),'-I','-c',FINAL_PROBE,str(target),str(out/'installed.json')],300)
        installed=json.loads((out/'installed.json').read_text())
        for pin in ['torch==2.8.0',*PINS]:
            name,wanted=pin.split('==',1);actual=installed['versions'][name]
            if (actual.split('+',1)[0] if name=='torch' else actual)!=wanted:
                raise RuntimeError('Pinned version mismatch: '+name)
        if core_versions()!=before:raise RuntimeError('Global core package metadata changed')
        result=dict(status='completed',venv_dir=str(target),python=str(target/'bin/python'),
                    pins=['torch==2.8.0',*PINS],versions=installed['versions'],
                    installed_inventory_sha256=digest(out/'installed.json'),global_core_before=before,
                    global_core_after=core_versions(),core_versions_unchanged=True,
                    strategy='Fresh cpu-tools-v002; ensurepip bypassed; no historical inventory dependency',
                    created_utc=utc(),installation_reports={name:digest(out/name) for name in
                        ('torch_install_report.json','tools_install_report.json')})
        save(out/'result.json',result);save(target/'installation_manifest.json',result);status='completed'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        after=core_versions();unchanged=before==after
        if not unchanged:status='failed';error=dict(type='CoreMetadataChanged',message='Global core package metadata changed')
        save(out/'after.json',dict(core_versions=after,unchanged=unchanged))
        summary=dict(status=status,error=error,commands=commands,core_versions_unchanged=unchanged,
                     process_group_id=os.getpgrp(),timeout_descendants_may_still_be_running=descendants_uncertain,
                     finished_utc=utc(),elapsed_seconds=time.monotonic()-started,
                     child_snapshot_scope='Provisional until adapter cleanup; adapter artifact_manifest.json seals final bytes')
        save(out/'status.json',summary)
        save(out/'artifact_manifest.json',dict(scope=summary['child_snapshot_scope'],sha256={
            str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        print(json.dumps(dict(status=status,error=error,venv_dir=str(args.venv_dir),core_versions_unchanged=unchanged)),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
