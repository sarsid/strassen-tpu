#!/usr/bin/env python3
"""Create model-tools-v003 with the diagnosed pinned protobuf dependency.

All v002 explicit pins and the complete installed inventory remain unchanged
apart from protobuf6.32.1. A different transitive resolution fails validation.

Official pip documents --python for an environment created --without-pip:
https://pip.pypa.io/en/stable/topics/python-option/ (added in pip22.3).
Invoke through application_tools_v003, which owns a dedicated process group
and verifies that pip descendants stop, including after timeout or termination.
This installs only into the explicit new interpreter. Global core packages and
the failed model-tools-v001 path are never installation targets or modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PINS=['transformers==4.56.2','tokenizers==0.22.0','huggingface-hub==0.34.4',
      'safetensors==0.6.2','numpy==2.2.6','pyarrow==21.0.0','sentencepiece==0.2.1','accelerate==1.10.1','protobuf==6.32.1']
CORE=('jax','jaxlib','libtpu','numpy','ml_dtypes','scipy','requests','pip')
PROJECT=Path('/content/Strassen_MM_Focus')
BASELINE_INVENTORY=PROJECT/'runs/20260919T074551Z-N9-model-tools-v002-v5e-v004-34afbd/artifacts/preparation-00/installed.json'
BASELINE_INVENTORY_SHA256='733c8bbdf054ddb3cc4228d95d8e00f5e94aafb370f72c3f7410ebc9030c7a46'


def utc(): return datetime.now(timezone.utc).isoformat()


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def core_versions():
    result={}
    for name in CORE:
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result



def compare_inventory(before,after):
    """The only allowed environment delta is the diagnosed pinned dependency."""
    old={row['name'].lower().replace('_','-'):row['version'] for row in before['all_distributions']}
    new={row['name'].lower().replace('_','-'):row['version'] for row in after['all_distributions']}
    added={name:new[name] for name in sorted(new.keys()-old.keys())}
    removed={name:old[name] for name in sorted(old.keys()-new.keys())}
    changed={name:{'before':old[name],'after':new[name]} for name in sorted(old.keys()&new.keys()) if old[name]!=new[name]}
    return {'added':added,'removed':removed,'changed':changed,
            'only_pinned_protobuf_added':added=={'protobuf':'6.32.1'} and not removed and not changed}


def validate_new_target(path):
    target=Path(path).absolute()
    if target.exists() or target.is_symlink(): raise FileExistsError('Refusing to reuse an existing or partial environment')
    if not target.resolve().is_relative_to(PROJECT) or target.name in ('model-tools-v001','model-tools-v002'):
        raise ValueError('New environment must be inside the authorized project and preserve model-tools-v001 and model-tools-v002')
    return target


def installation_commands(global_python,target,output):
    """Every pip mutation has an explicit target interpreter before install."""
    target_python=str(Path(target)/'bin/python')
    prefix=[str(global_python),'-m','pip','--isolated','--python',target_python]
    options=['install','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:']
    return [
      [str(global_python),'-m','venv','--without-pip',str(target)],
      prefix+options+['--index-url','https://download.pytorch.org/whl/cpu',
                      '--report',str(Path(output)/'torch_install_report.json'),'torch==2.8.0'],
      prefix+options+['--index-url','https://pypi.org/simple',
                      '--report',str(Path(output)/'tools_install_report.json'),*PINS],
      prefix+['check'],
      prefix+['freeze','--all'],
    ]


PREFIX_PROBE="""import json,site,sys
from pathlib import Path
target=Path(sys.argv[1]).resolve()
value={'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable,
       'user_site_enabled':site.ENABLE_USER_SITE}
with Path(sys.argv[2]).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
assert Path(sys.prefix).resolve()==target and sys.prefix!=sys.base_prefix
assert site.ENABLE_USER_SITE is False
"""

FINAL_PROBE="""import importlib,importlib.metadata as md,importlib.util,json,sys
from pathlib import Path
target=Path(sys.argv[1]).resolve()
names=['torch','transformers','tokenizers','huggingface-hub','safetensors','numpy','pyarrow','sentencepiece','accelerate','protobuf']
versions={name:md.version(name) for name in names}
modules={}
for name in ['torch','transformers','tokenizers','huggingface_hub','safetensors','numpy','pyarrow','sentencepiece','accelerate','google.protobuf']:
    module=importlib.import_module(name);modules[name]=str(Path(module.__file__).resolve())
    assert Path(module.__file__).resolve().is_relative_to(target), name+' imported outside target environment'
import torch
import google.protobuf
assert google.protobuf.__version__=='6.32.1','Expected pinned protobuf runtime'
assert torch.version.cuda is None,'Expected CPU-only PyTorch distribution'
assert importlib.util.find_spec('jax') is None,'Isolated model tools must not expose core JAX'
value={'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':sys.version,'versions':versions,
       'module_paths':modules,'torch_cuda_version':torch.version.cuda,'jax_visible':False,
       'all_distributions':sorted([{'name':d.metadata['Name'],'version':d.version} for d in md.distributions()],key=lambda x:x['name'].lower())}
with Path(sys.argv[2]).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
"""


def main(argv=None):
    # The adapter masks TERM/INT only while creating and recording our group.
    # Unblock the inherited mask before launching any package subprocesses.
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--venv-dir',type=Path,default=PROJECT/'model-tools-v003')
    p.add_argument('--max-wall-seconds',type=float,default=5400)
    p.add_argument('--managed-process-group',action='store_true')
    args=p.parse_args(argv)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+args.max_wall_seconds
    before=core_versions();commands=[];result=None;error=None;status='failed';descendants_uncertain=False
    env={key:value for key,value in os.environ.items() if not key.startswith('PIP_')
         and key not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')}
    env.update(PIP_CONFIG_FILE=os.devnull,PYTHONDONTWRITEBYTECODE='1')
    def run(label,command,limit=1800):
        nonlocal descendants_uncertain
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Model-tools installation wall budget exhausted')
        index=len(commands);record={'index':index,'label':label,'argv':command,'started_utc':utc(),
                                  'timeout_seconds':min(limit,remaining)}
        save(out/f'command-{index:02d}.json',record)
        start=time.monotonic()
        try:
            with (out/f'command-{index:02d}.stdout.log').open('xb') as stdout, (out/f'command-{index:02d}.stderr.log').open('xb') as stderr:
                child=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,timeout=record['timeout_seconds'],check=False)
            record['returncode']=child.returncode
            if child.returncode:raise RuntimeError('Installation command failed: '+label)
        except BaseException as problem:
            # pip --python may leave its target interpreter alive when the
            # direct pip process is killed. All descendants inherit our group;
            # the owning adapter always terminates and verifies that group.
            if isinstance(problem,subprocess.TimeoutExpired):descendants_uncertain=True
            record['error_type']=type(problem).__name__;raise
        finally:
            record.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-start)
            save(out/f'command-{index:02d}.result.json',record);commands.append(record)
    try:
        if args.max_wall_seconds<=0:raise ValueError('Positive wall budget required')
        if not args.managed_process_group or os.getpgrp()!=os.getpid():
            raise RuntimeError('Run through application_tools_v003; adapter-owned process group required')
        if 'jax' in sys.modules:raise RuntimeError('Installer must run without importing JAX')
        if digest(BASELINE_INVENTORY)!=BASELINE_INVENTORY_SHA256:
            raise RuntimeError('Prior isolated package inventory differs from sealed v002 evidence')
        baseline=json.loads(BASELINE_INVENTORY.read_text())
        with (out/'baseline_inventory.json').open('xb') as stream:stream.write(BASELINE_INVENTORY.read_bytes())
        target=validate_new_target(args.venv_dir)
        version=re.match(r'^(\d+)\.(\d+)',before.get('pip') or '')
        if not version or tuple(map(int,version.groups()))<(22,3):
            raise RuntimeError('Existing global pip22.3+ required; global pip will not be upgraded')
        save(out/'before.json',{'global_python':sys.executable,'core_versions':before,'target':str(target),
            'strategy':'venv --without-pip; existing pip --isolated --python TARGET install',
            'official_documentation':'https://pip.pypa.io/en/stable/topics/python-option/',
            'configuration_documentation':'https://pip.pypa.io/en/stable/topics/configuration/',
            'pip_config_policy':'PIP_CONFIG_FILE=os.devnull, isolated mode, explicit public indices, no pip cache',
            'pins':['torch==2.8.0',*PINS]})
        planned=installation_commands(sys.executable,target,out);save(out/'planned_commands.json',planned)
        run('create_without_pip',planned[0],120)
        run('verify_new_prefix',[str(target/'bin/python'),'-I','-c',PREFIX_PROBE,str(target),str(out/'prefix_probe.json')],60)
        cfg=(target/'pyvenv.cfg').read_text()
        if not re.search(r'^include-system-site-packages\s*=\s*false\s*$',cfg,re.I|re.M):
            raise RuntimeError('Target environment unexpectedly includes global site packages')
        for label,command in zip(('cpu_torch','pinned_tools','pip_check','pip_freeze'),planned[1:]):run(label,command)
        run('verify_installed_imports',[str(target/'bin/python'),'-I','-c',FINAL_PROBE,str(target),str(out/'installed.json')],300)
        installed=json.loads((out/'installed.json').read_text())
        for pin in ['torch==2.8.0',*PINS]:
            name,wanted=pin.split('==',1);actual=installed['versions'][name]
            if (actual.split('+',1)[0] if name=='torch' else actual)!=wanted:
                raise RuntimeError('Pinned model-tool version mismatch: '+name)
        comparison=compare_inventory(baseline,installed)
        save(out/'inventory_comparison.json',comparison)
        if not comparison['only_pinned_protobuf_added']:
            raise RuntimeError('Fresh environment changed packages beyond pinned protobuf; inspect inventories')
        after=core_versions()
        if before!=after:raise RuntimeError('Global core package versions changed during isolated installation')
        result={'venv_dir':str(target),'python':str(target/'bin/python'),'pins':['torch==2.8.0',*PINS],
                'versions':installed['versions'],'core_jax_environment_modified':False,
                'global_core_before':before,'global_core_after':after,'created_utc':utc(),
                'strategy':'pip --python into fresh --without-pip venv; ensurepip bypassed; protobuf6.32.1 only addition',
                'baseline_inventory_sha256':BASELINE_INVENTORY_SHA256,
                'installed_inventory_sha256':digest(out/'installed.json'),'inventory_comparison':comparison,
                'installation_reports':{'torch_install_report.json':digest(out/'torch_install_report.json'),
                                        'tools_install_report.json':digest(out/'tools_install_report.json')}}
        save(out/'result.json',result);save(target/'installation_manifest.json',result);status='completed'
    except BaseException as problem:
        error={'type':type(problem).__name__,'message':str(problem)}
    finally:
        after=core_versions();save(out/'after.json',{'core_versions':after,'unchanged':before==after})
        summary={'action':'model-tools-v003','status':status,'started_monotonic':started,
                 'finished_utc':utc(),'elapsed_seconds':time.monotonic()-started,
                 'core_versions_unchanged':before==after,'error':error,'commands':commands}
        summary.update(process_group_id=os.getpgrp(),descendant_state_requires_adapter_verification=True,
                       timeout_descendants_may_still_be_running=descendants_uncertain,
                       child_snapshot_scope='Provisional until adapter process-group cleanup; adapter artifact_manifest.json seals final file bytes')
        save(out/'status.json',summary)
        save(out/'artifact_manifest.json',{'scope':'Provisional child snapshot; authoritative final bytes are sealed by adapter after process-group cleanup',
            'files':{str(path.relative_to(out)):{'bytes':path.stat().st_size,'sha256':digest(path)}
            for path in sorted(out.rglob('*')) if path.is_file()}})
        print(json.dumps({'action':'model-tools-v003','status':status,'venv_dir':str(args.venv_dir),
                          'core_versions_unchanged':before==after,'error':error}),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
