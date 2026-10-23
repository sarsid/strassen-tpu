"""Isolated CPU preparation with ensurepip bypass and bounded child cleanup.

Use only in a known-idle runtime under the existing launcher lock. No JAX import,
device probe, global package installation, cache replacement or credential log.
The executed v001 preparation and its partial cpu-tools directory stay intact.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from strassen_mm import benchmark_v001 as base
from strassen_mm.application_tools_v003 import cleanup_group

PRIVATE_BASE=Path('/content/Strassen_MM_Focus/.runtime_private')
UNBLOCK_EXEC="""import os,signal,sys
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
"""


def target_paths(private_root,venv_dir=None):
    supplied=Path(private_root)
    if not supplied.is_absolute() or '..' in supplied.parts:raise ValueError('Absolute private root required')
    root=supplied.resolve()
    if not root.is_relative_to(PRIVATE_BASE) or root==PRIVATE_BASE:raise ValueError('Dedicated .runtime_private root required')
    target=Path(venv_dir) if venv_dir is not None else root/'cpu-tools-v002'
    if not target.is_absolute() or '..' in target.parts or target.name!='cpu-tools-v002':raise ValueError('Expected cpu-tools-v002 target')
    if target.is_symlink():raise ValueError('Environment target must not be a symlink')
    target=target.resolve()
    if not target.is_relative_to(root):raise ValueError('Environment escaped its private root')
    return root,target


def host_identity(endpoint,expected):
    return dict(colab_endpoint=endpoint,hostname=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                versions={name:metadata.version(name) for name in expected['versions']})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity','private-root'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('phase','allocation-id'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=3600)
    parser.add_argument('--action',choices=('tools','inputs'),required=True)
    parser.add_argument('--model-key',choices=('qwen','mistral','gemma'))
    parser.add_argument('--venv-dir',type=Path)
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    journal=base.Journal(out,args.phase);started=time.monotonic();deadline=started+args.max_wall_seconds
    error=None;completed=False;current=None;expected=None;commands=[];cleanups=[];children=[];tool_result=None
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):raise InterruptedError('Preparation received '+signal.Signals(signum).name)
    for sig in previous:signal.signal(sig,interrupted)

    def run(command,environment,label):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Preparation budget exhausted')
        index=len(commands);record=dict(label=label,argv=command,timeout_seconds=remaining,started_utc=base.utc_now())
        commands.append(record);base.exclusive_json(out/f'command-{index:02d}.json',record)
        journal.emit('preparation_step',index=index+1,action=args.action,model_key=args.model_key,step=label)
        child=None
        with (out/f'command-{index:02d}.log').open('xb') as log:
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
            try:
                # The bootstrap unmasks TERM/INT before exec, keeping the same
                # PID/group. All pip --python descendants inherit this group.
                actual=[command[0],'-c',UNBLOCK_EXEC,*command[1:]]
                child=subprocess.Popen(actual,env=environment,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                base.exclusive_json(out/f'process-{index:02d}.json',dict(pid=child.pid,pgid=child.pid,argv=actual))
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
            try:
                child.wait(timeout=remaining)
                if child.returncode:raise RuntimeError('CPU preparation command failed: '+label)
            finally:
                record.update(returncode=child.poll(),finished_utc=base.utc_now())
                base.exclusive_json(out/f'command-{index:02d}.result.json',record)
                cleanup=cleanup_group(child);cleanups.append(cleanup)
                if not cleanup['proven_stopped']:raise RuntimeError('CPU preparation descendants could not be proven stopped')
                children.remove(child)

    try:
        if args.max_wall_seconds<=0:raise ValueError('Positive wall budget required')
        if 'jax' in sys.modules:raise RuntimeError('CPU preparation must not import JAX')
        root,target=target_paths(args.private_root,args.venv_dir)
        expected=json.loads(args.expected_identity.read_text());expected=expected.get('identity',expected)
        current=host_identity(args.allocation_id,expected)
        base.exclusive_json(out/'environment.json',dict(identity=current,
             qualification='Read-only host/core metadata; device qualification inherited from recorded allocation, no JAX/device probe'))
        for key in ('colab_endpoint','hostname','boot_id','versions'):
            if current[key]!=expected[key]:raise RuntimeError('Preparation host/core identity mismatch: '+key)
        journal.emit('identity_check',status='matched',scope='host/core metadata without JAX')
        cfg=json.loads(args.campaign.read_text())
        base.snapshot_sources(out,args.campaign,args.campaign.parent/cfg['shape_manifest'],args.campaign.parent/cfg['distribution_manifest'])
        source=Path(__file__).resolve().parents[2];tools=source/'tools'
        sys.path.insert(0,str(tools))
        from install_mlsys_model_tools_v001 import PINS,PREFIX_PROBE,install_environment,digest
        env=install_environment()
        if args.action=='tools':
            command=[sys.executable,str(tools/'install_mlsys_model_tools_v001.py'),
                     '--private-root',str(root),'--venv-dir',str(target),'--output-dir',str(out/'installation'),
                     '--max-wall-seconds',str(max(1,deadline-time.monotonic()-15)),'--managed-process-group']
            run(command,env,'isolated_cpu_tools_bootstrap')
            status=json.loads((out/'installation/status.json').read_text())
            if status.get('status')!='completed' or status.get('core_versions_unchanged') is not True:
                raise RuntimeError('CPU-tools bootstrap did not qualify')
            tool_result=json.loads((out/'installation/result.json').read_text())
            base.exclusive_json(out/'tools_result.json',tool_result)
        else:
            if args.model_key is None:raise ValueError('Model key required for inputs')
            manifest_path=target/'installation_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            if (manifest.get('status')!='completed' or manifest.get('venv_dir')!=str(target)
                    or manifest.get('python')!=str(target/'bin/python') or manifest.get('pins')!=['torch==2.8.0',*PINS]
                    or manifest.get('core_versions_unchanged') is not True):
                raise RuntimeError('CPU-tools installation manifest is incompatible')
            base.exclusive_json(out/'tools_input_provenance.json',dict(path=str(manifest_path),sha256=digest(manifest_path),manifest=manifest))
            run([str(target/'bin/python'),'-I','-c',PREFIX_PROBE,str(target),str(out/'prefix_probe.json')],env,'verify_input_tools_prefix')
            token_file=root/'hf_token'
            if token_file.is_file():env['HF_TOKEN']=token_file.read_text().strip()
            run([str(target/'bin/python'),str(tools/'prepare_large_models_v001.py'),
                 '--model-key',args.model_key,'--output-dir',str(out/'prepared'),
                 '--cache-root',str(root/(args.model_key+'-cache-v002'))],env,'prepare_model_inputs')
            bundle=json.loads((out/'prepared/input_bundle.json').read_text())
            if bundle.get('status')!='ready':raise RuntimeError('Model inputs unavailable; preserved input bundle describes the failure')
        if host_identity(args.allocation_id,expected)!=current:raise RuntimeError('Host/core metadata changed during CPU preparation')
        completed=True
    except BaseException as exc:
        # Third-party acquisition errors may contain signed URLs or credentials.
        error=dict(type=type(exc).__name__);journal.emit('preparation_error',**error)
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        for child in children:
            cleanup=cleanup_group(child);cleanups.append(cleanup)
        proven=all(c.get('proven_stopped') is True for c in cleanups)
        if not proven:completed=False;error=dict(type='UncertainDescendantState')
        unchanged=None
        if current is not None:
            try:
                after=host_identity(args.allocation_id,expected);unchanged=after==current
                base.exclusive_json(out/'after_identity.json',dict(identity=after,unchanged=unchanged))
                if not unchanged:completed=False;error=dict(type='HostCoreMetadataChanged')
            except Exception:
                completed=False;error=dict(type='HostCoreMetadataUnavailable')
        base.exclusive_json(out/'subprocess-cleanup.json',dict(all_proven=proven,groups=cleanups))
        summary=dict(completed=completed,status='completed' if completed else 'failed',error=error,
            action=args.action,model_key=args.model_key,core_metadata_unchanged=unchanged,
            subprocess_cleanup_all_proven=proven,jax_imported='jax' in sys.modules,
            elapsed_seconds=time.monotonic()-started,commands=commands,
            interpretation='CPU preparation only; no TPU or model performance measurements')
        if completed and args.action=='inputs':summary['input_bundle']='prepared/input_bundle.json'
        if tool_result:summary['tools_result']='tools_result.json'
        journal.emit('run_complete',**summary);journal.close();base.exclusive_json(out/'summary.json',summary)
        base.exclusive_json(out/'artifact_manifest.json',dict(sha256={str(path.relative_to(out)):base.digest_file(path)
            for path in sorted(out.rglob('*')) if path.is_file()},sealed_utc=base.utc_now()))
        for sig,handler in previous.items():signal.signal(sig,handler)
    return 0 if completed else 1


if __name__=='__main__':raise SystemExit(main())
