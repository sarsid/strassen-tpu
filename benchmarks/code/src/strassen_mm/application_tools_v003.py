"""Standard archived-run adapter for the isolated protobuf dependency recovery installer."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from . import benchmark_v001 as base


def group_exists(pgid):
    try:
        os.killpg(pgid,0)
        return True
    except ProcessLookupError:return False
    except PermissionError:return True


def cleanup_group(child):
    """Bounded cleanup includes orphan pip --python descendants after leader exit."""
    pgid=child.pid;record={'pgid':pgid,'signals':[],'proven_stopped':False}
    started=time.monotonic()
    try:
        for sig in (signal.SIGTERM,signal.SIGKILL):
            child.poll()
            if not group_exists(pgid):break
            try:os.killpg(pgid,sig);record['signals'].append(sig.name)
            except ProcessLookupError:break
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                child.poll()
                if not group_exists(pgid):break
                time.sleep(.05)
        child.poll()
        record['leader_returncode']=child.returncode
        record['proven_stopped']=not group_exists(pgid)
    except BaseException as problem:
        record['error']={'type':type(problem).__name__,'message':str(problem)}
    record['elapsed_seconds']=time.monotonic()-started
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','phase','output-dir','expected-identity','allocation-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=5400)
    p.add_argument('--venv-dir',default='/content/Strassen_MM_Focus/model-tools-v003')
    args=p.parse_args(argv);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    journal=base.Journal(out,args.phase);started=time.monotonic();status='failed';error=None;record=None;child=None
    cleanup={'proven_stopped':True,'reason':'Installer not launched'}
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        raise InterruptedError('Model-tools adapter received '+signal.Signals(signum).name)
    for sig in previous_handlers:signal.signal(sig,interrupted)
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
        tool=Path(__file__).resolve().parents[2]/'tools/install_model_tools_v003.py'
        command=[sys.executable,str(tool),'--output-dir',str(out/'preparation-00'),
                 '--venv-dir',args.venv_dir,'--max-wall-seconds',str(args.max_wall_seconds),
                 '--managed-process-group']
        base.exclusive_json(out/'command-00.json',{'argv':command})
        with (out/'preparation-00.log').open('xb') as log:
            # Close the small signal-delivery window between creating the
            # session and retaining its PID for guaranteed cleanup.
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous_handlers))
            try:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                base.exclusive_json(out/'installer-process.json',{'pid':child.pid,'pgid':child.pid,
                    'policy':'All installer/pip descendants inherit one adapter-owned group; no nested sessions'})
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
            child.wait(timeout=args.max_wall_seconds+10)
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
        # The outer launcher sends TERM, then allows 30 seconds before KILL.
        # Our at-most-ten-second cleanup runs even when TERM interrupted wait.
        # Repeated TERM/INT cannot interrupt group cleanup halfway through.
        for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
        if child is not None:cleanup=cleanup_group(child)
        base.exclusive_json(out/'subprocess-cleanup.json',cleanup)
        if not cleanup['proven_stopped']:
            status='failed'
            error={'type':'UncertainDescendantState','message':'Installer/pip process group could not be proven stopped'}
        summary={'phase':args.phase,'status':status,'completed':status=='completed','record':record,
                 'error':error,'elapsed_seconds':time.monotonic()-started,'jax_imported':'jax' in sys.modules,
                 'subprocess_cleanup_all_proven':cleanup['proven_stopped'],'subprocess_cleanup':cleanup}
        journal.emit('run_complete',**summary);journal.close();base.exclusive_json(out/'summary.json',summary)
        base.exclusive_json(out/'artifact_manifest.json',{'sha256':{str(path.relative_to(out)):base.digest_file(path)
              for path in sorted(out.rglob('*')) if path.is_file()},'sealed_utc':base.utc_now()})
        print(json.dumps({'phase':args.phase,'status':status,'error':error}),flush=True)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

