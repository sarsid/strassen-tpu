"""Launch the frozen Gemma worker after resolving its bundled imports.

The Colab exec-file RPC executes a source string without sibling modules on
sys.path. This standard-library-only entry point verifies/extracts the bundle
before invoking its original launcher as a normal Python module. No numerical
or experiment source is changed. Existing remote runs are never relaunched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import tarfile

BASE=Path('/content/Strassen_MM_Focus')


def launch(archive,sha256,run_id,base=BASE):
    archive=Path(archive).resolve();base=Path(base).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',run_id):raise ValueError('Invalid run ID')
    if not archive.is_relative_to(base) or hashlib.sha256(archive.read_bytes()).hexdigest()!=sha256:
        raise ValueError('Invalid source archive')
    if (base/'runs'/run_id).exists():raise FileExistsError('Remote run already exists; do not launch another worker')
    staging=base/'launchers'/(run_id+'-v002');staging.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as handle:handle.extractall(staging,filter='data')
    runtime=staging/'runtime';script=runtime/'gemma4_batch_remote_v001.py'
    if not script.is_file():raise ValueError('Frozen worker missing from bundle')
    old_path=list(sys.path);old_argv=list(sys.argv)
    try:
        sys.path.insert(0,str(runtime))
        sys.argv=[str(script),'--archive',str(archive),'--sha256',sha256,'--run-id',run_id]
        runpy.run_path(str(script),run_name='__main__')
    finally:
        sys.path[:]=old_path;sys.argv[:]=old_argv


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    launch(a.archive,a.sha256,a.run_id)


if __name__=='__main__':main()
