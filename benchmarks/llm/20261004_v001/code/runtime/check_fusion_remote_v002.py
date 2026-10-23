"""Execute the frozen qualification in a subprocess and retain its outcomes."""
import argparse, hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if not a.output.is_relative_to('/content/Strassen_MM_Focus') or '..' in a.output.parts:
        raise ValueError('Invalid qualification path')
    assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==a.sha256
    a.output.mkdir(exist_ok=False)
    source=a.output/'source';source.mkdir()
    with tarfile.open(a.archive) as t:t.extractall(source,filter='data')
    env=dict(os.environ,PYTHONPATH=str(source/'src'),PYTHONUNBUFFERED='1',JAX_PLATFORMS='tpu')
    command=[sys.executable,str(source/'tools/check_fused_tpu_v002.py'),'--output-dir',str(a.output/'artifacts')]
    status=dict(command=command,source_archive_sha256=a.sha256)
    try:
        with (a.output/'execution.log').open('x') as log:
            child=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        status['returncode']=child.returncode
    except Exception as exc:
        status.update(returncode=-1,error_type=type(exc).__name__)
    (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    archive=a.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(a.output,arcname=a.output.name)
    print(json.dumps(dict(kind='fusion_check_finished',archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),**status)),flush=True)

if __name__=='__main__':main()
