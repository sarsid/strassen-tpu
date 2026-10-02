"""Retrieve and verify completed setup after a transport-only failure."""
import argparse,hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('session','endpoint','remote-output','controller-python'):p.add_argument('--'+n,required=True)
a=p.parse_args();run=Path(os.environ['STRASSEN_EXECUTION_DIR']);out=run/'artifacts';out.mkdir()
source=Path(__file__).resolve().parents[1];archive=out/'remote-setup.tar.gz'
command=[a.controller_python,str(source/'runtime/colab_control_v003.py'),'download','--session',a.session,
    '--expect-endpoint',a.endpoint,'--remote',a.remote_output+'.tar.gz','--local',str(archive)]
with (out/'download.log').open('x')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
with tarfile.open(archive)as t:t.extractall(out,filter='data')
setup=out/Path(a.remote_output).name
manifest=json.loads((setup/'artifact-manifest.json').read_text())
for name,entry in manifest.items():
    f=(setup/name).resolve();assert f.is_relative_to(setup.resolve())
    assert f.stat().st_size==entry['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
status=json.loads((setup/'status.json').read_text());identity=json.loads((setup/'identity.json').read_text())
assert status['status']=='completed' and identity['colab_endpoint']==a.endpoint
assert identity['backend']=='tpu' and identity['device_count']==1 and identity['devices'][0]['kind']=='TPU v6 lite'
for name,version in {'jax':'0.11.2','jaxlib':'0.11.2','libtpu':'0.0.48'}.items():assert identity['versions'][name]==version
result=dict(completed=True,recovery_only=True,files_verified=len(manifest),remote_identity=a.remote_output+'/identity.json',
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),identity=identity)
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
