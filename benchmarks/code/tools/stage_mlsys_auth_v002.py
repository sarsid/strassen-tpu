"""Stage the existing Hugging Face credential privately; never archive its value."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import importlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--endpoint',required=True)
    p.add_argument('--private-root',required=True);p.add_argument('--hardware',choices=['v5e','v6e'],required=True);args=p.parse_args()
    version='v003' if args.hardware=='v6e' else 'v002'
    control=importlib.import_module('colab_control_'+version)
    private=control.remote_path(args.private_root)
    if not private.startswith('/content/Strassen_MM_Focus/.runtime_private/'):raise ValueError('Private path required')
    token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        paths=[Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token',Path.home()/'.huggingface/token']
        for path in paths:
            if path.is_file():token=path.read_text().strip();break
    if token:control.remember_secret(token)
    common=[sys.executable,str(ROOT/'runtime'/('colab_control_'+version+'.py')),'exec-file','--session',args.session,
            '--expect-endpoint',args.endpoint,'--file',str(ROOT/'runtime/prepare_mlsys_private_v001.py'),
            '--timeout','60','--script-arg=--private-root','--script-arg='+private]
    subprocess.run(common,check=True)
    if token:
        namespace=SimpleNamespace(config=control.DEFAULT_CONFIG,session=args.session,expect_endpoint=args.endpoint)
        with control.api(control.DEFAULT_TOKEN) as client:
            _,saved=control.refreshed_session(namespace,client)
            remote=private+'/hf_token'
            try:control.content_request(saved,'GET',remote)
            except Exception as exc:
                if getattr(getattr(exc,'response',None),'status_code',None)!=404:raise
            else:raise FileExistsError('Credential destination exists')
            control.content_request(saved,'PUT',remote,payload={'name':'hf_token','path':remote,'type':'file',
                'format':'base64','content':base64.b64encode(token.encode()).decode(),'chunk':1})
        subprocess.run(common+['--script-arg=--seal-token'],check=True)
    print(json.dumps({'status':'completed','existing_hf_credential_available':bool(token),
                      'private_workspace':private,'credential_in_artifacts':False}))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}));raise SystemExit(1)
