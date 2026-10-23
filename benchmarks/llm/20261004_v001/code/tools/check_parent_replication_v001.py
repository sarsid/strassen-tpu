"""Source and public transport preflight; no network or performance claim."""
import ast, hashlib, json, os, tempfile
from pathlib import Path
from types import SimpleNamespace
import parent_stage_v001 as stage

def main():
    root=Path(__file__).resolve().parents[1]
    parent=stage.PARENT
    records=[]
    provenance=json.loads((parent/'provenance.json').read_text())
    for name,info in provenance['files'].items():
        data=(parent/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==info['sha256'] and len(data)==info['bytes'],name
    records.append(dict(check='parent_source_hashes',count=len(provenance['files'])))
    for pattern in ['tools/*parent*v001.py','runtime/*parent*v001.py']:
        for path in root.glob(pattern):ast.parse(path.read_text(),filename=str(path))
    records.append(dict(check='runner_syntax'))
    with tempfile.TemporaryDirectory() as temp:
        class Loader:pass
        layer=SimpleNamespace(BASE='https://huggingface.co/public/model/resolve/pinned',ShardedSafetensors=Loader)
        stage.install_transport(layer,Path(temp),lambda x:None)
        class Response:
            status_code=206
            headers={'Content-Range':'bytes 0-7/16'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_content(self,size):yield b'abcdefgh'
        class Session:
            calls=0
            def get(self,*args,**kwargs):self.calls+=1;return Response()
        instance=Loader();instance.session=Session()
        url=layer.BASE+'/tensor'
        assert instance._get(url,'bytes=0-7').content==b'abcdefgh'
        assert instance._get(url,'bytes=0-7').content==b'abcdefgh'
        assert instance.session.calls==1
        path=next(Path(temp).glob('*.bin'));path.write_bytes(b'changed!')
        try:instance._get(url,'bytes=0-7')
        except ValueError:pass
        else:raise AssertionError('Corrupt cache was accepted')
    records.append(dict(check='exact_range_cache_reuse_and_corruption_rejection'))
    run=stage.configure(Path(temp)/'outputs')
    import importlib
    for name in run.STAGES.values():
        if name in [run.STAGES[s] for s in ('layer','streamed','quality')]:importlib.import_module(name)
    records.append(dict(check='all_three_parent_stage_imports'))
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    (out/'summary.json').write_text(json.dumps(dict(status='passed',checks=records),indent=2)+'\n')
    print(json.dumps(records))

if __name__=='__main__':main()
