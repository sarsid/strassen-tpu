"""Exercise inline RPC launch with sibling imports absent until extraction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('launcher',root/'runtime/launch_gemma4_bundle_v002.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
with tempfile.TemporaryDirectory() as td:
    base=Path(td);src=base/'fixture';(src/'runtime').mkdir(parents=True)
    (src/'runtime/helper.py').write_text('VALUE = 17\n')
    marker=base/'started.json'
    (src/'runtime/gemma4_batch_remote_v001.py').write_text('from helper import VALUE\nimport json,sys\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text(json.dumps({"value":VALUE,"argv":sys.argv}))\n')
    archive=base/'source.tar'
    with tarfile.open(archive,'w') as f:f.add(src/'runtime',arcname='runtime')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    launcher.launch(archive,digest,'fixture',base)
    result=json.loads(marker.read_text());assert result['value']==17 and result['argv'][-1]=='fixture'
    (base/'runs/duplicate').mkdir(parents=True)
    try:launcher.launch(archive,digest,'duplicate',base)
    except FileExistsError:pass
    else:raise AssertionError('Duplicate remote run accepted')
    try:launcher.launch(archive,'0'*64,'bad-hash',base)
    except ValueError:pass
    else:raise AssertionError('Bad source hash accepted')
print(json.dumps({'status':'passed','sibling_import_after_extract':True,'duplicate_launch_rejected':True,'source_hash_checked':True,'scope':'local launcher check; no TPU'}))
