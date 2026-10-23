"""Verify exported LLM source, result groups and decompressed evidence hashes."""
import argparse,gzip,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();root=a.root
    manifest=json.loads((root/'MANIFEST.json').read_text());actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(manifest['files'])|{'MANIFEST.json'}
    for name,row in manifest['files'].items():
        f=root/name;assert not f.is_symlink() and f.stat().st_size==row['bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],name
    checked=set();index=json.loads((root/'evidence/index.json').read_text())
    for group in index['groups']:
        for f in group['files']:
            if f['blob'] in checked:continue
            with gzip.open(root/f['blob'],'rb') as stream:data=stream.read()
            assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'],f['source_path'];checked.add(f['blob'])
    rows=[json.loads(x) for x in (root/'results/completed_entries.jsonl').read_text().splitlines() if x];ids={json.loads(x)['entry_id'] for x in (root/'results/planned_entries.jsonl').read_text().splitlines() if x};status=json.loads((root/'checkpoint.json').read_text())
    assert len(ids)==200 and len(rows)==status['active_completed'] and len({r['entry_id'] for r in rows})==len(rows)
    assert all(r['entry_id'] in ids and r['output_dtype']=='bfloat16' and len(r['measurement']['samples'])==15 for r in rows)
    groups={}
    for r in rows:groups.setdefault((r['model'],r['batch'],r['sequence']),set()).add(r['algorithm'])
    assert all(v=={'native_default','native_tuned','cubic_tuned','s1','s2'} for v in groups.values())
    print(json.dumps(dict(passed=True,files=len(manifest['files']),unique_evidence_blobs=len(checked),active_comparisons=len(rows),whole_workloads=len(groups))))
if __name__=='__main__':main()
