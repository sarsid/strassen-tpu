"""Restore one exported evidence group to a new directory, verifying every file."""
import argparse,gzip,hashlib,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--group');p.add_argument('--output',type=Path);a=p.parse_args()
    groups=json.loads((a.root/'evidence/index.json').read_text())['groups']
    if not a.group:
        print('\n'.join(g['name'] for g in groups));return
    if a.output is None:p.error('--output is required with --group')
    group=next(g for g in groups if g['name']==a.group);a.output.mkdir(parents=True,exist_ok=False)
    for row in group['files']:
        name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
        with gzip.open(a.root/row['blob'],'rb') as source:data=source.read()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps(dict(group=a.group,files=len(group['files']),output=str(a.output))))
if __name__=='__main__':main()
