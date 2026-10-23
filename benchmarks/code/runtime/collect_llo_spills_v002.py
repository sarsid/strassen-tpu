"""Preserve XProf outputs automatically spilled to /tmp outside trace artifacts."""
import json,shutil,tarfile,hashlib
from pathlib import Path
profile=Path('/content/Strassen_MM_Focus/runs/20260923-v6e-diagnostic-v001-llo-profile-6690ea/artifacts/profiles/llo_probe__profile_0')
out=Path('/content/Strassen_MM_Focus/llo-spill-recovery-v002');out.mkdir(exist_ok=False)
records=[]
for f in profile.glob('get_*.stdout.txt'):
 d=json.loads(f.read_text())
 if not isinstance(d,dict):continue
 if d.get('status')!='SAVED_TO_FILE':continue
 source=Path(d['file_path'])
 if source.parent!=Path('/tmp') or not source.name.startswith('xprof_spill_'):raise ValueError('Unexpected profiler spill location')
 dest=out/(f.name.removesuffix('.stdout.txt')+'.json');shutil.copyfile(source,dest)
 records.append(dict(command=f.stem,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
with tarfile.open(str(out)+'.tar.gz','w:gz')as t:t.add(out,arcname=out.name)
print(json.dumps(dict(archive=str(out)+'.tar.gz',files=records)),flush=True)
