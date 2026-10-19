"""Restore hash-verified batches, then run the unchanged frozen scientific worker."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

BASE=Path('/content/Strassen_MM_Focus')
MODELS=('qwen3_14b','qwen3_32b','gemma4_31b')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_seed(path, expected, source_sha):
    if sha(path)!=expected: raise ValueError('Recovery archive hash mismatch')
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Recovery archive requires unique regular files')
        receipt=json.loads(archive.extractfile('seed-manifest.json').read())
        if receipt['source_sha256']!=source_sha or set(names)!=set(receipt['files'])|{'seed-manifest.json'}:
            raise ValueError('Recovery archive membership/source mismatch')
        for name, expected_hash in receipt['files'].items():
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or len(p.parts)<3 or p.parts[0]!='artifacts' or p.parts[1] not in MODELS:
                raise ValueError('Invalid checkpoint path')
            relative='/'.join(p.parts[2:])
            if relative not in ('manifest.json','execution-contract.json') and not re.fullmatch(r'batches/(mmlu|ppl|gsm8k)-[0-9]{3}\.json',relative):
                raise ValueError('Only checkpoint data may be restored')
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=expected_hash:
                raise ValueError('Checkpoint content hash mismatch')
    return receipt


def prepare(archive, source_sha, seed, seed_sha, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',run_id or ''): raise ValueError('Invalid run ID')
    if not archive.resolve().is_relative_to(BASE) or not seed.resolve().is_relative_to(BASE) or sha(archive)!=source_sha:
        raise ValueError('Source archive path/hash mismatch')
    receipt=inspect_seed(seed,seed_sha,source_sha)
    run=BASE/'runs'/run_id; run.mkdir(parents=True,exist_ok=False)
    source=run/'source'; source.mkdir()
    with tarfile.open(archive) as handle: handle.extractall(source,filter='data')
    with tarfile.open(seed) as handle:
        for name in receipt['files']:
            target=run/name; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(handle.extractfile(name).read())
    (run/'recovery-seed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    command=[sys.executable,str(source/'runtime/accuracy_study_remote_v001.py'),'--worker',str(run)]
    (run/'launch.json').write_text(json.dumps(dict(source_sha256=source_sha,seed_sha256=seed_sha,command=command,models=MODELS),indent=2)+'\n')
    return run,command


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--seed',type=Path,required=True);p.add_argument('--seed-sha256',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();run,command=prepare(a.archive,a.sha256,a.seed,a.seed_sha256,a.run_id)
    with (run/'worker.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(kind='accuracy_study_launched',run=str(run),pid=child.pid,models=MODELS,restored_seed_sha256=a.seed_sha256)),flush=True)


if __name__=='__main__': main()
