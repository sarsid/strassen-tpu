"""Publish a lossless, scoped Qwen32 batch-study package from archived evidence."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from analyze_llm_benchmark1_v002 import dump, digest, reconstruct, render, verify

ROOTS = ['results/v6e/qwen32_parent_batch_20261014_v001',
         'results/v6e/qwen32_parent_batch_retry_20261015_v002',
         'results/v6e/qwen32_b64_current_stack_20261015_v004']
RUNS = ['20261014-qwen32-batch-v001','20261015-qwen32-large-batch-v002',
        '20261015-qwen32-b64-v003','20261015-qwen32-b64-v004',
        '20261015-qwen32-b64-v005','20261016-qwen32-b64-v006','20261016-qwen32-b64-v007']
PATTERNS = [rb'\bhf_[A-Za-z0-9]{20,}\b',rb'\bgh[pousr]_[A-Za-z0-9]{25,}\b',
            rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',rb'\bya29\.[A-Za-z0-9_-]{30,}',
            rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
            rb'(?i)authorization["\s:]+bearer\s+[A-Za-z0-9_.-]{25,}']


def scan(raw, name):
    for p in PATTERNS:
        if re.search(p,raw): raise ValueError('Potential credential in '+name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root,out=a.project,a.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'evidence/blobs').mkdir(parents=True)
    index=dict(files={},selected_batches=[])
    def add(p):
        name=str(p.relative_to(root));raw=p.read_bytes();scan(raw,name)
        sha=digest(raw);blob=f'evidence/blobs/{sha}.gz'
        if not (out/blob).exists():(out/blob).write_bytes(gzip.compress(raw,mtime=0))
        index['files'][name]=dict(sha256=sha,bytes=len(raw),blob=blob)
    for rel in ROOTS:
        for p in sorted((root/rel).rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.md'):add(p)
    prov=dict(runs=[],exporter_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    for rn in RUNS:
        run=root/'runs'/rn;dest=out/'provenance'/rn;dest.mkdir(parents=True)
        raw=(run/'source.tar').read_bytes();manifest=json.loads((run/'source-manifest.json').read_text())
        assert digest(raw)==manifest['source_sha256']
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for member in tf:
                if member.isfile():scan(tf.extractfile(member).read(),rn+'/'+member.name)
        (dest/'source.tar.gz').write_bytes(gzip.compress(raw,mtime=0))
        for name in ('source-manifest.json','completion.json','sleep-diagnosis.json','awake-policy.json'):
            p=run/name
            if p.exists():scan(p.read_bytes(),str(p));shutil.copyfile(p,dest/name)
        for p in sorted((run/'remote'/rn/'artifacts').rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):add(p)
        artifact=run/'remote'/rn/'artifacts'
        for name in ('freeze.txt','installation.json','plan.json','profile-reuse.json'):
            if (artifact/name).exists():shutil.copyfile(artifact/name,dest/name)
        release=run/'control/release.log'
        if release.exists():scan(release.read_bytes(),str(release));shutil.copyfile(release,dest/'release.log')
        prov['runs'].append(dict(run=rn,source_archive=str((dest/'source.tar.gz').relative_to(out)),
            source_manifest=str((dest/'source-manifest.json').relative_to(out)),
            completion=str((dest/'completion.json').relative_to(out))))
    for b in (1,4,8,16,32,64):
        ix=0 if b<32 else 1 if b==32 else 2
        rn=RUNS[0] if b<32 else RUNS[1] if b==32 else RUNS[-1]
        index['selected_batches'].append(dict(batch=b,run=rn,artifacts=f'{ROOTS[ix]}/{rn}/b{b}/artifacts'))
    dump(out/'evidence/index.json',index);dump(out/'provenance/index.json',prov)
    shutil.copyfile(Path(__file__).with_name('analyze_llm_benchmark1_v002.py'),out/'analyze.py')
    shutil.copyfile(__file__,out/'build_package.py')
    report=reconstruct(out);dump(out/'summary.json',report);(out/'README.md').write_text(render(report))
    # This package uses no external Python packages, model weights, or network calls.
    dump(out/'MANIFEST.json',dict(files={str(p.relative_to(out)):dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    validation=verify(out);validation['credential_scan']='No targeted credential patterns in raw evidence or frozen source archives'
    validation['source_and_raw_evidence_files']=len(index['files']);dump(out/'validation.json',validation)
    m=json.loads((out/'MANIFEST.json').read_text());p=out/'validation.json'
    m['files']['validation.json']=dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size);dump(out/'MANIFEST.json',m)
    print(json.dumps(verify(out)))
    print(json.dumps(dict(output=str(out),files=sum(p.is_file() for p in out.rglob('*')),
                         bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))))


if __name__=='__main__': main()
