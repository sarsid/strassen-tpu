"""Publish a committed LLM checkpoint with deduplicated, lossless evidence."""
import argparse, collections, gzip, hashlib, json, shutil, subprocess, tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root,out=a.root.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    def committed(path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
    tracked={line.split('\t',1)[1]:line.split()[2] for line in subprocess.check_output(['git','ls-tree','-r',commit],cwd=root,text=True).splitlines()}
    def write(name,data):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    def obj(name,value):write(name,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    code_roots={'src','tools','configs','runtime','docs','plans','tests','third_party'}
    paths=[s for s in tracked if s.split('/')[0] in code_roots or s in ('AGENTS.md','README.md','strassen_optimized.py') or (s.startswith('status/') and s.endswith(('.py','.html','.md')))]
    proc=subprocess.Popen(['git','archive',commit,'--',*paths],cwd=root,stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
        for member in archive:
            if member.isdir():continue
            assert member.isfile() and '..' not in Path(member.name).parts
            write('code/'+member.name,archive.extractfile(member).read())
    assert proc.wait()==0
    campaign='results/v6e/llm_campaign_20261003_v001'
    ledger=committed(campaign+'/completed_entries.jsonl');rows=[json.loads(x) for x in ledger.splitlines() if x]
    plan=committed(campaign+'/planned_grid_four_models_v001/planned_entries.jsonl');ids={json.loads(x)['entry_id'] for x in plan.splitlines() if x}
    active=[r for r in rows if r['entry_id'] in ids];assert len(ids)==200 and len({r['entry_id'] for r in active})==len(active)
    groups=collections.defaultdict(list)
    for row in active:groups[(row['model'],row['batch'],row['sequence'])].append(row)
    assert all(len(g)==5 and len({r['algorithm'] for r in g})==5 and all(r['output_dtype']=='bfloat16' for r in g) for g in groups.values())
    assert all(len(r['measurement']['samples'])==15 for r in active)
    write('results/completed_entries_all_history.jsonl',ledger)
    write('results/completed_entries.jsonl',b''.join((json.dumps(r)+'\n').encode() for r in active))
    write('results/planned_entries.jsonl',plan)
    for path in tracked:
        if path.startswith(campaign+'/'):
            rel=path[len(campaign)+1:]
            if rel.split('/')[0] in ('inputs_v001','inputs_v002','inputs_gemma4_v001','qwen8_complete_v001','qwen32_interim_3_workloads_v001','planned_grid_v001','planned_grid_bf16_v001'):
                write('results/'+rel,committed(path))
        if path.startswith('results/v6e/parent_llm_'):
            write('parent_study/'+path.removeprefix('results/v6e/'),committed(path))
    blobs={};evidence=[]
    def add_file(path):
        relative=path.relative_to(root).as_posix();assert relative in tracked and path.is_file() and not path.is_symlink()
        data=path.read_bytes();gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert gitsha==tracked[relative], 'Uncommitted or changing evidence: '+relative
        h=sha(data);blob='evidence/blobs/'+h+'.gz'
        if h not in blobs:
            write(blob,gzip.compress(data,compresslevel=6,mtime=0));blobs[h]=dict(blob=blob,bytes=len(data),sha256=h)
        return dict(source_path=relative,**blobs[h])
    def add_tree(base,label):
        records=[]
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.relative_to(root).as_posix() in tracked and not path.name.endswith(('.tar.gz','.tgz')):
                records.append(dict(path=path.relative_to(base).as_posix(),**add_file(path)))
        assert records,label;evidence.append(dict(name=label,files=records));return records
    state=json.loads((root/campaign/'queue_v001/state.json').read_text())
    jobs={(j['model'],j['batch'],j['sequence']):j for j in state['jobs']}
    for key,items in sorted(groups.items()):
        run=items[0]['evidence_run'];assert all(r['evidence_run']==run for r in items)
        folder=root/campaign/run
        if run.startswith('session-'):base=folder/('case-'+jobs[key]['job_id'])/'case'
        else:base=folder/'remote'/run/'artifacts'
        saved=[json.loads(x) for x in (base/'main-prefill/completed_entries.jsonl').read_text().splitlines() if x]
        mapped={r['entry_id']:r for r in saved}
        assert all(mapped[r['entry_id']]['measurement']==r['measurement'] for r in items)
        add_tree(base,key[0]+f'-b{key[1]}s{key[2]}')
    for run in sorted({r['evidence_run'] for r in active}):
        files=[]
        for name in ('frozen.json','source.tar','job.json','completion.json','bundle-validation.json'):
            path=root/campaign/run/name
            if path.relative_to(root).as_posix() in tracked:files.append(dict(path=name,**add_file(path)))
        evidence.append(dict(name=run+'-source',files=files))
    qualification_roots=sorted({s.split('/')[1] for s in tracked if s.startswith('runs/') and ('gemma4-' in s.split('/')[1] or 'qwen32-b1s4096-summary' in s.split('/')[1])})
    for run in qualification_roots:
        base=root/'runs'/run;records=[]
        for path in sorted(base.rglob('*')):
            relative=path.relative_to(base).as_posix()
            if path.is_file() and path.relative_to(root).as_posix() in tracked and not relative.startswith('source/'):
                records.append(dict(path=relative,**add_file(path)))
        evidence.append(dict(name=run,files=records))
        if 'qwen32-b1s4096-summary' in run:write('results/qwen32_b1s4096_summary.json',(base/'artifacts/summary.json').read_bytes())
    obj('evidence/index.json',dict(groups=evidence,unique_blobs=len(blobs),rule='Lossless content-addressed gzip; repeated bytes stored once. Restore original filenames with restore.py.'))
    counts=collections.Counter(r['model'] for r in active)
    checkpoint=dict(saved_utc=datetime.now(timezone.utc).isoformat(),source_commit=commit,active_completed=len(active),active_expected=200,all_completed_quality_gates_passed=all(r['quality_gate_passed'] for r in active),by_model=dict(counts),historical_entries=len(rows)-len(active),full_campaign_complete=False,model_order=state['model_order'],gemma4_status='awaiting_official_checkpoint_and_runtime_qualification',snapshot_scope='Completed whole five-arm groups only; running workload excluded.',omitted=['Private weights, caches, credentials and virtual environments','Duplicate archive containers and duplicate bytes','Transient controller state and partial running workload'],ledger_sha256=sha(ledger))
    obj('checkpoint.json',checkpoint)
    write('verify.py',committed('tools/verify_public_llm_v001.py'))
    write('restore.py',committed('tools/restore_public_llm_v001.py'))
    write('README.md',committed('docs/PUBLIC_LLM_CHECKPOINT_20261004_v001.md'))
    obj('MANIFEST.json',dict(source_commit=commit,files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file()}))
    assert all(p.stat().st_size<100_000_000 for p in out.rglob('*') if p.is_file())
    print(json.dumps(dict(output=str(out),comparisons=len(active),unique_blobs=len(blobs),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),source_commit=commit)))


if __name__=='__main__':main()
