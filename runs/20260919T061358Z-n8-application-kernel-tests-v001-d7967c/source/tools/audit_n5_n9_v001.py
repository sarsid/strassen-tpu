#!/usr/bin/env python3
"""Audit explicit immutable experiment runs; never discover or pool cohorts.

Hash validation covers the complete root execution manifest and canonical
benchmark artifacts. Descriptive comparison counts are emitted only for
numerically eligible results. A scientific loss is not an archive failure.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

EXPECTED={'N5-screen':1312,'N5-confirm':96,'N7-screen':1312,'N7-confirm':96,'N7-evaluate':128,'N7-replicate':128}
REPEATS={'N5-screen':7,'N5-confirm':30,'N7-screen':7,'N7-confirm':30,'N7-evaluate':30,'N7-replicate':30}

def load(path): return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''):h.update(data)
    return h.hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def verify(directory,records,issues):
    for name,item in records.items():
        p=(directory/name).resolve()
        expected=item if isinstance(item,str) else item['sha256']
        if not p.is_relative_to(directory.resolve()) or not p.is_file():issues.append('Missing/unsafe sealed path: '+name)
        elif sha(p)!=expected:issues.append('Hash mismatch: '+name)

def audit(path):
    path=path.resolve();issues=[]
    execution=load(path/'execution.json');completion=load(path/'completion.json')
    verify(path,load(path/'artifact-manifest.json'),issues)
    source=load(path/'source-manifest.json')
    if sha(path/'source.tar')!=source['archive_sha256']:issues.append('Source archive hash mismatch')
    verify(path/'source',source['source_sha256'],issues)
    art=path/'artifacts';summary=load(art/'summary.json') if (art/'summary.json').is_file() else completion.get('summary') or {}
    phase=summary.get('phase',execution['label'])
    if (art/'artifact_manifest.json').is_file():
        seal=load(art/'artifact_manifest.json')
        if 'sha256' in seal:verify(art,seal['sha256'],issues)
        elif 'files' in seal:verify(art,seal['files'],issues)
    env=load(art/'environment.json') if (art/'environment.json').is_file() else {}
    rows=[]
    if (art/'results.jsonl').is_file():
        for index,line in enumerate((art/'results.jsonl').read_text().splitlines(),1):
            try:r=json.loads(line)
            except ValueError:issues.append('Malformed journal line '+str(index));continue
            if 'sequence' in r and r['sequence']!=index:issues.append('Journal sequence mismatch '+str(index))
            rows.append(r)
    cases=[r for r in rows if r.get('event')=='case_result']
    keys=[(r.get('case_id') or r.get('group_id'),r.get('arm_id'),r.get('scope')) for r in cases]
    if len(set(keys))!=len(keys):issues.append('Duplicate case result identity')
    if phase in EXPECTED and summary.get('completed') is True and len(cases)!=EXPECTED[phase]:
        issues.append(f'Completed phase has {len(cases)} rows, expected {EXPECTED[phase]}')
    counts=dict(Counter(r.get('status','unknown') for r in cases))
    if 'case_status_counts' in summary and summary['case_status_counts']!=counts:issues.append('Summary status counts differ from journal')
    samples=defaultdict(list)
    for r in rows:
        if r.get('event')=='sample':samples[(r.get('case_id') or r.get('group_id'),r.get('arm_id'),r.get('scope'))].append(r)
    for r,key in zip(cases,keys):
        if r.get('status')=='ok' and phase in REPEATS:
            raw=samples[key];times=[s['elapsed_ms'] for s in raw]
            if len(raw)!=REPEATS[phase] or len({s['round'] for s in raw})!=REPEATS[phase]:issues.append('Incomplete/duplicate timing rounds: '+str(key))
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in times):issues.append('Invalid raw latency: '+str(key))
            if times and not math.isclose(statistics.mean(times),r['timing']['mean_ms'],rel_tol=1e-10,abs_tol=1e-10):issues.append('Mean differs from raw timings: '+str(key))
            if (r.get('correctness') or {}).get('pass') is not True:issues.append('Passing case lacks numerical pass: '+str(key))
        if r.get('eligible_for_speedup_claim') and r.get('status')!='ok':issues.append('Failing case marked speedup eligible: '+str(key))
    contrasts=defaultdict(Counter);ratios=defaultdict(list)
    for r in cases:
        if r.get('scope')!='call':continue
        for pair in r.get('comparisons') or []:
            if not pair.get('valid_numerical_comparison'):continue
            ci=pair.get('speedup_ci95') or []
            if len(ci)!=2:continue
            label=r['arm_id']+' vs '+pair['reference_arm']
            verdict='win' if ci[0]>1 else 'loss' if ci[1]<1 else 'inconclusive'
            contrasts[label][verdict]+=1;ratios[label].append(pair['speedup_ratio_of_means'])
    root=path.parent.parent
    log=subprocess.run(['git','log','-1','--format=%H %s','--',str((path/'artifact-manifest.json').relative_to(root))],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    if not log:issues.append('No Git commit contains execution artifact manifest')
    return {'run_id':path.name,'phase':phase,'execution_status':completion.get('status'),
      'audit_pass':not issues,'issues':issues,'source_commit':execution['source_commit'],'archive_commit':log,
      'artifact_manifest_sha256':sha(path/'artifact-manifest.json'),'identity':env.get('identity'),
      'case_status_counts':counts,'record_count':len(rows),'summary':summary,
      'complete_call_comparisons':{k:{'counts':dict(v),'median_ratio':statistics.median(ratios[k]),'min_ratio':min(ratios[k]),'max_ratio':max(ratios[k])} for k,v in contrasts.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False);reports=[]
    for path in args.run:
        try:reports.append(audit(path))
        except Exception as e:reports.append({'run_id':path.name,'audit_pass':False,'issues':[type(e).__name__+': '+str(e)]})
    registry_issues=[]
    root=args.run[0].resolve().parent.parent
    registry=root/'protocols/code_freezes_v001.jsonl'
    if registry.is_file():
        for line in registry.read_text().splitlines():
            entry=json.loads(line); target=(root/entry['path']).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha(target)!=entry['sha256']:
                registry_issues.append('Executed code differs from frozen fingerprint: '+entry['path'])
    value={'frozen_working_source_issues':registry_issues,'created_utc':datetime.now(timezone.utc).isoformat(),'audit_pass':not registry_issues and all(r['audit_pass'] for r in reports),'runs':reports,
      'interpretation':'Hash/protocol audit is separate from scientific success. Screens are not confirmation; cohorts are never pooled; CIs are per-comparison without multiplicity correction.'}
    save(args.output_dir/'audit.json',value)
    lines=['# N5–N9 evidence audit','', 'Audit: '+('PASS' if value['audit_pass'] else 'FAIL'),'']
    for r in reports:
        lines+=['## '+r['run_id'],'','Archive integrity: '+('PASS' if r['audit_pass'] else 'FAIL'),'','Status counts: `'+json.dumps(r.get('case_status_counts',{}))+'`','']
        for issue in r['issues']:lines.append('- '+issue)
        for label,c in r.get('complete_call_comparisons',{}).items():lines.append('- '+label+': '+json.dumps(c))
        lines.append('')
    with (args.output_dir/'audit.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'audit_pass':value['audit_pass'],'runs':len(reports),'output_dir':str(args.output_dir)}))
    return 0 if value['audit_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
