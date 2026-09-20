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

def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def same_mean(values,reported):
    return bool(values) and finite_number(reported) and math.isclose(
        statistics.mean(values),reported,rel_tol=1e-10,abs_tol=1e-10)

def quality_gate(value,thresholds):
    keys=('nll_delta','mean_kl','top1_agreement')
    if 'max_relative_l2' in thresholds:keys+=('relative_l2',)
    return bool(value.get('all_finite') is True and all(finite_number(value.get(k)) for k in keys)
        and abs(value['nll_delta'])<=thresholds['max_abs_nll_delta']
        and value['mean_kl']<=thresholds['max_mean_kl']
        and value['top1_agreement']>=thresholds['min_top1_agreement']
        and ('max_relative_l2' not in thresholds or value['relative_l2']<=thresholds['max_relative_l2']))

def audit_n9(art,summary,rows,cases,issues):
    """Validate terminal failures as evidence, and complete policy claims from raw events."""
    campaign=load(art/'effective_campaign.json');planned=campaign['models']
    if load(art/'planned_cases.json')!=planned:issues.append('N9 planned cases differ from effective campaign')
    expected_models=[item['model_id'] for item in planned]
    terminal=summary.get('results',[]);model_ids=[item.get('model_id') for item in terminal]
    if len(set(model_ids))!=len(model_ids):issues.append('Duplicate N9 terminal model outcome')
    if set(model_ids)!=set(expected_models):issues.append('N9 planned/terminal model coverage differs')
    if any(row.get('model_id') not in expected_models for row in cases):issues.append('Unexpected N9 case-result model')
    observed={item['model_id']:item for item in terminal};outcomes=[]
    for index,item in enumerate(planned):
        model_id=item['model_id'];result=observed.get(model_id)
        if result is None:continue
        events=[r for r in rows if r.get('model_id')==model_id]
        model_cases=[r for r in cases if r.get('model_id')==model_id]
        directory=art/f'model-{index:02d}';status=result.get('status')
        if status=='failed':
            failures=[r for r in events if r.get('event')=='model_error']
            if len(failures)!=1:issues.append('N9 failed model lacks one terminal error: '+model_id)
            elif any(failures[0].get(k)!=v for k,v in result.items()):issues.append('N9 model error differs from summary: '+model_id)
            outcomes.append({'model_id':model_id,'status':status,'scope':'model-level terminal outcome'})
            continue
        if not (directory/'model_summary.json').is_file():issues.append('Missing N9 model summary: '+model_id)
        elif load(directory/'model_summary.json')!=result:issues.append('N9 model summary differs from run summary: '+model_id)
        if status=='blocked_access':
            if item.get('status')!='blocked_access':issues.append('Unexpected N9 access blocker: '+model_id)
        elif status not in ('completed','failed_native_qualification'):
            issues.append('Unknown N9 terminal model status: '+model_id+'/'+str(status));continue
        if status!='blocked_access':
            qualification=result.get('qualification') or {}
            qualified=quality_gate(qualification,campaign['qualification_thresholds'])
            if qualification.get('positions')!=63 or qualification.get('thresholds')!=campaign['qualification_thresholds']:
                issues.append('N9 qualification scope/threshold mismatch: '+model_id)
            if qualification.get('passed') is not qualified:issues.append('N9 qualification gate inconsistent: '+model_id)
            if (status=='completed')!=qualified:issues.append('N9 terminal outcome contradicts native qualification: '+model_id)
            records=[r for r in events if r.get('event')=='native_qualification']
            if len(records)!=1 or any(records[0].get(k)!=v for k,v in qualification.items()):
                issues.append('N9 qualification journal differs from summary: '+model_id)
            if not (directory/'qualification.json').is_file() or load(directory/'qualification.json')!=qualification:
                issues.append('N9 qualification artifact missing/different: '+model_id)
        if status!='completed':
            if len(model_cases)!=1 or model_cases[0].get('arm_id') is not None or any(model_cases[0].get(k)!=v for k,v in result.items()):
                issues.append('N9 model-level case outcome differs/missing: '+model_id)
            if result.get('quality_measured') is not False or result.get('eligible_for_speedup_claim') is not False:
                issues.append('N9 unqualified/blocked model claims quality or speedup: '+model_id)
            outcomes.append({'model_id':model_id,'status':status,'scope':'model-level terminal outcome'});continue
        arms={'native',*item.get('policies',{})};quality=result.get('quality') or {};failures=result.get('failures') or {}
        eligibility=result.get('eligible_for_speedup_claim') or {}
        actual_arms=[r.get('arm_id') for r in model_cases]
        if set(actual_arms)!=arms or len(actual_arms)!=len(arms) or any(r.get('scope') is not None for r in model_cases):
            issues.append('N9 planned/per-policy case-result coverage differs: '+model_id)
        if set(quality)|set(failures)!=arms or set(eligibility)!=arms or 'native' not in quality:
            issues.append('N9 planned policy outcome/eligibility coverage differs: '+model_id)
        if result.get('quality_measured') is not True:issues.append('N9 completed model lacks measured quality: '+model_id)
        window_rows=[r for r in events if r.get('event')=='quality_window']
        if len(window_rows)!=32 or {r.get('window') for r in window_rows}!=set(range(32)):
            issues.append('N9 quality-window coverage differs: '+model_id)
        if not (directory/'quality.json').is_file() or load(directory/'quality.json')!=quality:
            issues.append('N9 quality artifact missing/different: '+model_id)
        for arm,value in quality.items():
            if value.get('positions')!=32736 or value.get('thresholds')!=campaign['quality_thresholds']:
                issues.append('N9 quality scope/threshold mismatch: '+model_id+'/'+arm)
            if value.get('passed') is not quality_gate(value,campaign['quality_thresholds']):
                issues.append('N9 quality gate inconsistent: '+model_id+'/'+arm)
            if any((r.get('metrics',{}).get(arm) or {}).get('positions')!=1023 for r in window_rows):
                issues.append('N9 policy lacks every scored window: '+model_id+'/'+arm)
        manifest_path=directory/'model_manifest.json'
        if not manifest_path.is_file():issues.append('Missing N9 model manifest: '+model_id);continue
        manifest=load(manifest_path);layers=manifest['config']['num_hidden_layers']
        if manifest.get('model_id')!=model_id or manifest.get('revision')!=item['revision']:
            issues.append('N9 model manifest identity differs: '+model_id)
        resident=defaultdict(list);streamed=defaultdict(list)
        for record in events:
            target=resident if record.get('event')=='resident_layer_sample' else streamed if record.get('event')=='streamed_forward_sample' else None
            if target is None:continue
            arm=record.get('arm_id');target[arm].append(record)
            if arm not in arms:issues.append('Unexpected N9 timing arm: '+model_id+'/'+str(arm))
            if not finite_number(record.get('elapsed_ms')) or record['elapsed_ms']<=0:
                issues.append('Invalid N9 raw latency: '+model_id+'/'+str(arm))
        resident_expected={(layer,repeat) for layer in range(layers) for repeat in range(campaign['resident_timing']['repeats'])}
        streamed_expected=set(range(campaign['streamed_repeats']))
        coverage=result.get('resident_sample_coverage') or {};resident_means=result.get('resident_layer_mean_ms') or {}
        streamed_values=result.get('streamed_forward_samples_ms') or {};streamed_means=result.get('streamed_forward_mean_ms') or {}
        if set(resident_means)!=set(quality):issues.append('N9 resident headline policy coverage differs: '+model_id)
        if set(streamed_values)!=set(quality):issues.append('N9 streamed policy coverage differs: '+model_id)
        for arm in arms:
            resident_rows=resident[arm];rkeys=[(r.get('layer'),r.get('repeat')) for r in resident_rows]
            if len(set(rkeys))!=len(rkeys) or not set(rkeys)<=resident_expected:
                issues.append('Invalid/duplicate N9 resident rounds: '+model_id+'/'+arm)
            full_resident=len(rkeys)==len(resident_expected) and set(rkeys)==resident_expected
            if arm in quality and not full_resident:issues.append('N9 surviving policy lacks complete resident timing: '+model_id+'/'+arm)
            if arm in coverage:
                expected_coverage={'observed':len(rkeys),'expected':len(resident_expected),'complete':full_resident}
                if coverage[arm]!=expected_coverage:issues.append('N9 resident coverage differs from raw samples: '+model_id+'/'+arm)
            elif resident_rows or arm in quality:issues.append('Missing N9 resident sample coverage: '+model_id+'/'+arm)
            if arm in resident_means and not same_mean([r['elapsed_ms'] for r in resident_rows],resident_means[arm]):
                issues.append('N9 resident mean differs from raw timings: '+model_id+'/'+arm)
            stream_rows=streamed[arm];skeys=[r.get('repeat') for r in stream_rows];times=[r['elapsed_ms'] for r in stream_rows]
            if len(set(skeys))!=len(skeys) or not set(skeys)<=streamed_expected:
                issues.append('Invalid/duplicate N9 streamed rounds: '+model_id+'/'+arm)
            full_stream=len(skeys)==len(streamed_expected) and set(skeys)==streamed_expected
            if arm in streamed_values and streamed_values[arm]!=times:issues.append('N9 streamed sample list differs from journal: '+model_id+'/'+arm)
            if (arm in streamed_means)!=full_stream:issues.append('N9 streamed headline coverage differs: '+model_id+'/'+arm)
            if arm in streamed_means and not same_mean(times,streamed_means[arm]):issues.append('N9 streamed mean differs from raw timings: '+model_id+'/'+arm)
            expected_eligible=bool(arm in quality and quality[arm].get('passed') is True and arm not in failures and full_stream)
            if eligibility.get(arm) is not expected_eligible:issues.append('N9 speedup eligibility inconsistent: '+model_id+'/'+arm)
            case=next((r for r in model_cases if r.get('arm_id')==arm),None)
            value=quality.get(arm);failure=failures.get(arm)
            expected_status=failure.get('status') if failure else 'ok' if value and value.get('passed') else 'failed_quality'
            if case and (case.get('status')!=expected_status or case.get('quality')!=value or case.get('eligible_for_speedup_claim') is not expected_eligible):
                issues.append('N9 policy case differs from terminal summary: '+model_id+'/'+arm)
            if failure and not any(r.get('event')=='policy_error' and r.get('arm_id')==arm and all(r.get(k)==v for k,v in failure.items()) for r in events):
                issues.append('N9 policy failure missing from journal: '+model_id+'/'+arm)
            outcomes.append({'model_id':model_id,'arm_id':arm,'status':expected_status,'failure':failure,
                             'quality_passed':value.get('passed') if value else None})
    return outcomes

def audit(path):
    path=path.resolve();issues=[]
    execution=load(path/'execution.json');completion=load(path/'completion.json')
    verify(path,load(path/'artifact-manifest.json'),issues)
    source=load(path/'source-manifest.json')
    root=path.parent.parent
    if execution['source_commit']!=source['source_commit']:issues.append('Execution/source manifest commits differ')
    roots=subprocess.check_output(['git','ls-tree','--name-only',source['source_commit']],cwd=root,text=True).splitlines()
    roots=[r for r in roots if r not in ('runs','status')]
    check=subprocess.Popen(['git','archive',source['source_commit'],'--',*roots],cwd=root,stdout=subprocess.PIPE)
    archive_hash=hashlib.sha256()
    for chunk in iter(lambda:check.stdout.read(1024**2),b''):archive_hash.update(chunk)
    check.stdout.close()
    if check.wait() or archive_hash.hexdigest()!=source['archive_sha256']:issues.append('Frozen source archive differs from its declared Git commit')
    if sha(path/'source.tar')!=source['archive_sha256']:issues.append('Source archive hash mismatch')
    verify(path/'source',source['source_sha256'],issues)
    art=path/'artifacts';summary=load(art/'summary.json') if (art/'summary.json').is_file() else completion.get('summary') or {}
    phase=summary.get('phase',execution['label'])
    if phase in ('N5-screen','N5-confirm','N6','N7-screen','N7-confirm','N7-evaluate','N7-replicate','N8','N9'):
        for name in ('artifact_manifest.json','summary.json','environment.json','results.jsonl'):
            if not (art/name).is_file():issues.append('Required canonical benchmark artifact missing: '+name)
    if phase in ('N8','N9'):
        for name in ('planned_cases.json','effective_campaign.json'):
            if not (art/name).is_file():issues.append('Required application protocol artifact missing: '+name)
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
    keys=[(r.get('case_id') or r.get('group_id') or r.get('model_id'),r.get('arm_id'),r.get('scope')) for r in cases]
    if len(set(keys))!=len(keys):issues.append('Duplicate case result identity')
    if phase in EXPECTED and summary.get('completed') is True and len(cases)!=EXPECTED[phase]:
        issues.append(f'Completed phase has {len(cases)} rows, expected {EXPECTED[phase]}')
    counts=dict(Counter(r.get('status','unknown') for r in cases))
    if 'case_status_counts' in summary and summary['case_status_counts']!=counts:issues.append('Summary status counts differ from journal')
    if phase=='N8' and all((art/name).is_file() for name in ('planned_cases.json','effective_campaign.json')):
        planned=load(art/'planned_cases.json')
        expected={(g['id'],a['arm_id'],scope) for g in planned for a in g['arms'] for scope in ('call','prepared_kernel')}
        actual={(r.get('group_id'),r.get('arm_id'),r.get('scope')) for r in cases}
        if expected!=actual:issues.append(f'N8 planned/result coverage mismatch: missing={len(expected-actual)}, unexpected={len(actual-expected)}')
        application_campaign=load(art/'effective_campaign.json')
        policy=application_campaign['application_timing'];gate=application_campaign['correctness']['gate']
        raw=defaultdict(list)
        for row in rows:
            if row.get('event')=='timing_sample':
                key=(row['group_id'],row['arm_id'],row['scope']);raw[key].append(row)
                if key not in expected:issues.append('Unexpected N8 timing identity: '+str(key))
                if not finite_number(row.get('elapsed_ms')) or row['elapsed_ms']<=0:issues.append('Invalid N8 raw latency: '+str(key))
                if row.get('repeat') not in range(policy['repeats']) or row.get('round')!=row.get('repeat'):
                    issues.append('Invalid N8 timing round: '+str(key))
        for key,values in raw.items():
            if len({r['repeat'] for r in values})!=len(values):issues.append('Duplicate N8 timing round: '+str(key))
        for row in cases:
            key=(row.get('group_id'),row.get('arm_id'),row.get('scope'))
            check=row.get('correctness')
            if check is not None:
                passed=bool(check.get('finite') is True and all(finite_number(check.get(k)) for k in ('relative_l2','max_abs_error','max_abs_reference'))
                    and check['relative_l2']<=gate['relative_l2_max']
                    and check['max_abs_error']<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*check['max_abs_reference'])
                if check.get('gate')!=gate or check.get('pass') is not passed or check.get('passed') is not passed:
                    issues.append('N8 numerical gate inconsistent: '+str(key))
                expected_raw=[{'round':r['repeat'],'elapsed_ms':r['elapsed_ms']} for r in raw[key]]
                timing=row.get('timing') or {}
                if timing.get('sample_count')!=len(expected_raw) or timing.get('raw_samples')!=expected_raw or row.get('raw_ms')!=[r['elapsed_ms'] for r in raw[key]]:
                    issues.append('N8 case timing payload differs from journal: '+str(key))
            if row.get('status')=='ok':
                samples8=raw[key]
                if len(samples8)!=policy['repeats'] or {x['repeat'] for x in samples8}!=set(range(policy['repeats'])):issues.append('N8 incomplete timing rounds: '+str(key))
                values=[x['elapsed_ms'] for x in samples8]
                if not same_mean(values,row['timing'].get('mean_ms')):issues.append('N8 mean differs from raw timings: '+str(key))
                if row.get('mean_ms')!=row['timing'].get('mean_ms'):issues.append('N8 top-level mean differs from timing payload: '+str(key))
                if (row.get('correctness') or {}).get('pass') is not True:issues.append('N8 passing case lacks numerical gate: '+str(key))
                preparation=row.get('preparation_raw_ms') or []
                if len(preparation)!=policy['preparation_repeats'] or any(not finite_number(v) or v<=0 for v in preparation):
                    issues.append('N8 successful case lacks valid preparation samples: '+str(key))
    samples=defaultdict(list)
    for r in rows:
        if r.get('event')=='sample':samples[(r.get('case_id') or r.get('group_id') or r.get('model_id'),r.get('arm_id'),r.get('scope'))].append(r)
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
    policy_outcomes=[]
    if phase=='N9' and all((art/name).is_file() for name in ('planned_cases.json','effective_campaign.json')):
        policy_outcomes=audit_n9(art,summary,rows,cases,issues)
    log=subprocess.run(['git','log','-1','--format=%H %s','--',str((path/'artifact-manifest.json').relative_to(root))],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    if not log:issues.append('No Git commit contains execution artifact manifest')
    else:
        revision=log.split(' ',1)[0];relative=str((path/'artifact-manifest.json').relative_to(root))
        committed=subprocess.check_output(['git','show',revision+':'+relative],cwd=root)
        if hashlib.sha256(committed).hexdigest()!=sha(path/'artifact-manifest.json'):issues.append('Current execution manifest differs from committed blob')
        history=subprocess.check_output(['git','log','--format=%H','--',relative],cwd=root,text=True).splitlines()
        if len(history)!=1:issues.append('Immutable execution manifest has multiple modifying commits')
    return {'run_id':path.name,'phase':phase,'execution_status':completion.get('status'),
      'audit_pass':not issues,'issues':issues,'source_commit':execution['source_commit'],'archive_commit':log,
      'artifact_manifest_sha256':sha(path/'artifact-manifest.json'),'identity':env.get('identity'),
      'case_status_counts':counts,'record_count':len(rows),'summary':summary,'terminal_model_policy_outcomes':policy_outcomes,
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
