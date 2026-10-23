"""Replay a completed synthetic five-method phase from immutable raw evidence."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from strassen_mm import benchmark_mlsys_shapes_v001 as study


def audit(artifacts):
    artifacts=Path(artifacts);checks=0
    def require(condition,message):
        nonlocal checks
        if not condition:raise ValueError(message)
        checks+=1
    def read(name):return json.loads((artifacts/name).read_text())
    summary=read('summary.json');groups=read('planned_cases.json');source=read('source_manifest.json')
    require(summary['completed']is True,'Phase did not complete')
    for name,expected in source['sha256'].items():
        require(hashlib.sha256((artifacts/name).read_bytes()).hexdigest()==expected,'Source hash mismatch '+name)
    cfg=read('config_snapshot/campaign.json');manifest=read('config_snapshot/shapes.json')
    study.validate_campaign(cfg,manifest['shapes']);checks+=1
    environment=read('environment.json')
    require(environment['qualified_single_v5e']is True,'Unqualified device')
    raw=(artifacts/'results.jsonl').read_text()
    require(raw.endswith('\n'),'Unterminated raw journal')
    events=[json.loads(line)for line in raw.splitlines()]
    require(len(events)==len({r['sequence']for r in events}),'Duplicate event sequence')
    cases=[r for r in events if r['event']=='case_result'];samples=defaultdict(list)
    for row in events:
        if row['event']=='sample':samples[(row['group_id'],row['arm_id'],row['seed'])].append(dict(round=row['round'],elapsed_ms=row['elapsed_ms']))
    expected={(g['group_id'],a['arm_id'],i['seed'])for g in groups for a in g['arms']for i in g['inputs']}
    case_keys=[(r['group_id'],r['arm_id'],r['seed'])for r in cases]
    require(len(cases)==len(set(case_keys))and set(case_keys)==expected,'Outcome inventory mismatch')
    require(summary['observed_outcomes']==summary['planned_outcomes']==len(cases),'Summary case count mismatch')
    require(summary['case_status_counts']==dict(Counter(r['status']for r in cases)),'Status count mismatch')
    require(summary['completed_groups']==[g['group_id']for g in groups],'Completed group sequence mismatch')
    require([r['group_id']for r in events if r['event']=='group_complete']==summary['completed_groups'],'Raw group completion mismatch')
    fingerprints={}
    for row in events:
        if row['event']=='case_start':
            key=(row['shape_id'],row['distribution'],row['seed'])
            if key in fingerprints:require(row['input_fingerprint']==fingerprints[key],'Candidate inputs differ within a shape/seed')
            else:fingerprints[key]=row['input_fingerprint']
    group_by_id={g['group_id']:g for g in groups};arm_by_key={(g['group_id'],a['arm_id']):a for g in groups for a in g['arms']}
    for row in cases:
        key=row['group_id'],row['arm_id'],row['seed'];group=group_by_id[row['group_id']]
        arm=arm_by_key[key[:2]]
        require(row['scope']=='call','Unexpected timing scope')
        require(row['shape_mkn']==[group['shape'][d]for d in ('m','k','n')],'Shape order mismatch')
        require(row['family']==arm['family']and row['candidate_id']==arm['candidate_id'],'Family/candidate mismatch')
        timing=row.get('timing') or {};observed=samples.get(key,[])
        require(len({s['round']for s in observed})==len(observed),'Duplicate timing round')
        if observed:
            values=[s['elapsed_ms']for s in observed]
            require(all(math.isfinite(v)and v>0 for v in values),'Invalid elapsed time')
            require(timing['sample_count']==len(values),'Sample count mismatch')
            require(math.isclose(timing['mean_ms'],statistics.mean(values),rel_tol=1e-12),'Timing mean mismatch')
        if row['status']in ('ok','numerical_failure'):
            require(len(observed)==cfg['timing'][summary['stage']]['repeats'],'Incomplete measured case')
            metadata=row['kernel_metadata'];metrics=row['correctness'];gate=cfg['correctness']['gate']
            require(metadata['input_dtype']=='bfloat16'and metadata['output_dtype']=='float32'and metadata['dot_precision']=='DEFAULT','Precision mismatch')
            require(metrics['reference_backend']=='host_numpy_fp64'and metrics['all_k_used']==group['shape']['k'],'Reference mismatch')
            require(metrics['sample_count']==len(metrics['sample_rows'])*len(metrics['sample_columns']),'Reference sample count mismatch')
            for values,axis in [(metrics['sample_rows'],'m'),(metrics['sample_columns'],'n')]:
                require(len(values)==len(set(values))and all(0<=v<group['shape'][axis]for v in values),'Invalid reference indices')
                require(0 in values and group['shape'][axis]-1 in values,'Reference sample omitted an edge')
            finite=bool(metrics['finite'])
            expected_pass=finite and metrics['relative_l2']<=gate['relative_l2_max']and metrics['max_abs_error']<=gate['max_abs_atol']+gate['max_abs_reference_rtol']*metrics['max_abs_reference']
            require(metrics['pass']==expected_pass,'Gate mismatch')
            require(row['eligible_for_speedup_claim']==expected_pass,'Eligibility mismatch')
            require((row['status']=='ok')==expected_pass,'Status/gate mismatch')
        else:
            require(row['eligible_for_speedup_claim']is False,'Failed outcome counted eligible')
    selected=None
    if summary['stage']=='screen':
        selection=read('selections.json');shape_ids={g['shape']['id']for g in groups}
        shapes=[s for s in manifest['shapes']if s['id']in shape_ids]
        replay=study.select_candidates(cases,cfg,shapes)
        require(selection['selected']==replay,'Frozen selection replay mismatch')
        require(selection['environment_identity']==environment['identity'],'Screen identity mismatch')
        require(selection['campaign_sha256']==hashlib.sha256((artifacts/'config_snapshot/campaign.json').read_bytes()).hexdigest(),'Selection campaign hash mismatch')
        selected=replay
    interval_replay='not applicable'
    if summary['stage']=='confirm':
        selection=read('selection_used.json');selected=selection['selected']
        require(selection['environment_identity']==environment['identity'],'Confirmation identity differs from screen')
        shape_indices=sorted(g['manifest_index']for g in groups)
        require(shape_indices==list(range(shape_indices[0],shape_indices[-1]+1)),'Noncontiguous confirmation chunk')
        require(groups==study.plan_groups(cfg,manifest['shapes'],'confirm',selected,shape_indices[0],len(shape_indices)),'Confirmation plan differs from frozen selection')
        # NumPy is optional for a controller inventory audit; when available,
        # replay the deterministic hierarchical bootstrap bit for bit as well.
        try:
            import numpy as np
            study.base.np=np
            replay=study.summarize_confirmation(SimpleNamespace(cases=cases,samples=samples),groups,selected,cfg)
            require(replay==read('confirmation_statistics.json'),'Confidence interval replay mismatch')
            interval_replay='passed'
        except ImportError:interval_replay='not replayed: NumPy unavailable; raw timings and frozen plan verified'
    return dict(passed=True,checks=checks,stage=summary['stage'],groups=len(groups),outcomes=len(cases),
        raw_samples=sum(len(v)for v in samples.values()),status_counts=dict(Counter(r['status']for r in cases)),
        interval_replay=interval_replay,shapes=len({g['shape']['id']for g in groups}),
        limitation='Audit checks integrity, selection, samples and recorded metric consistency; reference outputs were checked during execution and are not reconstructed here.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.artifacts)
    if args.output:
        with args.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
