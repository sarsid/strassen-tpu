"""Summarize sealed five-method phases without reselecting confirmation winners.

Creates a new output directory. Incomplete/failed phases remain explicit and
are never pooled with a successful retry. One logical allocation is required.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

LABELS={'native':'Tuned Native','cubic':'Tuned cubic','one_level':'Tuned Strassen 1',
        'two_level':'Tuned Strassen 2','native_default':'Default Native'}
SCOPES={'mlsys_shapes_v001':'Main development shapes, synthetic Gaussian operands',
        'mlsys_llm_shapes_v001':'LLM-derived shapes, synthetic Gaussian operands'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def checked_phase(cohort,receipt_path):
    receipt=read(receipt_path);run=Path(receipt['run']).resolve()
    if not run.is_relative_to(cohort/'phases'):raise ValueError('Phase escaped cohort: '+str(run))
    if sha(run/'completion.json')!=receipt['completion_sha256']:raise ValueError('Completion hash mismatch')
    completion=read(run/'completion.json')
    if receipt['status']!='completed' or completion.get('remote_may_still_be_running'):
        return run,None,receipt['status']
    for name,expected in read(run/'artifact-manifest.json')['sha256'].items():
        path=(run/name).resolve()
        if not path.is_relative_to(run) or sha(path)!=expected:raise ValueError('Artifact hash mismatch: '+name)
    summary=read(run/'artifacts/summary.json')
    if not summary.get('completed'):return run,summary,'incomplete'
    return run,summary,'completed'


def report(cohort,output):
    cohort=Path(cohort).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Report output must be new')
    rows=[];shortlist=[];phases=[];identities=[];seen=set();inventory={};events_by_phase={}
    applications=[];application_rows=[]
    receipts=sorted(cohort.glob('*-finished.json'))
    for receipt_path in receipts:
        run,summary,state=checked_phase(cohort,receipt_path)
        phase_id=receipt_path.name[:-len('-finished.json')]
        item=dict(phase=phase_id,state=state,run=str(run),summary=str(run/'artifacts/summary.json'))
        phases.append(item)
        if state!='completed':continue
        art=run/'artifacts';config_path=art/'config_snapshot/campaign.json'
        if not config_path.exists():item['scope']='Other preparation/application stage';continue
        cfg=read(config_path);campaign_id=cfg['campaign_id'];item['campaign_id']=campaign_id
        manifest=read(art/'config_snapshot'/cfg['shape_manifest'])
        if (art/'model_summary.json').exists():
            model=read(art/'model_summary.json');item['scope']='Actual original N9 models, separate from large synthetic shapes'
            applications.append(dict(phase=phase_id,evidence=str(art/'model_summary.json'),result=model))
            identities.append(read(art/'environment.json')['identity'])
            for method in model.get('policies',[]):
                quality=model.get('quality',{}).get(method,{})
                streamed=model.get('streamed_model_timings',{}).get(method,{})
                resident=model.get('resident_layer_timings',{}).get(method,{})
                application_rows.append(dict(phase=phase_id,model_id=model['model_id'],model_key=model['model_key'],
                    status=model['status'],model_size_scope=model.get('model_size_scope'),method=method,
                    independent_native_qualification_passed=model['qualification']['passed'],
                    quality_measured=model['quality_measured'],quality_passed=quality.get('passed'),
                    nll_delta=quality.get('nll_delta'),mean_kl=quality.get('mean_kl'),
                    top1_agreement=quality.get('top1_agreement'),logit_relative_l2=quality.get('relative_l2'),
                    candidate_perplexity=quality.get('candidate_perplexity'),reference_perplexity=quality.get('reference_perplexity'),
                    streamed_model_mean_ms=streamed.get('mean_ms'),streamed_model_sample_count=streamed.get('sample_count'),
                    streamed_model_ci95_ms=json.dumps(streamed.get('latency_ci95_ms')),
                    pooled_resident_layer_mean_ms=resident.get('mean_ms'),
                    timing_caveat=model.get('streamed_caveat','No full-model timing claim available'),
                    evidence=str(art/'model_summary.json')))
            continue
        if campaign_id not in SCOPES and not (art/'confirmation_statistics.json').exists():
            item['scope']='Application/preparation evidence outside generic projection comparison';continue
        item['scope']=SCOPES.get(campaign_id,'Real-weight projection measurements; consult application-stage scope')
        inventory.setdefault(campaign_id,dict(expected_shapes=len(manifest['shapes']),confirmed_shapes=set(),scope=item['scope']))
        identity=read(art/'environment.json')['identity'];identities.append(identity)
        raw=[json.loads(line)for line in (art/'results.jsonl').read_text().splitlines()]
        cases=[r for r in raw if r['event']=='case_result'];events_by_phase[phase_id]=raw
        item.update(stage=summary.get('stage'),outcomes=len(cases),status_counts=dict(Counter(r['status']for r in cases)),
            raw_samples=sum(r['event']=='sample'for r in raw))
        if summary.get('stage')!='confirm' or not (art/'confirmation_statistics.json').exists():continue
        statistics_by_shape=read(art/'confirmation_statistics.json')['by_shape'];selection=read(art/'selection_used.json')['selected']
        groups=read(art/'planned_cases.json');group_by_shape={g['shape']['id']:g for g in groups}
        for shape_id,stats in statistics_by_shape.items():
            key=campaign_id,shape_id
            if key in seen:raise ValueError('Duplicate confirmation requires explicit adjudication, not automatic pooling: '+str(key))
            seen.add(key);inventory[campaign_id]['confirmed_shapes'].add(shape_id)
            group=group_by_shape[shape_id];shape=group['shape']
            sampling=sorted({r.get('sampling_group','')for r in shape.get('source_rows',[])if r.get('sampling_group')})
            for method,label in LABELS.items():
                info=stats['headline'].get(method,{'available':False});candidate_id=info.get('candidate_id')
                observed=[r for r in cases if r.get('group_id')==group['group_id']and r.get('arm_id')==candidate_id]
                samples=[r['elapsed_ms']for r in raw if r['event']=='sample'and r.get('group_id')==group['group_id']and r.get('arm_id')==candidate_id]
                metrics=[r['correctness']for r in observed if r.get('correctness')]
                pair=info.get('comparison_vs_tuned_native')or {};ci=pair.get('ci95');ratio=pair.get('candidate_over_reference_latency_ratio')
                metadata=next((r['kernel_metadata']for r in observed if r.get('kernel_metadata')), {})
                accuracy=[v['relative_l2']for v in metrics if isinstance(v.get('relative_l2'),(int,float))]
                max_errors=[v['max_abs_error']for v in metrics if isinstance(v.get('max_abs_error'),(int,float))]
                rows.append(dict(campaign_id=campaign_id,scope=item['scope'],phase=phase_id,shape_id=shape_id,
                    m=shape['m'],k=shape['k'],n=shape['n'],sampling_groups=';'.join(sampling),method=method,label=label,
                    available=bool(info.get('available')),candidate_id=candidate_id,
                    tile_bm_bn_bk=json.dumps(metadata.get('tile_bm_bn_bk')),
                    compiler_options=json.dumps(metadata.get('compiler_options',{}),sort_keys=True),
                    mean_ms=statistics.mean(samples)if samples else None,sample_count=len(samples),
                    input_count=len(observed),status_counts=json.dumps(dict(Counter(r['status']for r in observed)),sort_keys=True),
                    screen_selection_eligible=info.get('screen_selection_eligible'),all_confirmation_inputs_eligible=info.get('all_inputs_eligible',False),
                    valid_comparison_vs_tuned_native=info.get('valid_numerical_comparison_vs_tuned_native',False),
                    speedup_vs_tuned_native=1/ratio if ratio else None,
                    speedup_ci95_low=1/ci[1]if ci else None,speedup_ci95_high=1/ci[0]if ci else None,
                    relative_l2_min=min(accuracy)if accuracy else None,relative_l2_mean=statistics.mean(accuracy)if accuracy else None,
                    relative_l2_max=max(accuracy)if accuracy else None,max_abs_error_max=max(max_errors)if max_errors else None,
                    reference_scopes=';'.join(sorted({v['reference_scope']for v in metrics})),
                    reference_backend=';'.join(sorted({v['reference_backend']for v in metrics})),
                    full_output_finite=all(v['finite']for v in metrics)if metrics else None,
                    allocation_id=identity['allocation_id'],raw_evidence=str(art/'results.jsonl'),
                    frozen_selection=str(art/'selection_used.json'),confidence_evidence=str(art/'confirmation_statistics.json')))
            for family,detail in stats['tile_uncertainty'].items():
                for candidate in detail['confirmed_candidates']:
                    interval=candidate.get('comparison_to_frozen_winner')or {};bounds=interval.get('ci95')or [None,None]
                    shortlist.append(dict(campaign_id=campaign_id,shape_id=shape_id,m=shape['m'],k=shape['k'],n=shape['n'],
                        family=family,candidate_id=candidate['candidate_id'],tile_bm_bn_bk=json.dumps(candidate['tile']),
                        frozen_winner=(detail.get('frozen_winner')or {}).get('candidate_id'),
                        latency_ratio_to_frozen_winner=interval.get('candidate_over_reference_latency_ratio'),
                        ci95_low=bounds[0],ci95_high=bounds[1],all_inputs_eligible=candidate.get('all_inputs_eligible'),
                        consistent_with_equal_latency=candidate['consistent_with_equal_latency'],
                        within_five_percent_upper_bound=candidate['within_five_percent_upper_bound'],
                        omitted_by_cap_count=len(detail['omitted_by_cap']),evidence=str(art/'confirmation_statistics.json')))
    if identities and any(identity!=identities[0]for identity in identities[1:]):
        raise ValueError('Environment/allocation mismatch: do not pool phases silently')
    planned=[]
    if (cohort/'plan.json').exists():
        planned=[s['id']for s in read(cohort/'plan.json')['stages']]
    received={p['phase']for p in phases}
    coverage={key:dict(expected_shapes=value['expected_shapes'],confirmed_shapes=len(value['confirmed_shapes']),
                       scope=value['scope'])for key,value in inventory.items()}
    result=dict(schema_version=1,cohort=str(cohort),allocation_id=identities[0]['allocation_id']if identities else None,
        confirmed_method_rows=len(rows),coverage=coverage,phase_inventory=phases,actual_model_stages=len(applications),
        planned_stages_without_phase_receipt=[x for x in planned if x not in received],
        limitations=['Frozen screen winners only; no confirmation reselection.',
                    'Pointwise paired bootstrap intervals conditional on the frozen shortlist; no global optimum guarantee.',
                    'Accuracy uses exact BF16 operands and full or sampled all-K reference; sampled max error is not full-output max error.',
                    'Development shapes, LLM-derived synthetic shapes, and real model execution have different scopes.',
                    'Missing and failed phases remain incomplete; no old runs or retry samples are pooled.'])
    output.mkdir(parents=True)
    write_csv(output/'headline_comparisons.csv',rows);write_csv(output/'tile_uncertainty.csv',shortlist)
    write_csv(output/'application_comparisons.csv',application_rows);dump(output/'application_results.json',applications)
    dump(output/'summary.json',result)
    lines=['# MLSys v5e campaign results','',f'Cohort: `{cohort.name}`.',
           f'Allocation: `{result["allocation_id"] or "No completed measurement phase"}`.','',
           'This report uses only sealed phase evidence. It preserves the screen-selected configurations; it does not select the fastest confirmation observation.','',
           '## Coverage','']
    if coverage:
        lines+=['| Study | Confirmed shapes | Planned shapes |','|---|---:|---:|']
        for key,value in coverage.items():lines.append(f'| {key} | {value["confirmed_shapes"]} | {value["expected_shapes"]} |')
    else:lines.append('No completed shape confirmation is available.')
    lines+=['','## Frozen choices and accuracy','',
            'The [complete comparison table](headline_comparisons.csv) contains every confirmed shape and all five labels, with raw-evidence links, timings, paired speedup intervals, numerical errors, and eligibility.',
            'The [tile uncertainty table](tile_uncertainty.csv) contains each confirmed finalist and its latency interval relative to the frozen winner. Near tiles are discrete tested tuples.','',
            '| Study | Shape M,K,N | Method | Mean ms | Speedup vs tuned Native (95% CI) | Max relative L2 across inputs | Eligible comparison |',
            '|---|---|---|---:|---|---:|---|']
    for row in rows:
        number=lambda value: '—'if value is None else f'{value:.4g}'
        speed=number(row['speedup_vs_tuned_native'])
        if row['speedup_ci95_low']is not None:speed+=f' [{number(row["speedup_ci95_low"])}, {number(row["speedup_ci95_high"])}]'
        lines.append(f'| {row["campaign_id"]} | {row["m"]},{row["k"]},{row["n"]} | {row["label"]} | {number(row["mean_ms"])} | {speed} | {number(row["relative_l2_max"])} | {row["valid_comparison_vs_tuned_native"]} |')
    lines+=['','## Actual model execution','',
            'The [application comparison table](application_comparisons.csv) and [complete model summaries](application_results.json) preserve quality qualification and execution scope separately from synthetic matrix tests. Streamed full-model measurements include checkpoint reads and device transfers; three repeats are descriptive, not a fully resident serving benchmark.','']
    for app in applications:
        model=app['result']
        lines.append(f'- {model["model_id"]}: {model["status"]}; quality measured: {model["quality_measured"]}; [evidence]({app["evidence"]}).')
    if not applications:lines.append('No sealed actual-model execution summary is available yet.')
    lines+=['','## Interpretation limits','']+[f'- {text}'for text in result['limitations']]
    lines+=['','## Phase status','']
    for phase in phases:lines.append(f'- {phase["phase"]}: {phase["state"]}; [evidence]({phase["summary"]}).')
    if result['planned_stages_without_phase_receipt']:
        lines+=['','Stages without a phase receipt (includes independent local analysis commands): '+', '.join(result['planned_stages_without_phase_receipt'])+'.']
    with (output/'RESULTS.md').open('x')as stream:stream.write('\n'.join(lines)+'\n')
    dump(output/'artifact-manifest.json',dict(sha256={p.name:sha(p)for p in sorted(output.iterdir())if p.is_file()}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    output=args.output_dir
    if output is None:
        if not os.environ.get('STRASSEN_EXECUTION_DIR'):raise ValueError('Use --output-dir or the scoped archive runner')
        output=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    result=report(args.cohort,output)
    print(json.dumps(dict(coverage=result['coverage'],confirmed_method_rows=result['confirmed_method_rows'],output=str(output)),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
