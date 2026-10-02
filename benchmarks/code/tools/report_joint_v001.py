"""Report frozen joint choices with measured reasons and scope-limited policy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from report_v6e_suite_v001 import paired


def recommendation(methods, rule):
    native = methods['native']; winner = methods['overall']
    if not native['eligible']:
        return dict(status='unavailable',reason='native_confirmation_ineligible',candidate=None)
    if winner['candidate']['depth'] == 0:
        return dict(status='native',reason='native_won_screening',candidate=native['candidate'])
    comp = winner['comparisons'].get('native')
    if not winner['eligible']:
        reason = 'frozen_winner_failed_confirmation_or_numerical_gate'
    elif not comp or comp['speedup'] < rule['minimum_mean_speedup']:
        reason = 'frozen_winner_did_not_clear_1_percent_mean_margin'
    elif comp['ci95'][0] <= rule['minimum_ci95_lower']:
        reason = 'frozen_winner_speedup_not_resolved_by_paired_95_percent_interval'
    else:
        return dict(status='custom',reason='frozen_winner_passed_errors_and_confirmed_speed_margin',
                    candidate=winner['candidate'],comparison_to_native=comp)
    return dict(status='native',reason=reason,candidate=native['candidate'],rejected_candidate=winner['candidate'])


def build(cohort, out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((cohort/'source/configs/joint_v001/campaign.json').read_text())
    shapes=json.loads((cohort/'source/configs/joint_v001/shapes.json').read_text())['shapes']
    hashes={}; missing=[]; results=[]; ledgers={}; counts={k:Counter() for k in ('screen','confirm')}
    def read(p):
        raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    for index,s in enumerate(shapes,1):
        receipts={stage:cohort/f'joint-{index:02d}-{stage}-finished.json' for stage in counts}
        if not all(p.exists() for p in receipts.values()):
            missing.append(s['id']);continue
        roots={stage:Path(read(p)['run'])/'artifacts' for stage,p in receipts.items()}
        records={}
        for stage,root in roots.items():
            p=root/'results.jsonl';raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest()
            records[stage]=[json.loads(line) for line in raw.splitlines()]
            counts[stage].update(r['status'] for r in records[stage] if r['event']=='case_result')
        selected=read(roots['screen']/'selections.json')['selected'];ledgers.update(selected)
        environment=read(roots['confirm']/'environment.json')
        for g in read(roots['confirm']/'planned_cases.json'):
            cases=[r for r in records['confirm'] if r['event']=='case_result' and r['shape_id']==g['shape']['id']]
            samples=[r for r in records['confirm'] if r['event']=='sample' and r['shape_id']==g['shape']['id']]
            raw={a['arm_id']:{(r['seed'],r['round']):r['elapsed_ms'] for r in samples if r['arm_id']==a['arm_id']} for a in g['arms']}
            methods={}
            for a in g['arms']:
                rows=[r for r in cases if r['arm_id']==a['arm_id']]
                metrics=[r['correctness'] for r in rows if r.get('correctness')]
                complete=len(rows)==3 and len(raw[a['arm_id']])==90
                errors={key:max(q[key] for q in metrics) for key in
                    ('relative_l2','max_abs_error','rmse','mean_abs_error','p50_abs_error','p99_abs_error','normwise_error')
                    if metrics and all(isinstance(q.get(key),(int,float)) and math.isfinite(q[key]) for q in metrics)}
                for role in a['headline_roles']:
                    methods[role]=dict(candidate=a,complete=complete,eligible=complete and all(r['eligible_for_speedup_claim'] for r in rows),
                        mean_ms=statistics.mean(raw[a['arm_id']].values()) if raw[a['arm_id']] else None,
                        errors=errors,case_statuses=[r['status'] for r in rows],comparisons={})
            for role,m in methods.items():
                for ref,b in methods.items():
                    if ref not in ('native','native_default','s1_other_accumulator','s2_other_accumulator','s1_other_buffers','s2_other_buffers','matched_s1','matched_s2') and not ref.startswith('prior_'):
                        continue
                    comparison=paired(raw[b['candidate']['arm_id']],raw[m['candidate']['arm_id']])
                    if comparison:
                        comparison['eligible']=m['eligible'] and b['eligible'];m['comparisons'][ref]=comparison
            decision=recommendation(methods,cfg['deployment_rule'])
            results.append(dict(shape=g['shape'],methods=methods,recommendation=decision,
                selection=selected[g['shape']['id']]['choices'],identity=environment['identity'],
                terminal_outcomes=len(cases),expected_outcomes=3*len(g['arms'])))
    complete=not missing and len(results)==2*len(shapes) and all(r['terminal_outcomes']==r['expected_outcomes'] for r in results)
    policy=dict(schema_version=1,scope=cfg['deployment_rule'],precision=cfg['precision'],correctness_gate=cfg['correctness'],
        source='screen-frozen recommendations gated by fresh confirmation; no confirmation reselection',
        entries=[dict(shape_mkn=[r['shape'][d] for d in ('m','k','n')],output_dtype=r['shape']['output_dtype'],
            identity=r['identity'],**r['recommendation']) for r in results],
        installation='Research artifact only; not installed as a general application dispatcher')
    lines=['# Joint MM tuning results and decisions','',
        'Six development geometries, separate FP32 and BF16 output. BF16 inputs/pre-adds; FP32 accumulation/reconstruction; final output store included in time and error.',
        'Screening ranks each candidate against the repeated Native_default anchor in its compilation batch. Choices are frozen before three fresh inputs × 30 paired confirmation rounds.',
        'Recommendations retain the frozen overall winner only if all error checks pass, mean speedup over tuned Native is at least 1.01, and the pointwise paired 95% interval lies above 1. Otherwise they fall back to the frozen tuned Native; they never promote another confirmation finalist.',
        'All intervals are conditional on screening, with no multiple-comparison adjustment. Shapes and Gaussian input family are development data; numerical eligibility is not an arbitrary-input or model-quality guarantee.','']
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    def name(a):
        return 'Native' if not a['depth'] else f'S{a["depth"]} {a["accumulator"]}; {a["tile"]}; b{a["buffers"]}'
    for dtype in ('float32','bfloat16'):
        lines += ['## '+dtype+' output','',
            '| M × K × N | Recommended configuration | Native ms | Recommended ms | Speedup vs Native | Relative L2 % | Decision |',
            '|---|---|---:|---:|---:|---:|---|']
        for r in [x for x in results if x['shape']['output_dtype']==dtype]:
            d=r['recommendation'];m=r['methods'];a=d.get('candidate')
            chosen=m['overall'] if d['status']=='custom' else m['native']
            c=chosen['comparisons'].get('native',{})
            lines.append('| '+' × '.join(str(r['shape'][k]) for k in ('m','k','n'))+' | '+(name(a) if a else 'unavailable')+' | '+fmt(m['native']['mean_ms'])+' | '+fmt(chosen['mean_ms'])+' | '+fmt(c.get('speedup'))+' | '+fmt(100*chosen['errors'].get('relative_l2',float('nan')))+' | '+d['reason']+' |')
        lines.append('')
    lines += ['## Why each depth chose its configuration','',
        'Counterfactuals below change only accumulator strategy or buffer count at the selected tile. A ratio above 1 favors the selected configuration. They help separate measured effects from memory-based hypotheses. Counterfactual failures remain visible.','']
    for r in results:
        lines += ['### '+r['shape']['id'],'']
        for depth in (1,2):
            role=f's{depth}';m=r['methods'].get(role)
            if not m:
                lines.append(f'- S{depth}: no eligible screening candidate.');continue
            lines.append(f'- S{depth}: {name(m["candidate"])}; confirmed {fmt(m["mean_ms"])} ms. Selected by the lowest eligible batch-normalized screening latency across both accumulator strategies, K-panel lengths and buffer counts.')
            for label in ('other_accumulator','other_buffers'):
                ref=role+'_'+label;c=m['comparisons'].get(ref);other=r['methods'].get(ref,{})
                lines.append('  - '+label+': '+(f'{c["speedup"]:.4f}× [{c["ci95"][0]:.4f}, {c["ci95"][1]:.4f}]' if c and c.get('eligible') else 'comparison ineligible/unavailable; outcomes '+str(other.get('case_statuses')))+'.')
        lines.append('')
    lines += ['## Coverage','',f'All planned outcomes present: {complete}.',str({k:dict(v) for k,v in counts.items()}),
        'candidate_decisions.json retains offered/pruned configurations, memory estimates, omitted K divisors, numerical checks, screening anchors, ranks and frozen choices. results.json retains every confirmation arm and comparison, including previous controls. tuner_policy.json is a research recommendation table, not an installed serving dispatcher.',
        'Compiled HLO and cost metadata are preserved in each phase. Memory estimates are search heuristics; no performance cause is inferred from them alone.','']
    design=['# Tuner design choices','',
        '1. Treat hardware/software identity, input/accumulation/output precision and supported input family as part of the tuning contract. Keep FP32 and BF16 output independent.',
        '2. Search two accumulation strategies: seven retained outer products; or immediate outer-quadrant updates (direct FP32 output, four FP32 scratch quadrants for BF16). S2 always reconstructs its inner S1 within each K panel. This varies outer accumulator lifetime without changing the seven-product formula or BF16 pre-add contract.',
        '3. Search output geometry, K-panel length, accumulation strategy and one/two input buffers together. Include past winning geometries and panels so a narrower new menu cannot silently discard known candidates.',
        '4. Bound compile cost explicitly: a registered output-tile menu and selected dimension-dividing K panels. Record every omitted divisor and every heuristic memory prune. A rough footprint above 160 MiB prunes candidates against a 112 MiB custom allowance; this is intentionally loose and is not proof a compiler would reject the candidate. Historical controls bypass that estimate.',
        '5. Compile randomized batches of at most twelve custom candidates, each with Native_default. Normalize screen latency by that batch anchor to reduce sensitivity to drift. Retain all raw timings, compilation failures and numerical exclusions.',
        '6. Choose only complete measurements passing the frozen numerical gate. Choose the minimum normalized latency; break exact ties by stable candidate ID. Freeze overall, Native, S1/S2 and accumulator/contraction-stratum choices before confirmation.',
        '7. Confirm on three fresh Gaussian inputs and thirty paired rounds per input. Rerun historical controls, each selected depth at the other accumulator/buffer setting, and tile/buffer-matched blocked cubic. These measurements explain tradeoffs; they do not trigger reselection.',
        '8. Require a 1% mean improvement and a paired 95% lower bound above one for the frozen overall custom winner, plus all confirmation error checks. Otherwise recommend tuned Native. The 1% threshold is a declared practical margin, not a hardware law.',
        '9. Persist configuration, identity, rule, numerical scope and measured evidence. Use Native outside the validated contract. This study does not train or validate an unseen-shape prediction model, install a general dispatcher, or certify LLM quality.',
        '10. Explain choices using measured alternatives. Memory layout may suggest a mechanism, but proving stalls, overlap or spills requires compiled/profiler evidence beyond a simple footprint estimate.','',
        'The per-shape choices and measured counterfactuals are in RESULTS.md; full records are in candidate_decisions.json and results.json.']
    for filename,value in [('results.json',dict(completed=complete,results=results,missing=missing,case_status_counts={k:dict(v) for k,v in counts.items()},input_sha256=hashes)),('candidate_decisions.json',ledgers),('tuner_policy.json',policy)]:
        (out/filename).write_text(json.dumps(value,indent=2)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'TUNER_DESIGN.md').write_text('\n'.join(design)+'\n')
    print(json.dumps(dict(completed=complete,precision_groups=len(results),case_status_counts={k:dict(v) for k,v in counts.items()})))
    return 0 if complete else 1


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();raise SystemExit(build(a.cohort,a.output_dir))
