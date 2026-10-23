"""Deterministic candidate registry and auditable, batch-normalized selection.

No JAX import: candidate decisions are reproducible before allocating a TPU.
This is a bounded development-set tuner, not an unseen-input accuracy policy.
"""
import math
import random
import statistics


def expanded(shapes):
    return [{**s, 'base_shape_id': s['id'], 'id': s['id'] + '__' + dtype,
             'output_dtype': dtype} for s in shapes for dtype in ('float32', 'bfloat16')]


def arm(tile=None, depth=0, accumulator='products', buffers=2, dtype='float32', mib=None):
    name = ('Native_default' if mib is None else f'Native_{mib}MiB') if depth == 0 else (
        f'S{depth}_{accumulator}_{tile[0]}_{tile[1]}_{tile[2]}_b{buffers}')
    return dict(arm_id=name, candidate_id=name, variant=name, family='native' if depth == 0 else f's{depth}',
        algorithm='native' if depth == 0 else 'current', implementation='native' if depth == 0 else 'current',
        depth=depth, accumulator=accumulator, tile=list(tile) if tile else None, buffers=buffers,
        output_dtype=dtype, vmem_limit_bytes=112*1024**2,
        compiler_options={} if mib is None else {'xla_tpu_scoped_vmem_limit_kib': mib*1024})


def registry(s, cfg):
    dtype = s['output_dtype']; m, k, n = (s[d] for d in ('m', 'k', 'n'))
    controls = s['historical_controls'][dtype]
    historical = {(tuple(c['tile']), c['depth'], c['buffers']) for c in controls}
    geometries = set(tuple(t) for t in cfg['search']['output_tiles'])
    geometries.update(tuple(c['tile'][:2]) for c in controls)
    if n % 2560 == 0:
        geometries.add((1024, 2560))
    divisors = [v for v in range(512, k+1, 512) if k % v == 0]
    panels = {k, *[c['tile'][2] for c in controls]}
    for ceiling in (1024, k//4, k//2):
        permitted = [v for v in divisors if v <= ceiling]
        if permitted:
            panels.add(max(permitted))
    offered = [arm(dtype=dtype, mib=mib) for mib in cfg['search']['native_mib']]
    decisions = []; known = set()
    for bm, bn in sorted(geometries):
        for bk in sorted(panels):
            for depth in (1, 2):
                for mode in ('products', 'outputs'):
                    for buffers in (1, 2):
                        a = arm((bm, bn, bk), depth, mode, buffers, dtype)
                        assert a['arm_id'] not in known
                        known.add(a['arm_id'])
                        protected = (tuple(a['tile']), depth, buffers) in historical and mode == 'products'
                        # Deliberately a rough scheduling estimate, not a lower bound:
                        # compilers may reuse invariant inputs, remove buffers, or spill.
                        scratch = 7*bm*bn if mode == 'products' else (4*bm*bn if dtype == 'bfloat16' else 4)
                        estimate = buffers*2*(bm*bk + bk*bn) + scratch + 2*(2 if dtype == 'bfloat16' else 4)*bm*bn
                        reasons = []
                        if m % bm or n % bn or k % bk:
                            reasons.append('nondivisible_shape; padding excluded')
                        if bm % (8*2**depth) or bn % (128*2**depth) or bk % (128*2**depth):
                            reasons.append('recursive_leaf_alignment')
                        if estimate > cfg['search']['estimate_prune_mib']*1024**2 and not protected:
                            reasons.append('rough_memory_estimate_exceeds_search_cap; heuristic_not_proven_infeasible')
                        decision = dict(candidate=a, disposition='pruned' if reasons else 'offered', reasons=reasons,
                            estimated_vmem_bytes=estimate, historical_control=protected,
                            contraction='full' if bk == k else 'short',
                            accumulator_storage=('seven_outer_products' if mode == 'products' else
                                'four_quadrants' if dtype == 'bfloat16' else 'output_tile'))
                        decisions.append(decision)
                        if not reasons:
                            offered.append(a)
    return offered, dict(shape=s, output_geometries=[list(x) for x in sorted(geometries)],
        all_aligned_k_divisors=divisors, offered_k_panels=sorted(panels),
        omitted_k_divisors=[v for v in divisors if v not in panels],
        panel_rule=cfg['candidate_policy'], candidates=decisions)


def screen_groups(s, cfg):
    offered, _ = registry(s, cfg)
    native = [a for a in offered if not a['depth']]
    custom = [a for a in offered if a['depth']]
    random.Random(cfg['order_seed'] + s['seed_offset']).shuffle(custom)
    anchor = next(a for a in native if a['arm_id'] == 'Native_default')
    size = cfg['search']['batch_size']
    batches = [native] + [[anchor, *custom[i:i+size]] for i in range(0, len(custom), size)]
    return [dict(group_id=s['id']+'__screen_'+str(i), shape=s, tile=None, arms=batch,
        scopes=['call'], timing='screen', inputs=[dict(distribution='gaussian', seed=cfg['screen_seed']+s['seed_offset'])])
        for i, batch in enumerate(batches)]


def select(cases, cfg, shapes):
    selected = {}
    for s in shapes:
        offered, trace = registry(s, cfg)
        registered = {a['arm_id']: a for a in offered}
        rows = [r for r in cases if r['shape_id'] == s['id']]
        anchors = {r['group_id']: r for r in rows if r['arm_id'] == 'Native_default'}
        scores = {}
        def eligible(r):
            return (r['status'] == 'ok' and r['eligible_for_speedup_claim'] and
                (r.get('timing') or {}).get('sample_count') == cfg['timing']['screen']['repeats'] and
                math.isfinite(r['timing']['mean_ms']) and r['timing']['mean_ms'] > 0)
        for a in offered:
            observations = [r for r in rows if r['arm_id'] == a['arm_id']]
            valid = [r for r in observations if eligible(r) and eligible(anchors[r['group_id']])]
            score = statistics.median(r['timing']['mean_ms']/anchors[r['group_id']]['timing']['mean_ms'] for r in valid) if valid else None
            scores[a['arm_id']] = dict(candidate=a, score=score,
                status='eligible' if valid and len(valid)==len(observations) else 'excluded',
                screen_mean_ms=statistics.mean(r['timing']['mean_ms'] for r in valid) if valid else None,
                observations=[dict(group_id=r['group_id'], status=r['status'], eligible=eligible(r),
                    mean_ms=(r.get('timing') or {}).get('mean_ms'),
                    anchor_ms=(anchors[r['group_id']].get('timing') or {}).get('mean_ms'),
                    correctness=r.get('correctness')) for r in observations])
        choices = {}
        filters = {'native': lambda a: not a['depth'], 'overall': lambda a: True}
        for depth in (1, 2):
            filters[f's{depth}'] = lambda a, d=depth: a['depth']==d
            for mode in ('products', 'outputs'):
                for contraction in ('short', 'full'):
                    filters[f's{depth}_{mode}_{contraction}'] = lambda a, d=depth, mode=mode, c=contraction: (
                        a['depth']==d and a['accumulator']==mode and (a['tile'][2]==s['k'])==(c=='full'))
        for role, accepts in filters.items():
            pool = [v for v in scores.values() if v['status']=='eligible' and accepts(v['candidate'])]
            if pool:
                best = min(pool, key=lambda v: (v['score'], v['candidate']['arm_id']))
                choices[role] = {k:best[k] for k in ('candidate','score','screen_mean_ms','status')}
                choices[role]['reason'] = 'lowest_eligible_latency_divided_by_same_batch_native; arm_id_breaks_exact_ties'
            else:
                choices[role] = dict(status='no_eligible_candidate', reason='all_candidates_failed_or_unmeasured')
        if 'candidate' not in choices['native']:
            raise ValueError('No eligible Native baseline: '+s['id'])
        selected[s['id']] = dict(choices=choices, candidate_scores=scores, registry=trace,
            selection_rule=cfg['selection_policy'])
    return selected


def confirm_group(s, cfg, selection):
    chosen = selection[s['id']]['choices']; offered, _ = registry(s, cfg)
    by_id = {}; registered = {a['arm_id']:a for a in offered}
    def add(a, role):
        value = by_id.setdefault(a['arm_id'], dict(a, headline_roles=[]))
        if role not in value['headline_roles']:
            value['headline_roles'].append(role)
    for role, value in chosen.items():
        if 'candidate' in value:
            add(value['candidate'], role)
    add(registered['Native_default'], 'native_default')
    for c in s['historical_controls'][s['output_dtype']]:
        a = arm(c['tile'], c['depth'], 'products', c['buffers'], s['output_dtype'])
        add(registered[a['arm_id']], c['role'])
    # Matched counterfactuals explain the S1/S2 choice without reselecting it.
    # These may OOM; that is useful evidence, not grounds to modify the winner.
    for depth in (1, 2):
        a = chosen[f's{depth}'].get('candidate')
        if a:
            add({**a, 'arm_id':f'Matched_cubic_s{depth}', 'candidate_id':f'Matched_cubic_s{depth}',
                 'implementation':'cubic', 'family':'cubic', 'algorithm':'cubic', 'depth':0}, f'matched_s{depth}')
            for label, mode, buffers in [('other_accumulator', 'outputs' if a['accumulator']=='products' else 'products', a['buffers']),
                                         ('other_buffers', a['accumulator'], 3-a['buffers'])]:
                add(arm(a['tile'], depth, mode, buffers, s['output_dtype']), f's{depth}_{label}')
    return dict(group_id=s['id']+'__confirm', shape=s, tile=None, arms=list(by_id.values()), scopes=['call'],
        timing='confirm', inputs=[dict(distribution='gaussian',seed=v+s['seed_offset']) for v in cfg['confirm_seeds']])


def groups(cfg, shapes, stage, selection=None, start=0, count=None):
    if stage == 'smoke':
        result = []
        for dtype in ('float32', 'bfloat16'):
            s = dict(id='integer_'+dtype, m=256, k=1536, n=1024, output_dtype=dtype)
            arms = [arm(dtype=dtype)]
            for depth in (1, 2):
                for mode in ('products', 'outputs'):
                    for buffers in (1, 2):
                        for bk in (512, 1536):
                            arms.append(arm((128,512,bk),depth,mode,buffers,dtype))
            result.append(dict(group_id=s['id']+'__smoke',shape=s,tile=None,arms=arms,scopes=['call'],
                timing='smoke',inputs=[dict(distribution='integer',seed=9260000)]))
        return result
    subset = expanded(shapes[start:None if count is None else start+count])
    if stage == 'screen':
        return [g for s in subset for g in screen_groups(s, cfg)]
    return [confirm_group(s,cfg,selection) for s in subset]
