"""M,N,K-only empirical selector for the frozen one-v5e power-grid catalog.

Runtime inference uses only Python's standard library. Coefficients are fitted
outside this module; no benchmark or timing lookup is performed at inference.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

VERSION = 'power_grid_selector_v001'
FEATURE_NAMES = ('intercept', 'padded_volume_Telements', 'padded_output_100Melements',
                 'padded_A_100Melements', 'padded_B_100Melements', 'copy_proxy_100Mbytes',
                 'padding_indicator')


def dimensions(m, n, k):
    values = (m, n, k)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('M,N,K must be positive integers (not booleans or rounded floats)')
    return values


def geometry(m, n, k, candidate):
    dimensions(m, n, k)
    tile = candidate.get('tile')
    if candidate['family'] == 'native':
        if tile is not None:
            raise ValueError('Native tiles must be compiler-managed')
        mp, np_, kp = m, n, k
        output_tiles = panels = None
    else:
        if not isinstance(tile, (list, tuple)) or len(tile) != 3 or any(type(v) is not int or v <= 0 for v in tile):
            raise ValueError('Custom tile must be a positive integer BM,BN,BK tuple')
        bm, bn, bk = tile
        mp, np_, kp = ((v+b-1)//b*b for v,b in zip((m,n,k),tile))
        output_tiles = (mp//bm)*(np_//bn)
        panels = kp//bk
    padded = (mp,np_,kp) != (m,n,k)
    # A source-level read/write/crop proxy; not measured hardware traffic.
    copy = 2*(m*k+k*n+mp*kp+kp*np_) + 4*(mp*np_+m*n) if padded else 0
    original = 2*(m*k+k*n)
    pad_inputs = 2*(mp*kp+kp*np_)
    # Mirrors the campaign's conservative preflight for one candidate.
    memory = original + max(original,pad_inputs) + (pad_inputs if padded else 0) + 8*mp*np_ + 8*128*128
    return {'padded_m':mp,'padded_n':np_,'padded_k':kp,'padding_required':padded,
            'padding_volume_ratio':mp*np_*kp/(m*n*k),'copy_proxy_bytes':copy,
            'logical_output_tiles':output_tiles,'sequential_k_panels':panels,
            'conservative_single_candidate_bytes':memory}


def features(m, n, k, candidate, kind='affine6', knot=None):
    g = geometry(m,n,k,candidate)
    mp,np_,kp = (g[a] for a in ('padded_m','padded_n','padded_k'))
    x = [1.,mp*np_*kp/1e12,mp*np_/1e8,mp*kp/1e8,kp*np_/1e8,g['copy_proxy_bytes']/1e8]
    if kind == 'piecewise7':
        if knot is None or knot <= 0:
            raise ValueError('Positive volume knot required')
        v = x[1]
        x[1:2] = [min(v,knot/1e12),max(v-knot/1e12,0.)]
    elif kind == 'affine7':
        x.append(float(g['padding_required']))
    elif kind != 'affine6':
        raise ValueError('Unknown cost feature version')
    return x


def predict_candidate(m, n, k, candidate, model):
    x = features(m,n,k,candidate,model['feature_kind'],model.get('volume_knot'))
    beta = model['coefficients'][candidate['candidate_id']]
    if len(x) != len(beta) or any(not math.isfinite(v) or v < 0 for v in beta):
        raise ValueError('Invalid nonnegative model coefficients')
    value = math.fsum(a*b for a,b in zip(x,beta))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Model produced invalid latency')
    return value


def select(model, m, n, k, tolerance=.05):
    dimensions(m,n,k)
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError('Tolerance must be finite and in [0,1]')
    budget = model['memory_budget_bytes']
    ranked, exclusions = [], []
    for candidate in model['candidates']:
        g = geometry(m,n,k,candidate)
        row = {**candidate, **g}
        if g['conservative_single_candidate_bytes'] > budget:
            exclusions.append({'candidate_id':candidate['candidate_id'],'reason':'conservative_memory_preflight',
                               'estimated_bytes':g['conservative_single_candidate_bytes']})
            continue
        row['predicted_ms'] = predict_candidate(m,n,k,candidate,model)
        row['tile_status'] = 'compiler_managed' if candidate['family']=='native' else 'tested_catalog_tuple'
        ranked.append(row)
    ranked.sort(key=lambda r:(r['predicted_ms'],r['candidate_id']))
    logs = [math.log2(v) for v in (m,n,k)]
    distance = min(math.sqrt(sum((a-math.log2(b))**2 for a,b in zip(logs,s))) for s in model['training_shapes_mnk'])
    on_shape = [m,n,k] in model['training_shapes_mnk']
    return {'version':VERSION,'input_mnk':[m,n,k],
            'status':'predicted_choice' if ranked else 'no_candidate_within_conservative_memory_budget',
            'selection':ranked[0] if ranked else None,
            'per_family_best':{f:next((r for r in ranked if r['family']==f),None) for f in ('native','cubic','strassen')},
            'per_family_top3':{f:[r for r in ranked if r['family']==f][:3] for f in ('native','cubic','strassen')},
            'per_family_predicted_near_set':{f:[r for r in ranked if r['family']==f and
                r['predicted_ms'] <= min(v['predicted_ms'] for v in ranked if v['family']==f)*(1+tolerance)]
                for f in ('native','cubic','strassen')},
            'predicted_near_set':[r for r in ranked if r['predicted_ms'] <= ranked[0]['predicted_ms']*(1+tolerance)] if ranked else [],
            'predicted_near_tolerance':tolerance,'ranked_candidates':ranked,'exclusions':exclusions,
            'support':{'is_training_geometry':on_shape,'nearest_training_log2_euclidean_distance':distance,
                       'within_observed_axis_bounds':all(1 <= v <= 131072 for v in (m,n,k))},
            'interpretation':'Empirical latency minimizer over the qualified catalog; not a proof of global optimality. Predicted near sets are not measured confidence-certified sets.',
            'calibration_contract':model['calibration_contract']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--m',type=int,required=True)
    parser.add_argument('--n',type=int,required=True)
    parser.add_argument('--k',type=int,required=True)
    parser.add_argument('--tolerance',type=float,default=.05)
    args=parser.parse_args()
    model=json.loads(args.model.read_text())
    print(json.dumps(select(model,args.m,args.n,args.k,args.tolerance),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
