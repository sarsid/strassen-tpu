"""Replay frozen model outputs against source evidence and run focused tests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from strassen_mm.power_grid_selector_v001 import features,select
from fit_power_grid_model_v001 import load,fit,metric,confirmation_check

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_power_grid_selector_v001'))
    if not result.wasSuccessful():raise AssertionError('Focused unit checks failed')
    art=a.artifacts; model=json.loads((art/'model.json').read_text());v=json.loads((art/'validation.json').read_text())
    provenance=model['provenance'];screen=Path(provenance['screen_source']);confirmation=Path(provenance['confirmation_source'])
    checks=0
    def check(condition,message):
        nonlocal checks
        checks+=1
        if not condition:raise AssertionError(message)
    for name,digest in json.loads((art/'artifact_hashes.json').read_text()).items():
        check(hashlib.sha256((art/name).read_bytes()).hexdigest()==digest,'artifact hash '+name)
    check(hashlib.sha256(screen.read_bytes()).hexdigest()==provenance['screen_sha256'],'screen hash')
    check(hashlib.sha256(confirmation.read_bytes()).hexdigest()==provenance['confirmation_sha256'],'confirmation hash')
    ids,held,shape,dims,candidates,y,config=load(screen,ROOT/'configs/generated_power_grid_v001/sampled_shapes.json',ROOT/'configs/generated_power_grid_v001/campaign_power_grid_v001.json')
    cids=[c['candidate_id'] for c in candidates];families=np.array([c['family'] for c in candidates]);idx={s:i for i,s in enumerate(ids)}
    cididx={s:i for i,s in enumerate(cids)};preds=json.loads((art/'predictions.json').read_text());full=json.loads((art/'full_fit_predictions.json').read_text())
    check(len(preds)==3120 and len(full)==3120,'complete prediction tables')
    check(set(ids).isdisjoint(held),'reserved disjoint')
    check(model['training_shape_ids']==ids and model['training_shapes_mnk']==dims,'training geometry manifest')
    mat=np.zeros_like(y);seen=set()
    for fold in v['outer_folds']:
        tr=fold['training_ids'];te=fold['test_ids']
        check(len(tr)==48 and len(te)==12 and set(tr).isdisjoint(te) and set(tr)|set(te)==set(ids),'outer partitions')
        check(set(map(int,fold['inner']['assignment']))=={idx[s] for s in tr},'inner belongs only to outer training')
    for r in preds:
        i=idx[r['shape_id']];j=cididx[r['candidate_id']];key=(i,j)
        check(key not in seen,'duplicate prediction');seen.add(key)
        check(r['observed_ms']==y[i,j],'source mean preserved')
        fold=v['outer_folds'][r['fold']]
        check(r['shape_id'] in fold['test_ids'] and r['shape_id'] not in fold['training_ids'],'no exact-shape fit leakage')
        x=features(*dims[i],candidates[j],fold['spec']['feature_kind'],fold['spec']['volume_knot'])
        expected=float(np.dot(x,fold['coefficients'][r['candidate_id']]))
        check(np.isclose(expected,r['predicted_ms'],rtol=1e-12,atol=1e-12),'coefficient replay')
        mat[i,j]=r['predicted_ms']
    choice=mat.argmin(1)
    recomputed=metric(y,choice,families)
    check(recomputed==v['metrics']['nested_selector'],'metrics replay')
    for r in v['selections']:
        i=idx[r['shape_id']];check(cids[int(choice[i])]==r['selected_candidate'],'argmin selection')
        check(np.isclose(r['family_regret']*r['within_family_regret'],r['regret']),'regret factorization')
    check(confirmation_check(confirmation,v['selections'],held)==json.loads((art/'confirmation_coverage.json').read_text()),'confirmation same-group replay')
    for i,d in enumerate(dims):
        answer=select(model,*d)
        check(not answer['exclusions'],'all original candidates admitted')
        rank={r['candidate_id']:r['predicted_ms'] for r in answer['ranked_candidates']}
        for r in full[i*52:(i+1)*52]:
            check(r['shape_id']==ids[i] and np.isclose(rank[r['candidate_id']],r['predicted_ms'],rtol=1e-12),'stdlib full-fit inference replay')
    # Refit using the complete source to ensure serialized coefficients reproduce the fit.
    x=np.array([[features(*d,c,model['feature_kind'],model.get('volume_knot')) for c in candidates] for d in dims])
    coef,_=fit(x,y,np.arange(60))
    for j,cid in enumerate(cids):check(np.allclose(coef[j],model['coefficients'][cid],rtol=1e-12,atol=1e-12),'full fit coefficient replay')
    record={'status':'passed','unit_tests':result.testsRun,'evidence_checks':checks,'reserved_shapes_used':0,
            'model_artifacts':str(art),'model_sha256':hashlib.sha256((art/'model.json').read_bytes()).hexdigest(),
            'checks':'Source hashes, 3120 OOF predictions, whole-shape split exclusion, inner partition membership, source means, coefficient replay, regret metrics, direct confirmation group coverage, capacity, 3120 stdlib predictions, full-data refit.'}
    (a.output_dir/'checks.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
