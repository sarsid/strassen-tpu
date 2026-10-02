"""Freeze an M,N,K-only finite-catalog latency model with nested shape validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from strassen_mm.power_grid_selector_v001 import features, geometry, select

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260920
SPECS = [dict(name='affine6',feature_kind='affine6',volume_knot=None)] + [
    dict(name=f'piecewise7_knot2pow{p}',feature_kind='piecewise7',volume_knot=2**p) for p in (33,36,39)]


def write(path, data):
    with path.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nnls(a,b):
    """Lawson-Hanson active set, small dense nonnegative least squares.

    Inputs are normalized by the caller; KKT residual is returned for audit.
    Rank-deficient passive systems use NumPy's minimum-norm least squares.
    """
    n=a.shape[1]; x=np.zeros(n); passive=np.zeros(n,bool)
    tol=1e-10*max(1.,float(np.linalg.norm(a.T@b,np.inf)))
    iterations=0
    while True:
        grad=a.T@(b-a@x)
        available=np.where(~passive,grad,-np.inf)
        if np.max(available)<=tol: break
        passive[int(np.argmax(available))]=True
        while True:
            iterations+=1
            if iterations>500: raise RuntimeError('NNLS active set failed to converge')
            z=np.zeros(n); z[passive]=np.linalg.lstsq(a[:,passive],b,rcond=None)[0]
            if np.all(z[passive]>0):
                x=z; break
            bad=passive & (z<=0)
            denom=x[bad]-z[bad]
            ratios=np.divide(x[bad],denom,out=np.zeros_like(denom),where=denom>0)
            alpha=float(np.min(ratios))
            x=x+alpha*(z-x)
            leaving=passive & (x<=1e-14)
            x[leaving]=0; passive[leaving]=False
    grad=a.T@(b-a@x)
    residual=max(float(np.max(np.maximum(grad,0))),float(np.max(np.abs(grad[x>1e-12]))) if np.any(x>1e-12) else 0.)
    if residual>1e-7: raise RuntimeError(f'NNLS KKT residual {residual}')
    return x,residual,iterations


def fit(x,y,train):
    coefs=[]; diagnostics=[]
    for c in range(y.shape[1]):
        a=x[train,c]/y[train,c,None]
        scale=np.sqrt(np.mean(a*a,axis=0)); scale=np.where(scale>1e-14,scale,1.)
        a=a/scale/math.sqrt(len(train)); b=np.ones(len(train))/math.sqrt(len(train))
        beta,kkt,it=nnls(a,b)
        coefs.append(beta/scale)
        diagnostics.append({'candidate_index':c,'kkt_residual':kkt,'active_iterations':it,
                            'normalized_design_rank':int(np.linalg.matrix_rank(a)),
                            'columns':a.shape[1],'training_rms_scale':scale.tolist()})
    return np.array(coefs),diagnostics


def folds_for(indices,nfold,seed):
    values=np.empty(len(indices),int)
    values[np.random.default_rng(seed).permutation(len(indices))]=np.arange(len(indices))%nfold
    return {int(i):int(f) for i,f in zip(indices,values)}


def predict(x,coef,indices):
    p=np.einsum('nci,ci->nc',x[indices],coef)
    if not np.isfinite(p).all() or np.min(p)<=0: raise ValueError('Invalid predicted cost')
    return p


def metric(y,choice,families):
    n=len(y); oracle=y.argmin(1); obs=y[np.arange(n),choice]; best=y.min(1); r=obs/best
    return {'n':n,'geomean_regret':float(np.exp(np.log(r).mean())),
            'median_regret':float(np.median(r)),'p95_regret':float(np.quantile(r,.95)),
            'max_regret':float(r.max()),'within3_count':int(np.sum(r<=1.03)),
            'within5_count':int(np.sum(r<=1.05)),'within10_count':int(np.sum(r<=1.10)),
            'exact_candidate_count':int(np.sum(choice==oracle)),
            'exact_family_count':int(np.sum(families[choice]==families[oracle])),
            'total_latency_ratio':float(obs.sum()/best.sum()),
            'mean_excess_ms':float((obs-best).mean())}


def choose_spec(xs,y,indices,nfold,seed):
    assignment=folds_for(indices,nfold,seed)
    scores=[]
    for spec in SPECS:
        preds=np.zeros((len(indices),y.shape[1])); position={int(v):j for j,v in enumerate(indices)}
        for f in range(nfold):
            tr=np.array([i for i in indices if assignment[int(i)]!=f]); te=np.array([i for i in indices if assignment[int(i)]==f])
            coef,_=fit(xs[spec['name']],y,tr)
            preds[[position[int(i)] for i in te]]=predict(xs[spec['name']],coef,te)
        choice=preds.argmin(1); obs=y[indices][np.arange(len(indices)),choice]
        score=float(np.exp(np.log(obs/y[indices].min(1)).mean()))
        scores.append({'spec':spec,'inner_geomean_regret':score})
    # Stable spec order resolves exact ties: fewer terms, then smaller knot.
    best=min(range(len(scores)),key=lambda i:(scores[i]['inner_geomean_regret'],i))
    return SPECS[best],{'assignment':assignment,'scores':scores,'chosen':SPECS[best]}


def load(screen,manifest_path,config_path):
    manifest=json.loads(manifest_path.read_text()); config=json.loads(config_path.read_text())
    ids=sorted(manifest['exploratory_shape_ids']); held=set(manifest['holdout_shape_ids'])
    shape={r['id']:r for r in manifest['shapes'] if r['id'] in ids}
    candidates=sorted([dict(c,family=f,compiler_options=c.get('compiler_options',{})) for f,cs in
                      config['experiments']['GRID-screen']['candidate_families'].items() for c in cs],key=lambda c:c['candidate_id'])
    rows={}
    with screen.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('event')!='case_result' or r.get('scope')!='call': continue
            assert r['shape_id'] in ids and r['shape_id'] not in held
            assert r['status']=='ok' and r['correctness']['pass'] and r['eligible_for_speedup_claim']
            assert r['phase']=='GRID-screen' and r['seed']==20260921 and r['distribution']=='gaussian'
            assert r['timing']['sample_count']==7 and math.isfinite(r['timing']['mean_ms']) and r['timing']['mean_ms']>0
            key=(r['shape_id'],r['candidate_id']); assert key not in rows; rows[key]=r
    assert len(ids)==60 and len(candidates)==52 and len(rows)==3120
    dims=[[shape[s][d] for d in ('m','n','k')] for s in ids]
    for i,s in enumerate(ids):
        m,n,k=dims[i]
        for c in candidates:
            r=rows[(s,c['candidate_id'])];g=geometry(m,n,k,c)
            assert r['shape_mkn']==[m,k,n]
            assert r['tile']==c['tile'] and r['family']==c['family']
            assert r['algorithm']==c['algorithm'] and r['variant']==c['variant']
            assert r.get('compiler_options',{})==c['compiler_options']
            assert r['kernel_metadata']['padded_shape_mkn']==[g['padded_m'],g['padded_k'],g['padded_n']]
    y=np.array([[rows[(s,c['candidate_id'])]['timing']['mean_ms'] for c in candidates] for s in ids])
    return ids,held,shape,dims,candidates,y,config


def make_model(spec,coef,ids,dims,candidates,config,provenance):
    return {'model_version':'power_grid_model_v001',**spec,
            'coefficients':{c['candidate_id']:b.tolist() for c,b in zip(candidates,coef)},
            'candidates':candidates,'training_shape_ids':ids,'training_shapes_mnk':dims,
            'memory_budget_bytes':int(config['memory']['estimated_live_device_budget_gib']*2**30),
            'latency_units':'milliseconds','target':'complete-call arithmetic screen mean (7 rounds)',
            'calibration_contract':{'hardware':'one v5e chip, one allocation',
                'inputs':'Gaussian random BF16','preadds':'BF16','accumulation_output':'FP32',
                'scope':'device preparation/pad + matmul + crop; no host transfer, compilation or input generation',
                'strassen_levels_per_tile':1,'catalog':'52 frozen configurations',
                'caution':'Accuracy gate is input-specific; no accuracy-equivalence claim. Sparse shapes, not a full-domain guarantee.'},
            'fit_objective':'Sum over training shapes of (predicted_ms / measured_ms - 1)^2 per candidate, beta >= 0',
            'validation_status':'Experimental formula. Nested shape-CV did not beat a training-selected fixed native preset; use custom tile shortlists as development guidance, not guaranteed optimal dispatch.',
            'provenance':provenance}


def confirmation_check(path,selections,held):
    groups=defaultdict(list)
    with path.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('event')=='case_result' and r.get('scope')=='call':
                assert r['shape_id'] not in held and r['status']=='ok' and r['correctness']['pass']
                groups[(r['shape_id'],r['group_id'])].append(r)
    out=[]
    for s in selections:
        sid=s['shape_id'];cid=s['selected_candidate']; matches=[]
        for (shape,gid),rows in groups.items():
            if shape==sid and any(r['candidate_id']==cid for r in rows): matches.append((gid,rows))
        result={'shape_id':sid,'selected_candidate':cid,'covered':bool(matches)}
        if matches:
            matches.sort(key=lambda p:('headline_confirmation' not in p[0],p[0]))
            gid,rows=matches[0];r=next(r for r in rows if r['candidate_id']==cid)
            headline=next(rs for (ss,gg),rs in groups.items() if ss==sid and 'headline_confirmation' in gg)
            if 'headline_confirmation' in gid:
                ref=next(rr for rr in rows if rr['candidate_id']=='native_default')
                contrast='selected_vs_default_native_in_headline'
            else:
                family=r['family']; winner=next(rr for rr in headline if rr['family']==family)
                # Native headline also includes default. Prefer the family-selected nondefault when present.
                if family=='native':
                    native=[rr for rr in headline if rr['family']=='native' and rr['candidate_id']!='native_default']
                    if native: winner=native[0]
                ref=next(rr for rr in rows if rr['candidate_id']==winner['candidate_id'])
                contrast='selected_vs_frozen_family_winner_in_finalist_group'
            result.update(group_id=gid,contrast=contrast,reference_candidate=ref['candidate_id'],
                          observed_selected_ms=r['timing']['mean_ms'],observed_reference_ms=ref['timing']['mean_ms'],
                          selected_over_reference_ratio=r['timing']['mean_ms']/ref['timing']['mean_ms'],
                          samples_each=r['timing']['sample_count'])
        out.append(result)
    return {'interpretation':'Selective confirmation coverage of frozen out-of-fold actions. One canonical group per action. Point ratios only; not an all-candidate oracle or population estimate.',
            'covered_count':sum(r['covered'] for r in out),'total_shapes':len(out),'rows':out}


def near_diagnostics(y,p,families):
    result={}
    for family in ('native','cubic','strassen'):
        cols=np.flatnonzero(families==family); obs=y[:,cols]; pred=p[:,cols]
        band=pred<=1.05*pred.min(1)[:,None]; good=obs<=1.05*obs.min(1)[:,None]
        top3=np.argsort(pred,axis=1)[:,:min(3,len(cols))]
        result[family]={'mean_predicted_near5_set_size':float(band.sum(1).mean()),
            'near5_contains_observed_best_count':int(np.sum(band[np.arange(len(y)),obs.argmin(1)])),
            'near5_contains_observed_within5_count':int(np.sum(np.any(band&good,axis=1))),
            'near5_all_members_observed_within5_count':int(np.sum(np.all(~band|good,axis=1))),
            'near5_member_precision':float(np.sum(band&good)/np.sum(band)),
            'top3_contains_observed_within5_count':int(np.sum(np.any(np.take_along_axis(good,top3,axis=1),axis=1))),
            'family_choice_metrics':metric(obs,pred.argmin(1),families[cols])}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',type=Path,required=True); ap.add_argument('--confirmation',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    manifest=ROOT/'configs/generated_power_grid_v001/sampled_shapes.json'
    configpath=ROOT/'configs/generated_power_grid_v001/campaign_power_grid_v001.json'
    ids,held,shape,dims,candidates,y,config=load(args.screen,manifest,configpath)
    n,c=y.shape;allidx=np.arange(n);families=np.array([c['family'] for c in candidates]);cids=[c['candidate_id'] for c in candidates]
    xs={s['name']:np.array([[features(*d,c,s['feature_kind'],s['volume_knot']) for c in candidates] for d in dims]) for s in SPECS}
    folds=folds_for(allidx,5,SEED);outer=[];oof=np.zeros_like(y);fixed_native=np.zeros(n,int)
    nativecols=np.flatnonzero(families=='native');default=cids.index('native_default')
    provenance={'screen_sha256':sha(args.screen),'confirmation_sha256':sha(args.confirmation),
                'shape_manifest_sha256':sha(manifest),'campaign_sha256':sha(configpath),
                'screen_source':str(args.screen),'confirmation_source':str(args.confirmation),
                'reserved_shapes_used':0,'numpy_version':np.__version__}
    for f in range(5):
        train=np.array([i for i in allidx if folds[int(i)]!=f]);test=np.array([i for i in allidx if folds[int(i)]==f])
        spec,inner=choose_spec(xs,y,train,3,SEED+100+f)
        coef,diag=fit(xs[spec['name']],y,train);oof[test]=predict(xs[spec['name']],coef,test)
        best=nativecols[np.argmin(np.log(y[train][:,nativecols]).mean(0))];fixed_native[test]=best
        outer.append({'fold':f,'training_ids':[ids[i] for i in train],'test_ids':[ids[i] for i in test],
                      'inner':inner,'spec':spec,'coefficients':{cid:b.tolist() for cid,b in zip(cids,coef)},
                      'solver_diagnostics':diag,'training_selected_fixed_native':cids[best]})
        print('outer',f,spec['name'],flush=True)
    spec,final_search=choose_spec(xs,y,allidx,5,SEED)
    coef,diag=fit(xs[spec['name']],y,allidx);full=predict(xs[spec['name']],coef,allidx)
    model=make_model(spec,coef,ids,dims,candidates,config,provenance)
    selections=[];predictions=[];fullpred=[];defaultchoice=np.full(n,default)
    choices=oof.argmin(1);oracle=y.argmin(1)
    for i,sid in enumerate(ids):
        j=int(choices[i]);best=int(oracle[i]);famcols=np.flatnonzero(families==families[j]);fbest=float(y[i,famcols].min())
        row={'shape_id':sid,**dict(zip(('m','n','k'),dims[i])),'design_family':shape[sid]['family'],
             'on_lattice':shape[sid]['on_lattice'],'fold':folds[i],
             'selected_candidate':cids[j],'selected_family':str(families[j]),'selected_tile':candidates[j]['tile'],
             'predicted_selected_ms':float(oof[i,j]),'observed_selected_ms':float(y[i,j]),'observed_best_ms':float(y[i,best]),
             'regret':float(y[i,j]/y[i,best]),'oracle_candidate':cids[best],'oracle_family':str(families[best]),
             'family_regret':fbest/float(y[i,best]),'within_family_regret':float(y[i,j])/fbest,
             'native_default_regret':float(y[i,default]/y[i,best]),
             'training_fixed_native_regret':float(y[i,fixed_native[i]]/y[i,best])}
        selections.append(row)
        for jj,cc in enumerate(candidates):
            base={'shape_id':sid,**dict(zip(('m','n','k'),dims[i])),
                  'candidate_id':cc['candidate_id'],'family':cc['family'],'observed_ms':float(y[i,jj])}
            predictions.append({**base,'fold':folds[i],'predicted_ms':float(oof[i,jj])})
            fullpred.append({**base,'predicted_ms':float(full[i,jj])})
    metrics={'nested_selector':metric(y,choices,families),'native_default':metric(y,defaultchoice,families),
             'training_fixed_native':metric(y,fixed_native,families),'full_fit_in_sample':metric(y,full.argmin(1),families)}
    groupmetrics={}
    for fam in sorted({shape[s]['family'] for s in ids}):
        ix=np.array([i for i,s in enumerate(ids) if shape[s]['family']==fam]); groupmetrics[fam]=metric(y[ix],choices[ix],families)
    # Fixed final specification only: stronger leave-geometry-cluster-out stress diagnostic.
    # Knot/spec was chosen using all exploratory shapes, so this is explicitly post-selection sensitivity.
    def cluster(d):
        nearby={1023:1024,1025:1024,2047:2048,2049:2048,4095:4096,4097:4096,8191:8192,8193:8192}
        return tuple(sorted(nearby.get(v,v) for v in d))
    clusters=defaultdict(list)
    for i,d in enumerate(dims): clusters[cluster(d)].append(i)
    stress=np.zeros_like(y)
    for key,values in clusters.items():
        te=np.array(values);tr=np.array([i for i in allidx if i not in values]);b,_=fit(xs[spec['name']],y,tr)
        stress[te]=predict(xs[spec['name']],b,te)
    validation={'interpretation':'Retrospective development CV; outer labels excluded from inner tuning/fitting. Model-family design informed by prior plots. Screen minima are noisy seven-round selection oracles. Reserved cohort untouched.',
        'metrics':metrics,'selections':selections,'outer_folds':outer,'final_search':final_search,
        'final_solver_diagnostics':diag,'by_design_family':groupmetrics,'near_set_diagnostics':near_diagnostics(y,oof,families),
        'geometry_cluster_stress':{'interpretation':'Post-selection sensitivity with fixed final feature specification; not nested independent validation.',
            'cluster_count':len(clusters),'clusters':[{ 'canonical_sorted_dimensions':list(k),'shape_ids':[ids[i] for i in v]} for k,v in clusters.items()],
            'metrics':metric(y,stress.argmin(1),families)},'provenance':provenance}
    conf=confirmation_check(args.confirmation,selections,held)
    write(out/'model.json',model);write(out/'validation.json',validation)
    write(out/'predictions.json',predictions);write(out/'full_fit_predictions.json',fullpred)
    write(out/'confirmation_coverage.json',conf)
    probes=[(1025,1025,1025),(4096,4096,4096),(8192,8192,8192),(16384,16384,16384),
            (131072,2048,2048),(2048,131072,2048),(2048,2048,131072)]
    write(out/'example_predictions.json',[select(model,*d) for d in probes])
    columns=['intercept','volume_low_Telements','volume_high_Telements','padded_output_100Melements','padded_A_100Melements','padded_B_100Melements','copy_proxy_100Mbytes'] if spec['feature_kind']=='piecewise7' else ['intercept','padded_volume_Telements','padded_output_100Melements','padded_A_100Melements','padded_B_100Melements','copy_proxy_100Mbytes']
    text=['# Frozen coefficient table\n','All coefficients produce milliseconds; inputs use the indicated feature scaling. Nonnegative effective prediction costs, not independently identified hardware rates.\n',f"Specification: `{spec['name']}`. Volume knot: `{spec['volume_knot']}` elements.\n",'| Candidate | '+' | '.join(columns)+' |','|'+'---|'*(len(columns)+1)]
    for cid in cids: text.append('| '+cid+' | '+' | '.join(f'{b:.10g}' for b in model['coefficients'][cid])+' |')
    (out/'coefficients.md').write_text('\n'.join(text)+'\n')
    table=['# Out-of-fold selector decisions\n','Screen complete-call means; each row predicted without its geometry in training. R is selected latency divided by the best of 52 seven-round screen means, not a confirmed optimum.\n',
           '| M | N | K | Formula choice | Tile BM,BN,BK | Predicted ms | Measured ms | Best screen ms | Regret |','|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in sorted(selections,key=lambda r:(r['m']*r['n']*r['k'],r['m'],r['n'],r['k'])):
        table.append(f"| {r['m']} | {r['n']} | {r['k']} | {r['selected_candidate']} | {r['selected_tile'] or 'compiler-managed'} | {r['predicted_selected_ms']:.6f} | {r['observed_selected_ms']:.6f} | {r['observed_best_ms']:.6f} | {r['regret']:.4f} |")
    (out/'decisions.md').write_text('\n'.join(table)+'\n')
    write(out/'artifact_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'chosen_spec':spec,'metrics':metrics,'confirmation_covered':conf['covered_count']},indent=2),flush=True)

if __name__=='__main__':main()
