#!/usr/bin/env python3
"""Freeze N8 geometry/controls or N9 model input and policy bindings.

No network, TPU or model execution. Root runs through its archive manager.
N9 bindings list local evidence manifests and their exact remote paths; the
output contains copies/hashes of the evidence and data-independent N8 policy
selection. Checkpoint binaries remain external, immutable reproducible inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''): h.update(data)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def n8(args,output):
    groups=[]
    for model,hidden,intermediate in (('qwen06',1024,3072),('mistral7b',4096,14336),('qwen32b',5120,25600)):
        for m in ((1024,2048) if model=='qwen06' else (512,1024,2048)):
            for kind,shape in (('swiglu',[m,hidden,intermediate*2]),('residual',[m,intermediate,hidden])):
                groups.append({'id':f'{model}_m{m}_{kind}','geometry':model,'kind':kind,'shape_mkn':shape})
    campaign={'schema_version':1,'campaign_id':'v5e_n8_applications_v1','base_campaign':args.base_campaign,
       'seed':20260919,'matched_tile':[1024,1024,512],'groups':groups,
       'application_timing':{'warmups':3,'repeats':30,'preparation_repeats':5},
       'sample_rows':32,'sample_columns':128,
       'correctness':{'gate':{'require_finite':True,'relative_l2_max':0.02,'max_abs_atol':0.001,'max_abs_reference_rtol':0.05}},
       'precision_contract':'BF16 inputs/preadds; FP32 accumulation; every projection rounds toBF16 before epilogue; SiLU FP32 roundsBF16 before BF16multiply; residual BF16+BF16→BF16',
       'input_scope':'Synthetic Gaussian BF16 inputs at LLM-like geometries, not model activations or model quality',
       'coverage_refinement':'D009: retain12main M1024/2048groups; add M512SwiGLU/residual for Mistral7B andQwen32B (4groups) toinclude exact N5geometry selectedcontrols whereavailable. Frozen beforeN8outcomes.',
       'ablation':'Unpacked-unfused vs packed-unfused isolates packing; packed-unfused vs standard-fused isolates epilogue fusion withmatchedlayout/schedule; standard-fused vs early changesfinalization anddocumentedFP32order',
       'predeclared_contrasts':['packed-unfused/unpacked-unfused peralgorithm','fused/packed-unfused peralgorithm (residual uses unpacked control)',
           'early/standard-fused Strassen','each Strassen/native','each Strassen/matched cubic_full and cubic_quadrant','N5 selectedStrassen/selectedcubic'],
       'selection':'Lowest eligible complete-call mean per family/geometry. Used only tofreeze N9policy before real quality measurements.'}
    if args.n5_selections:
        value=json.loads(args.n5_selections.read_text())
        if value.get('phase')!='N5-screen': raise ValueError('Expected exact N5 screen selection file')
        campaign['n5_selections']=args.n5_remote_selections
        campaign['n5_selections_sha256']=digest(args.n5_selections)
        shutil.copyfile(args.n5_selections,output/'n5_selections.json')
    save(output/'campaign_applications_n8_v001.json',campaign)
    return campaign


def policy_choice(row):
    return {key:row[key] for key in ('algorithm','variant','tile','packed','fused','early') if key in row}


def n9(args,output):
    source=json.loads(args.n8_selections.read_text())
    if source.get('phase')!='N8': raise ValueError('N9 policies require complete N8 selection evidence')
    selection_sha=digest(args.n8_selections);shutil.copyfile(args.n8_selections,output/'n8_selections.json')
    bindings=json.loads(args.bindings.read_text());models=[]
    geometries={'Qwen/Qwen3-0.6B':'qwen06','mistralai/Mistral-7B-v0.3':'mistral7b'}
    allowed={'Qwen/Qwen3-0.6B':'c1899de289a04d12100db370d81485cdf75e47ca',
             'mistralai/Mistral-7B-v0.3':'caa1feb0e54d415e2df31207e5f4e273e33509b1'}
    for index,binding in enumerate(bindings['models']):
        if binding.get('status')=='blocked_access':
            if binding['model_id']!='google/gemma-3-1b-pt': raise ValueError('Unexpected blocked model')
            row={key:binding[key] for key in ('model_id','status','reason') if key in binding}
            if binding.get('access_evidence'):
                p=Path(binding['access_evidence']);shutil.copyfile(p,output/f'model-{index:02d}-access.json')
                row['access_evidence_sha256']=digest(p)
            models.append(row);continue
        model=binding['model_id']
        if model not in allowed: raise ValueError('Only preregistered public Qwen/Mistral models supported')
        row={'model_id':model,'revision':allowed[model],'input_sha256':{},'policies':{},'unsupported_families':[]}
        manifests={}
        for name in ('model_manifest','tokens_manifest','reference_manifest'):
            spec=binding[name]
            if isinstance(spec,str): spec={'remote_path':spec,'local_path':binding['local_'+name]}
            path=Path(spec['local_path'])
            remote=Path(spec['remote_path'])
            if not remote.is_absolute() or not remote.is_relative_to('/content/Strassen_MM_Focus'):
                raise ValueError('Remote input must be an absolute path under the authorized project cache/runs')
            row[name]=str(remote);row['input_sha256'][name]=digest(path)
            manifests[name]=json.loads(path.read_text());shutil.copyfile(path,output/f'model-{index:02d}-{name}.json')
        mm=manifests['model_manifest'];tm=manifests['tokens_manifest'];rm=manifests['reference_manifest']
        if mm['model_id']!=model or mm['revision']!=allowed[model]: raise ValueError('Unexpected exact checkpoint')
        if tm['model_manifest_sha256']!=row['input_sha256']['model_manifest'] or rm['model_manifest_sha256']!=row['input_sha256']['model_manifest']:
            raise ValueError('Token/reference manifests identify different checkpoint evidence')
        if rm['tokens_sha256']!=tm['token_array_sha256']: raise ValueError('Reference and quality token sources differ')
        for family in ('cubic','strassen'):
            policy={};count=0;provenance={}
            for name,kind in (('gateup','swiglu'),('down','residual')):
                group=f'{geometries[model]}_m1024_{kind}';selected=source['by_group'][group].get(family)
                if selected is None:
                    policy[name]={'algorithm':'native','variant':'plain','tile':None,'fused':False}
                    provenance[name]={'group':group,'status':'no_numerically_eligible_family_candidate_native_fallback'}
                else:
                    policy[name]=policy_choice(selected);count+=1
                    provenance[name]={'group':group,'arm_id':selected['arm_id'],'mean_ms':selected['mean_ms']}
            if count:
                policy['selection_provenance']={'source_sha256':selection_sha,'projections':provenance,
                                                'custom_projection_count':count,'quality_data_used':False}
                row['policies'][family+'_n8_selected']=policy
            else: row['unsupported_families'].append(family)
        if not row['policies']: raise ValueError('No custom application policy qualified for any projection')
        models.append(row)
    if set(geometries)-{m['model_id'] for m in models}: raise ValueError('Both Qwen and Mistral bindings are required')
    campaign={'schema_version':1,'campaign_id':'v5e_n9_real_models_v1','base_campaign':args.base_campaign,'seed':20260919,
        'n8_selection_sha256':selection_sha,'selection_environment_identity':source['environment_identity'],
        'qualification_thresholds':{'max_relative_l2':0.03,'max_mean_kl':0.01,'max_abs_nll_delta':0.05,'min_top1_agreement':0.90},
        'quality_thresholds':{'max_abs_nll_delta':0.01,'max_mean_kl':0.02,'min_top1_agreement':0.97},
        'resident_timing':{'warmups':2,'repeats':7},'streamed_repeats':3,'models':models,
        'quality_scope':'First32768 contiguous WikiText2 raw test tokens per official tokenizer;32x1024 windows;32736scoredpositions;nativeCPUqualification first64tokens; no prompt selection by quality outcomes',
        'model_scope':'Qwen3-0.6B and Mistral7B full teacher-forced streamed prefill; Gemma3-1B permission blocker retained; no generation, task accuracy, decode or production serving claim',
        'policy_scope':'Attention native in allarms. Replace MLPgate/up+down using frozen N8complete-call selections and explicit nativefallbacks; independent real-model quality gates still required',
        'timing_scope':'Allarms JIT full-layer graphs and same streamed hostloop. Residentlayer comparison same native incoming state; fullmodelstreamed wall includes reads/layout/H2D/layers/head, excludes downloads/tokenization/compilation. Three repeats are descriptive.',
        'threshold_origin':'Frozen before measurements and approved by root; quality failure cannot be repaired by silently changing thresholds or testtexts'}
    save(output/'campaign_applications_n9_v001.json',campaign)
    return campaign


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    for phase in ('n8','n9'):
        q=sub.add_parser(phase);q.add_argument('--output-dir',type=Path,required=True)
        q.add_argument('--base-campaign',required=True,help='Path from generated campaign location to frozen base campaign')
        if phase=='n8':
            q.add_argument('--n5-selections',type=Path);q.add_argument('--n5-remote-selections')
        else:
            q.add_argument('--n8-selections',type=Path,required=True);q.add_argument('--bindings',type=Path,required=True)
    args=p.parse_args(argv)
    if args.phase=='n8' and bool(args.n5_selections)!=bool(args.n5_remote_selections): p.error('Both N5 paths required together')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result=(n8 if args.phase=='n8' else n9)(args,args.output_dir)
    save(args.output_dir/'artifact_manifest.json',{'sha256':{f.name:digest(f) for f in args.output_dir.iterdir() if f.is_file()}})
    print(json.dumps({'phase':args.phase,'output_dir':str(args.output_dir),'campaign_id':result['campaign_id']}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
