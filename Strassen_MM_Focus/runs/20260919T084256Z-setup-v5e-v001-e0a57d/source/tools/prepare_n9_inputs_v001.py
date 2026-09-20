"""Sequentially acquire and qualify immutable application inputs.

Each step delegates to the versioned execution manager, which snapshots source,
archives its result and commits before the next step. No TPU timing may overlap
this workflow. Failed steps are preserved; dependent steps are not fabricated.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODELS=[('qwen','Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca'),
        ('mistral','mistralai/Mistral-7B-v0.3','caa1feb0e54d415e2df31207e5f4e273e33509b1')]
BASE='/content/Strassen_MM_Focus'

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser()
    for key in ('session','endpoint','expected-identity','controller-python','output-dir'):p.add_argument('--'+key,required=True)
    args=p.parse_args()
    if not (ROOT/'.git').exists():
        raise RuntimeError('Run this workflow from the canonical project tools path, not an archive snapshot; each child archives itself')
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'models':MODELS,
                         'arguments':vars(args),'policy':'Sequential immutable child executions; no concurrent TPU timings'})
    records=[]
    def run(label,action,extra=(),budget=3600):
        phase='N9-'+label
        command=[sys.executable,str(ROOT/'tools/run_phase_v004.py'),'--phase',phase,
                 '--session',args.session,'--endpoint',args.endpoint,'--expected-identity',args.expected_identity,
                 '--controller-python',args.controller_python,'--benchmark-module','strassen_mm.application_prep_v001',
                 '--timeout-seconds',str(budget)]
        command+=['--runner-arg='+v for v in ('--action',action,*extra)]
        log=out/(label+'.controller.log')
        save(out/(label+'.command.json'),{'argv':command})
        print('Starting '+phase,flush=True)
        with log.open('x') as f:child=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        paths=[line.split('=',1)[1] for line in log.read_text().splitlines() if line.startswith('EXECUTION_DIR=')]
        if len(paths)!=1:raise RuntimeError('No unique child execution path: '+str(log))
        directory=Path(paths[0]).resolve()
        if not directory.is_relative_to(ROOT/'runs'):raise RuntimeError('Child execution escaped the canonical run directory')
        completion=json.loads((directory/'completion.json').read_text())
        record={'phase':phase,'run_id':directory.name,'run_path':str(directory),'status':completion['status'],
                'exit_code':child.returncode,'completion_sha256':hashlib.sha256((directory/'completion.json').read_bytes()).hexdigest()}
        preparation_status=directory/'artifacts/preparation-00/status.json'
        record['preparation_status']=json.loads(preparation_status.read_text()).get('status') if preparation_status.is_file() else 'missing'
        records.append(record)
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        print(json.dumps(record),flush=True)
        if child.returncode or record['status']!='completed' or record['preparation_status']!='completed':return None
        return {'local':directory/'artifacts/preparation-00',
                'remote':BASE+'/runs/'+directory.name+'/artifacts/preparation-00','run_id':directory.name}
    models=[]
    for slug,model,revision in MODELS:
        item={'model_id':model,'revision':revision,'status':'input_preparation_pending'}
        result=run('download-'+slug,'download',('--model-id',model,'--revision',revision),7200)
        if result:
            manifest=json.loads((result['local']/'model_manifest.json').read_text())
            item.update(status='checkpoint_ready',model_manifest=manifest['cache_dir']+'/manifest.json',
                        local_model_manifest=str(result['local']/'model_manifest.json'),download_run_id=result['run_id'])
        else:item['status']='download_failed'
        models.append(item)
    corpus=run('corpus','corpus',budget=1800)
    model_tools=run('model-tools','model-tools',budget=7200)
    if corpus and model_tools:
        corpus_manifest=json.loads((corpus['local']/'corpus_manifest.json').read_text())
        for (slug,_,_),item in zip(MODELS,models):
            if item['status']!='checkpoint_ready':continue
            tokens=run('tokenize-'+slug,'tokenize',('--model-manifest',item['model_manifest'],
                '--corpus-manifest',corpus_manifest['cache_dir']+'/manifest.json'),3600)
            if not tokens:item['status']='tokenization_failed';continue
            item.update(tokens_manifest=tokens['remote']+'/tokens_manifest.json',
                        local_tokens_manifest=str(tokens['local']/'tokens_manifest.json'),tokenize_run_id=tokens['run_id'])
            reference=run('reference-'+slug,'reference',('--model-manifest',item['model_manifest'],
                '--tokens',tokens['remote']+'/token_ids.npy'),7200)
            if not reference:item['status']='reference_failed';continue
            item.update(status='inputs_ready',reference_manifest=reference['remote']+'/reference_manifest.json',
                        local_reference_manifest=str(reference['local']/'reference_manifest.json'),reference_run_id=reference['run_id'])
    models.append({'model_id':'google/gemma-3-1b-pt','status':'blocked_access',
      'reason':'Official access probe denied available credentials; no terms accepted or mirrors used',
      'access_probe_run_id':'20260919T055501Z-N9-access-probe-v5e-v004-ebb556'})
    bundle={'created_utc':datetime.now(timezone.utc).isoformat(),'cohort':args.endpoint,'models':models,'steps':records,
            'scope':'Reproducible external input preparation only; no TPU model-quality result yet'}
    save(out/'input_bundle.json',bundle)
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'tools');import archive_v002 as a;print(a.commit('Archive N9 application input workflow'))"],cwd=ROOT,check=True)
    print('INPUT_BUNDLE='+str(out/'input_bundle.json'),flush=True)
    return 0 if all(x['status'] in ('inputs_ready','blocked_access') for x in models) else 1

if __name__=='__main__':raise SystemExit(main())
