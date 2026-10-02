"""Materialize and freeze a separate replay; never alter the active campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import prepare_llm_tradeoff_v002 as campaign


def materialize():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    a=p.parse_args(sys.argv[2:]);broad=json.loads(a.report.read_text());pilot=json.loads(a.pilot.read_text())
    assert broad['completed'] and pilot['completed']
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(exist_ok=False)
    cfg=json.loads((campaign.ROOT/'configs/fullk_v002/campaign.json').read_text())
    cfg.update(campaign_id='fullk_replay_v001',confirm_seeds=[9252000,9253000,9254000],
        study='Preregistered replay of known pilot full-K tiles on two geometries and both outputs; no selection from replay',
        candidate_policy='Frozen broad-study Native/default/full/short choices plus earlier pilot winners and omitted pilot full-K geometries, each with one/two buffers and separate output dtypes. Retain the original FP32 kernel at each prior winning tile with its original two buffers.',
        selection_policy='No selection from replay results. Report every fixed arm against within-allocation reruns of the broad-study frozen choices.')
    cfg['memory']['execution_order']='Two serial geometry batches, each with separate FP32 and BF16 groups on one new allocation.'
    cfg['timing']['measure']=dict(warmups=5,repeats=30)
    shapes=[]
    for index,base_id in enumerate(('k8192','parent')):
        pairs=[r for r in broad['results'] if r['shape']['base_shape_id']==base_id]
        assert len(pairs)==2
        s=pairs[0]['shape'];dims=tuple(s[d] for d in ('m','k','n'))
        old=next(r for r in pilot['results'] if tuple(r['shape'][d] for d in ('m','k','n'))==dims and r['shape']['output_dtype']=='float32')
        selected_tiles={}
        for depth in (1,2):
            pool=[m for m in old['methods'].values() if m['eligible'] and m['candidate']['implementation']=='current'
                  and m['candidate']['depth']==depth and m['candidate']['tile'][2]==s['k']]
            selected_tiles[depth]=min(pool,key=lambda m:m['mean_ms'])['candidate']['tile']
        missing_tiles=[[bm,bn,s['k']] for bm,bn in old['shape']['output_tiles'] if [bm,bn] not in s['output_tiles']]
        replay_tiles={depth:list(dict.fromkeys(tuple(t) for t in [selected_tiles[depth],*missing_tiles])) for depth in (1,2)}
        item=dict(id=base_id,m=s['m'],k=s['k'],n=s['n'],seed_offset=index,arms_by_dtype={},pilot_winning_tiles=selected_tiles,
                  replay_tiles=replay_tiles)
        for r in pairs:
            dtype=r['shape']['output_dtype'];arms=[]
            for role in ('native','native_default','short_s1','full_s1','short_s2','full_s2'):
                m=r['methods'].get(role)
                if not m:continue
                arm=dict(m['candidate']);arm.update(arm_id='Broad_'+role,candidate_id='Broad_'+role,role=role,headline_roles=[role])
                arms.append(arm)
            for depth in (1,2):
                for tile in replay_tiles[depth]:
                    for buffers in (1,2):
                        name=f'Pilot_S{depth}_{tile[0]}_{tile[1]}_b{buffers}'
                        arms.append(dict(arm_id=name,candidate_id=name,algorithm='current',family='current',implementation='current',
                            variant=name,depth=depth,tile=list(tile),buffers=buffers,output_dtype=dtype,
                            compiler_options={},vmem_limit_bytes=112*1024**2,role=f'pilot_s{depth}',headline_roles=[name]))
                if dtype=='float32':
                    tile=selected_tiles[depth];name=f'Original_S{depth}_{tile[0]}_{tile[1]}_b2'
                    arms.append(dict(arm_id=name,candidate_id=name,algorithm='incumbent',family='incumbent',implementation='incumbent',
                        variant=name,depth=depth,tile=list(tile),buffers=2,output_dtype=dtype,
                        compiler_options={},vmem_limit_bytes=112*1024**2,role=f'original_s{depth}',headline_roles=[name]))
            item['arms_by_dtype'][dtype]=arms
        shapes.append(item)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.report,a.pilot)}
    (out/'campaign.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'shapes.json').write_text(json.dumps(dict(shapes=shapes,input_sha256=provenance),indent=2)+'\n')
    (out/'distributions.json').write_text((campaign.ROOT/'configs/fullk_v002/distributions.json').read_text())
    print(json.dumps(dict(completed=True,geometries=len(shapes),cases=3*sum(len(arms) for s in shapes for arms in s['arms_by_dtype'].values()))))


def build(args):
    if args.hardware!='v6e':raise ValueError('v6e required')
    cfgpath='configs/fullk_replay_v001/campaign.json'
    shapes=json.loads((campaign.ROOT/'configs/fullk_replay_v001/shapes.json').read_text())['shapes']
    stages=[dict(id='replay-smoke',label='Requalify both output precisions on this runtime',operation='phase',kind='measurement',
        campaign=cfgpath,module='strassen_mm.benchmark_fullk_v002',timeout_seconds=3600,expected=28)]
    for i,s in enumerate(shapes):
        stages.append(dict(id=f'replay-{i+1:02d}-measure',label=s['id']+': known pilot tiles and frozen broad choices',
            operation='phase',kind='measurement',campaign=cfgpath,module='strassen_mm.benchmark_fullk_replay_v001',
            timeout_seconds=10800,expected=3*sum(len(a) for a in s['arms_by_dtype'].values()),requires=[stages[-1]['id']],
            runner_args=['--shape-start',str(i),'--shape-count','1']))
    stages.extend([
        dict(id='report',label='Known pilot tiles versus frozen broad-study choices',operation='command',kind='analysis',timeout_seconds=900,
             continue_on_failure=True,command=['{analysis_python}','{source}/tools/report_fullk_replay_v001.py','--cohort','{cohort}','--output-dir','{cohort}/operations/report/artifacts']),
        dict(id='release',label='Release replay runtime after retrieval',operation='command',kind='preparation',timeout_seconds=180,releases_allocation=True,
             command=['{controller_python}','{source}/runtime/release_allocation_v002.py','--session','{session}','--expect-endpoint','{endpoint}'])])
    return dict(schema_version=1,cohort_id=args.cohort.name,session=args.session,allocation_id=args.endpoint,
        remote_identity=args.identity,controller_python=args.controller_python,analysis_python=args.analysis_python,
        hardware='v6e',release='runtime/release_allocation_v002.py',transport='tools/run_phase_v005.py',
        title='Full-K replay: retain prior pilot winners in both precisions',stages=stages,
        decisions=['Separate follow-up on one new allocation; rerun all comparators together, never pool cross-allocation samples.',
            'Replay earlier pilot FP32-winning full-K tiles and omitted prior full-K output geometries on the 8192 cube and parent shape, with both output dtypes and buffer counts. Include the previously infeasible larger cube tile.',
            'Broad-study choices stay frozen. Three new seeds × 30 paired rounds. Every preregistered arm reported; no replay-driven reselection.',
            'At each earlier winning FP32 tile, retain the unchanged original kernel with two buffers to distinguish tile selection from an implementation effect.',
            'Same BF16 inputs/preadds, FP32 accumulation, final output contract and 112 MiB custom allowance; no padding or depths3/4.'])


campaign.build_plan=build
if __name__=='__main__':
    if sys.argv[1]=='materialize':materialize()
    else:campaign.main()
