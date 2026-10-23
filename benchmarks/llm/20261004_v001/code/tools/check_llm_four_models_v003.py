"""Validate a model-scope migration, real scheduler barrier and Qwen bundle."""
import copy, io, json, os, shutil, sys, tarfile, tempfile
from pathlib import Path
from unittest.mock import patch
from llm_queue_state_v002 import read, save, ledger
from llm_model_scope_v001 import revise, planned_entries, pending_job, qualification_block, ORDER
from llm_scope_v001 import active_ids, measured_count
from llm_bundle_v002 import validate_archive
import run_llm_queue_v008 as scheduler


def main():
    root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
    campaign = root / 'results/v6e/llm_campaign_20261003_v001'
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir(exist_ok=True)
    protocol = read(Path('configs/llm_v6e_v002.json'))
    metadata = read(Path('results/v6e/llm_campaign_20261003_v001/inputs_gemma4_v001/models.json'))[0]
    old = read(campaign / 'queue_v001/state.json')
    state = revise(old, protocol, metadata)
    checks = []
    def check(condition, name):
        assert condition, name
        checks.append(name)
    check(state['active_session'] == old['active_session'], 'active controller and remote session unchanged')
    check(len(state['jobs']) == 40 and len(active_ids(state)) == 200, '40 workloads / 200 active identities')
    check(len(state['deferred_jobs']) == 40, '40 excluded workload records retained')
    check(state['model_order'] == ORDER, 'requested model order')
    check(revise(state, protocol, metadata) == state, 'migration is idempotent')
    old_jobs = {j['job_id']: j for j in old['jobs']}
    check(all(j == old_jobs[j['job_id']] for j in state['jobs'] if j['model'].startswith('qwen')), 'existing Qwen identities and attempts unchanged')
    rows = ledger(campaign / 'completed_entries.jsonl')
    check(measured_count(state, rows) == measured_count(old, rows), 'saved active groups preserved')
    check(not any(r['output_dtype'] == 'float32' for k,r in rows.items() if k in active_ids(state)), 'historical FP32 excluded')
    previous = [json.loads(line) for line in (campaign / 'planned_grid_v001/planned_entries.jsonl').read_text().splitlines()]
    planned = planned_entries(state, previous, protocol)
    check(len(planned) == len({r['entry_id'] for r in planned}) == 200, 'unique complete active plan')
    gemma = [j for j in state['jobs'] if j['model'] == 'gemma4_31b']
    check(len(gemma) == 10 and all(qualification_block(j) for j in gemma), 'all Gemma4 jobs gated')
    check(all(j['revision'] == metadata['revision'] for j in gemma), 'Gemma4 checkpoint revision pinned')
    check(all(r['projection_shapes_mkn'] is None for r in planned if r['model'] == 'gemma4_31b'), 'unsupported geometry is not fabricated')
    with tempfile.TemporaryDirectory() as temporary:
        test_campaign = Path(temporary) / 'campaign'; queue = test_campaign / 'queue_v001';queue.mkdir(parents=True)
        gated = copy.deepcopy(state);gated.pop('active_session');gated['status']='queued'
        for job in gated['jobs']:
            if job['model'] in ORDER[:2]:job['status']='completed'
        complete_ids = [ident for job in gated['jobs'] if job['model'] in ORDER[:2] for ident in job['groups']['bfloat16']]
        (test_campaign/'completed_entries.jsonl').write_text(''.join(json.dumps(dict(entry_id=i))+'\n' for i in complete_ids))
        save(queue/'state.json', gated)
        check(pending_job(gated)['model'] == 'gemma4_31b', 'qualification barrier precedes Qwen14')
        calls=[]
        def command(args, **kwargs):
            calls.append(args)
            assert 'record_progress_v001.py' in str(args), 'Unexpected process or TPU operation at qualification barrier'
        with patch.dict(os.environ, STRASSEN_PROJECT_ROOT=str(Path(temporary).resolve())), patch.object(sys, 'argv', ['queue','--campaign',str(test_campaign),'--controller-python',sys.executable]), patch.object(scheduler.subprocess,'run',command):
            scheduler.main()
        stopped=read(queue/'state.json')
        check(stopped['status']=='blocked_model_qualification' and stopped['blocking_model']=='gemma4_31b', 'actual supervisor stops before allocation or skipping Gemma4')
        check(len(calls)==1, 'barrier issues only a progress journal update')
    # Validate the same packaged Qwen launch contract used after the migration.
    cases=[dict(j,output_dtypes=['bfloat16']) for j in state['jobs'] if j['model']=='qwen3_32b']
    job=dict(cases[0],session_jobs=cases)
    with tempfile.TemporaryDirectory() as temporary:
        package=Path(temporary)/'source.tar'
        with tarfile.open(package,'w') as t:
            for directory in ('src','tools','runtime','configs'):t.add(directory,arcname=directory)
            raw=json.dumps(job).encode();item=tarfile.TarInfo('job.json');item.size=len(raw);t.addfile(item,io.BytesIO(raw))
        validation=validate_archive(package,job)
        check(validation['status']=='passed','actual Qwen32 source archive and CLI contracts pass')
        save(out/'bundle-validation.json',validation)
    # Isolated proposed campaign for browser checks, without editing the live queue.
    fixture=out/'fixture/campaign';fixture.mkdir(parents=True)
    for relative in ('inputs_v002/models.json','planned_grid_v001/planned_entries.jsonl','completed_entries.jsonl','active-run.json'):
        target=fixture/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(campaign/relative,target)
    target=fixture/'inputs_gemma4_v001/models.json';target.parent.mkdir();save(target,[metadata])
    target=fixture/state['active_plan'];target.parent.mkdir();target.write_text(''.join(json.dumps(r)+'\n' for r in planned))
    (fixture/'queue_v001').mkdir();save(fixture/'queue_v001/state.json',state)
    run=state['active_session']['run'];(fixture/run).mkdir()
    for name in ('latest.json','job.json','frozen.json'):
        if (campaign/run/name).exists():shutil.copyfile(campaign/run/name,fixture/run/name)
    (fixture/'pilot_20261003_v002').mkdir()
    summary=dict(status='completed',checks=checks,fixture=str(fixture),measured=measured_count(state,rows))
    save(out/'scope-summary.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
