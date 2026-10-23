"""Validate scoped counts, recovery and the real BF16-only session launch path."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'runtime')]
from llm_queue_state_v002 import read, save, plan, merge, ledger
from llm_scope_v001 import remaining, active_ids, measured_count
from run_llm_queue_v007 import reconcile, finish
from llm_bundle_v002 import validate_archive, WORKER
from stop_llm_precision_v001 import matches


def main():
    checks = []
    def check(value, label):
        assert value, label
        checks.append(label)
    protocol = read(ROOT / 'configs/llm_v6e_v002.json')
    jobs = plan(protocol)
    for job in jobs:job['active_output_dtypes'] = ['bfloat16']
    state = dict(jobs=jobs, expected_comparisons=350)
    check(len(active_ids(state)) == 350, '350 active BF16 identities; original groups preserved')
    j0, j1 = jobs[:2]
    def rows(job, dtype):
        return [dict(entry_id=ident, **{k:job[k] for k in ('model','revision','batch','sequence','architecture')},
                     algorithm=alg, output_dtype=dtype, status='completed', measurement={'fixture':True})
                for alg, ident in zip(protocol['algorithms'], job['groups'][dtype])]
    with tempfile.TemporaryDirectory() as folder:
        t = Path(folder);dest = t / 'ledger.jsonl'
        merge(dest, rows(j0, 'bfloat16') + rows(j0, 'float32') + [], j0, 'historical')
        merge(dest, rows(j1, 'bfloat16'), j1, 'legacy')
        saved = dest.read_bytes();data = ledger(dest)
        reconcile(state, data)
        check(state['measured_comparisons'] == 10 and state['historical_comparisons'] == 5, 'FP32 history cannot inflate BF16 progress')
        check(j0['status'] == j1['status'] == 'completed' and remaining(j1,data) == [], 'saved BF16 completes workload despite missing FP32')
        check(dest.read_bytes() == saved, 'reconciliation does not mutate historical results')
        j1['attempts'] = [dict(status='running', source_commit='legacy')]
        state.update(status='running',active_session=dict(run='legacy',legacy=True,job_ids=[j1['job_id']],source_commit='legacy',model=j1['model']))
        finish(state,dict(status='failed',release_exit_code=0,remote_summary={'stages':[dict(name='main-prefill',returncode=-15)]}),data)
        check(j1['status'] == 'completed' and state['status'] == 'queued', 'operator-stopped FP32 does not requeue completed BF16')
        pending = jobs[2];pending['attempts'] = [dict(status='running',source_commit='new')]
        state['active_session'] = dict(run='failed',job_ids=[pending['job_id']],source_commit='new',model=pending['model'])
        finish(state,dict(status='failed',release_exit_code=1),data)
        check(state['status'] == 'blocked' and state.get('active_session'), 'unverified release still blocks next allocation')
        finish(state,dict(status='failed',release_exit_code=0),data)
        check(remaining(pending,data) == ['bfloat16'], 'recovery requests only missing BF16 group')
        allrows = {ident: {} for job in jobs for group in job['groups'].values() for ident in group}
        check(measured_count(state,allrows) == 350, 'completion count ignores all 350 historical FP32 IDs')
        check(matches(['python','/r/source/tools/run_llm_case_v002.py','--output','/r/artifacts/main-prefill'],Path('/r/source/tools/run_llm_case_v002.py'),Path('/r/artifacts/main-prefill')), 'stop matches exact measurement and output')
        check(not matches(['python','/other/script','--output','/r/artifacts/main-prefill'],Path('/r/source/tools/run_llm_case_v002.py'),Path('/r/artifacts/main-prefill')), 'stop rejects unrelated process')
        check(not matches(['python','/r/source/tools/run_llm_case_v002.py','--output','/other'],Path('/r/source/tools/run_llm_case_v002.py'),Path('/r/artifacts/main-prefill')), 'stop rejects another workload')
        base = t / 'remote';run = base / 'runs/fixture';run.mkdir(parents=True);source=run/'source'
        for name in ('src','tools','runtime','configs'):shutil.copytree(ROOT/name,source/name)
        selected = [dict(j,output_dtypes=remaining(j,{})) for j in jobs if j['model']=='gemma3_12b'][:2]
        job = dict(selected[0],session_jobs=selected);save(source/'job.json',job);save(run/'launch.json',{'fixture':True})
        package=t/'source.tar'
        with tarfile.open(package,'w') as tar:
            for item in source.iterdir():tar.add(item,arcname=item.name)
        check(validate_archive(package,job)['status']=='passed','actual archived launch validates BF16 subset and pinned jobs')
        spec=importlib.util.spec_from_file_location('session',source/WORKER);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.BASE=base
        import parent_replication_remote_v001 as parent
        calls=[]
        def execute(command,log,env,timeout,cwd):calls.append(command);log.write_text('external process replaced\n');return 0
        original=Path.read_text
        def read_text(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return 'MemAvailable: 999999999 kB\nMemTotal: 999999999 kB\n'
            return original(path,*args,**kwargs)
        with patch.object(parent,'execute',side_effect=execute),patch.object(worker.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**15)),patch.object(Path,'read_text',read_text):worker.worker(run)
        mains=[c for c in calls if any('tools/run_llm_case_v003.py' in x for x in c)]
        check(len(mains)==2 and all(c[c.index('--output-dtypes')+1:] in (['bfloat16'],['bfloat16','--pilot']) for c in mains), 'real worker passes only BF16 to every workload')
        check('--pilot' in mains[0] and '--pilot' not in mains[1], 'first BF16 workload retains cache equivalence pilot')
        check(sum('-m' in c and 'venv' in c for c in calls)==1, 'one setup reused across model workloads')
        check((run/'archive-ready.json').exists() and len(list((run/'artifacts/cases').glob('*/case-ready.json')))==2, 'case and final archives still produced')
        frozen=read(ROOT/'configs/llm_v6e_bf16_v001.json')
        check(frozen['timing']==protocol['timing'] and frozen['gates']==protocol['gates'] and frozen['algorithms']==protocol['algorithms'], 'only output scope changes; timing and numerical gates intact')
    output=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=len(checks))))


if __name__=='__main__':main()
