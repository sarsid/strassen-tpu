"""Offline checks for serial campaign planning and exact progress counting."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import run_mlsys_unattended_v001 as controller
from prepare_mlsys_campaign_v001 import build_plan

root=Path(__file__).resolve().parents[1]
for path in ['tools/run_mlsys_unattended_v001.py','tools/prepare_mlsys_campaign_v001.py',
             'tools/stage_mlsys_auth_v001.py','tools/audit_mlsys_stage_v001.py',
             'runtime/prepare_mlsys_private_v001.py','src/strassen_mm/benchmark_mlsys_prepare_v001.py']:
    ast.parse((root/path).read_text())
plan=build_plan(Path('/tmp/fixture'), 'fixture-session','fixture-endpoint','/identity.json',sys.executable,sys.executable)
ids=[s['id'] for s in plan['stages']];assert len(ids)==len(set(ids))
assert ids.index('LLM-confirm')<ids.index('qwen-actual')<ids.index('MAIN-01-screen')
assert sum(s.get('expected',0) or 0 for s in plan['stages'] if s['id'].startswith('MAIN-'))==8736
assert sum(s.get('expected',0) or 0 for s in plan['stages'] if s['id'].startswith('LLM-'))==936
assert [s for s in plan['stages'] if s.get('releases_allocation')][0]['command'][-1]=='{endpoint}'
for i,s in enumerate(plan['stages']):
    for dependency in s.get('requires',[]):assert ids.index(dependency)<i
    if s.get('selection_stage'):assert ids.index(s['selection_stage'])<i
    if s['operation']=='command':controller.format_command(s,Path('/tmp/fixture'),plan)
row=dict(event='case_result',group_id='g',seed=1,arm_id='a',scope='call',status='ok',timing={'sample_count':30})
assert controller.counts([row,row,dict(row,scope='kernel'),dict(row,arm_id='b',status='compile_error',timing={})])==dict(completed=2,succeeded=1,failed=1,measured=1)
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'rows.jsonl';path.write_text(json.dumps(row)+'\n{"event":')
    rows,offset=controller.read_new_rows(path,0);assert rows==[row]
    with path.open('a') as f:f.write('"group_start"}\n')
    rows2,offset2=controller.read_new_rows(path,offset);assert rows2==[{'event':'group_start'}]
    assert controller.read_new_rows(path,offset2)==([],offset2)
subprocess.run([sys.executable,str(root/'tools/check_mlsys_shapes_v001.py')],check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_campaign_progress_v001.py','-v'],cwd=root,check=True)
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'launch-checks.json'
with out.open('x') as f:json.dump(dict(status='passed',stages=len(ids),main_screen_outcomes=8736,llm_screen_outcomes=936,
    checks=['syntax','dependency_order','source_plan_commands','exact_terminal_progress','partial_JSONL','shape_scientific_checks','dashboard_tests']),f,indent=2)
print(json.dumps({'status':'passed','stage_count':len(ids)}))
