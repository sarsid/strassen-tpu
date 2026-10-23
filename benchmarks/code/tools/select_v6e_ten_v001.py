"""Select ten historical v5e wins without observing any new v6e timings."""
import json, os, hashlib
from pathlib import Path
from collections import defaultdict
from scan_strassen2_interim_v001 import paired_speedup
root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
history=root/'runs/20260921T212710Z-strassen2-interim-scan-v001-6331e5/artifacts/shape_results.json'
rows=json.loads(history.read_text());cache={};eligible=[]
for row in rows:
 if not row['all_five_eligible']:continue
 if not all(row['comparisons'][x]['ci_win'] for x in ('native','native_default')):continue
 art=Path(row['evidence'])
 if art not in cache:
  stats=json.loads((art/'confirmation_statistics.json').read_text())['by_shape']
  groups={g['shape']['id']:g['group_id'] for g in json.loads((art/'planned_cases.json').read_text())}
  samples=defaultdict(lambda:defaultdict(dict))
  for line in (art/'results.jsonl').open():
   r=json.loads(line)
   if r['event']=='sample':samples[(r['group_id'],r['arm_id'])][r['seed']][r['round']]=r['elapsed_ms']
  cache[art]=(stats,groups,samples)
 stats,groups,samples=cache[art];h=stats[row['shape_id']]['headline'];gid=groups[row['shape_id']]
 intervals={}
 for baseline in ('native','native_default'):
  intervals[baseline]=paired_speedup(samples,gid,h[baseline]['candidate_id'],h['one_level']['candidate_id'])
 if all(c['ci95'][0]>1 for c in intervals.values()):
  eligible.append({**row,'one_level_comparisons':intervals})
selected=sorted(eligible,key=lambda x:(-x['comparisons']['native']['speedup'],x['shape_id']))[:10]
assert len(selected)==10
policy='From the preserved 102-shape v5e confirmation scan, require numerical eligibility and pointwise 95% paired-bootstrap speedup lower bound >1 for BOTH S1 and S2 versus BOTH Native default and tuned. Rank by historical S2 speedup versus tuned Native; take ten. No new v6e timings used. Winner-enriched selection, not general workload win rate.'
hashes={str(history):hashlib.sha256(history.read_bytes()).hexdigest()}
for art in {Path(r['evidence']) for r in selected}:
 for f in ('confirmation_statistics.json','planned_cases.json','results.jsonl'):
  p=art/f;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'evidence.json').write_text(json.dumps(dict(policy=policy,selected=selected,source_sha256=hashes),indent=2)+'\n')
(out/'shapes.json').write_text(json.dumps({'shapes':[dict(id=r['shape_id'],m=r['m'],k=r['k'],n=r['n'])for r in selected]},indent=2)+'\n')
print(policy)
for r in selected:print(r['shape_id'],r['one_level_comparisons']['native']['ci95'],r['comparisons']['native']['ci95'])
