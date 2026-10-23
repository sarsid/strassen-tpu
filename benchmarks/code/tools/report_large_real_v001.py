"""Summarize sealed per-model evidence, preserving failed quality gates."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=False);data={};lines=['# Larger real-weight model results','',
 'Actual checkpoint weights and natural text. Projection row counts2048/8192/16384 concatenate independent2048-token contexts. Full-model measurements use2048-token contexts.','']
for key in ['qwen','mistral','gemma']:
    receipt=a.cohort/(key+'-actual-finished.json')
    if not receipt.is_file():data[key]={'status':'not_executed'};lines += [f'## {key}','Not executed; inspect preparation/controller records.',''];continue
    run=Path(json.loads(receipt.read_text())['run']);summary=run/'artifacts/model_summary.json'
    if not summary.is_file():
        d=json.loads((run/'artifacts/summary.json').read_text());data[key]=d;lines += [f'## {key}',f'Execution did not produce a complete model summary: {d.get("error")}.',''];continue
    d=json.loads(summary.read_text());data[key]=d;lines += [f'## {d["model_id"]}',f'Status: {d["status"]}. Independent Native qualification: {d["qualification"]["passed"]}.','']
    if d.get('quality_measured'):
        lines += ['| Policy | Quality gate | Perplexity | Top1 agreement | Resident layer mean ms | Streamed forward mean s |','|---|---|---:|---:|---:|---:|']
        for arm,q in d['quality'].items():
            lines += [f'| {arm} | {q["passed"]} | {q["candidate_perplexity"]:.6f} | {q["top1_agreement"]:.6%} | {d["resident_layer_timings"][arm]["mean_ms"]:.6f} | {d["streamed_model_timings"][arm]["mean_ms"]/1000:.6f} |']
        lines += ['', 'Resident composed-layer means and three streamed repeats are descriptive. Separate full-JIT Native controls are retained.','']
    lines += [f'Evidence: {summary}','']
(a.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n');(a.output_dir/'results.json').write_text(json.dumps(data,indent=2)+'\n')
(a.output_dir/'artifact-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output_dir.iterdir() if p.is_file()},indent=2)+'\n')
print(json.dumps({'reported_models':list(data),'output':str(a.output_dir)}))
