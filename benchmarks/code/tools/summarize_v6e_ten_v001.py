"""Join ten-shape confirmations to historical evidence without reselecting."""
import argparse,json,hashlib,math,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
report=a.cohort/'operations/report/artifacts/results.json';data=json.loads(report.read_text())['results']
screen=Path(json.loads((a.cohort/'opt-screen-finished.json').read_text())['run'])/'artifacts/selections.json'
selected=json.loads(screen.read_text())['selected']
hist=a.cohort/'source/configs/v6e_ten_v001/v5e_evidence.json';history=json.loads(hist.read_text())['selected']
confirmation=Path(json.loads((a.cohort/'opt-confirm-finished.json').read_text())['run'])/'artifacts/results.jsonl'
events=[json.loads(l)for l in confirmation.read_text().splitlines()];samples=[r for r in events if r['event']=='sample']
lines=['# Ten historical v5e winners, retested on v6e','',
 'Dimensions are M,K,N; milliseconds are complete synchronized device-call latency. Same quantized BF16 operands and preadditions, FP32 accumulation/output. Current JAX/jaxlib 0.11.2, libtpu 0.0.48. Synthetic operands, not full-model inference.',
 '', 'All 210 screening candidates were frozen in advance. All methods for a shape use common inputs and paired randomized timing rounds. Each family selects its fastest numerically eligible screen candidate. The depth headline below chooses between original and revised families using SCREEN means only, before examining fresh confirmation timings. Three fresh seeds, 30 paired rounds each. No confirmation reselection.',
 '', '| M × K × N | v5e S1 speedup | v5e S2 speedup | v6e Native ms | v6e S1 ms | v6e S2 ms | v6e S1 speedup [95%] | v6e S2 speedup [95%] |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
rows=[]
def raw(shape,arm):return {(r['seed'],r['round']):r['elapsed_ms']for r in samples if r['shape_id']==shape and r['arm_id']==arm}
def interval(shape,baseline,candidate):
 n=raw(shape,baseline);c=raw(shape,candidate);assert n.keys()==c.keys()and len(n)==90
 keys=sorted(n);nn=np.array([n[k]for k in keys]).reshape(3,30);cc=np.array([c[k]for k in keys]).reshape(3,30)
 rng=np.random.default_rng(2390);si=rng.integers(0,3,(4000,3,1));ri=rng.integers(0,30,(4000,3,30))
 z=nn[si,ri].mean((1,2))/cc[si,ri].mean((1,2));return dict(speedup=float(nn.mean()/cc.mean()),ci95=np.quantile(z,[.025,.975]).tolist())
for h in history:
 shape=h['shape_id'];s=selected[shape];methods={}
 for d in (1,2):
  choices=[k for k in ('original'+str(d),'optimized'+str(d))if s[k]['status']=='eligible']
  key=min(choices,key=lambda k:(s[k]['screen_mean_ms'],k));v=data[shape][key];assert 'mean_ms'in v
  timing=interval(shape,s['native']['candidate']['arm_id'],s[key]['candidate']['arm_id'])
  default=interval(shape,'native_default',s[key]['candidate']['arm_id'])
  metrics=[r['correctness']for r in v['cases']]
  methods[str(d)]=dict(family=key,candidate=v['candidate'],mean_ms=v['mean_ms'],vs_tuned=timing,vs_default=default,
      numerically_eligible=v['numerically_eligible'],worst_errors={k:max(q[k]for q in metrics)for k in ('relative_l2','max_abs_error','rmse','mean_abs_error','p99_abs_error')if all(k in q for q in metrics)})
 native=data[shape]['native'];row=dict(shape_id=shape,shape_mkn=[h[k]for k in ('m','k','n')],v5e=h,methods=methods,native_ms=native['mean_ms'],native_default_ms=data[shape]['native_default']['mean_ms']);rows.append(row)
 def fmt(d):
  v=methods[str(d)]['vs_tuned'];return f'{v["speedup"]:.3f}× [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}]'
 lines.append(f'| {h["m"]} × {h["k"]} × {h["n"]} | {h["one_level_comparisons"]["native"]["speedup"]:.3f}× | {h["comparisons"]["native"]["speedup"]:.3f}× | {native["mean_ms"]:.6f} | {methods["1"]["mean_ms"]:.6f} | {methods["2"]["mean_ms"]:.6f} | {fmt(1)} | {fmt(2)} |')
lines+=['', 'Speedup above 1 is faster. Historical and new numbers are each ratios to their own tuned Native baseline. They come from different allocations/compiler environments and are not a controlled hardware-only comparison. The historical set is deliberately enriched for wins; it does not establish a general win rate. Intervals are pointwise hierarchical paired bootstraps, conditional on the frozen selection, without multiplicity adjustment.', '', '## Numerical cost and decisions','',
 '| Shape | Depth | Frozen family | Tile BM,BN,BK | Relative L2 | Sampled max absolute error | RMSE | Gate |','|---|---:|---|---|---:|---:|---:|---|']
for r in rows:
 for d,v in r['methods'].items():
  e=v['worst_errors'];lines.append(f'| {r["shape_id"]} | {d} | {v["family"]} | {v["candidate"]["tile"]} | {100*e["relative_l2"]:.4f}% | {e["max_abs_error"]:.6g} | {e.get("rmse",float("nan")):.6g} | {v["numerically_eligible"]} |')
summary={}
for d in ('1','2'):
 v=[r['methods'][d]for r in rows];summary[d]=dict(mean_wins=sum(x['numerically_eligible']and x['vs_tuned']['speedup']>1 for x in v),ci_wins=sum(x['numerically_eligible']and x['vs_tuned']['ci95'][0]>1 for x in v),ci_losses=sum(x['vs_tuned']['ci95'][1]<1 for x in v),geomean_speedup=math.exp(statistics.mean(math.log(x['vs_tuned']['speedup'])for x in v)),worst_relative_l2=max(x['worst_errors']['relative_l2']for x in v))
lines += ['',
 'Error references use exact BF16-quantized inputs and host FP64 accumulation over every K, on 128 sampled rows × 128 sampled columns. Finiteness is checked over the full output. These are sampled matrix error metrics, not prediction accuracy or perplexity. Gates remain relative L2 ≤2% and max absolute error ≤0.001 + 0.05·max|reference|; all raw metrics and failed candidates are retained.', '',
 '## Evidence','',
 '- Architecture diagnostic: runs/20260923T181010Z-v6e-diagnostic-summary-v001-e874b9/artifacts/RESULTS.md.',
 '- Full per-family timings, numerical errors, compiler failures and selections: operations/report/artifacts/RESULTS.md within this cohort.',
 '- Ordinary device profiles are collected separately after first-input confirmation; profile samples are not mixed with synchronized-call timings.', '']
(a.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(a.output_dir/'summary.json').write_text(json.dumps(dict(summary=summary,rows=rows,input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in (report,screen,hist,confirmation)}),indent=2)+'\n')
print(json.dumps(summary,indent=2))
