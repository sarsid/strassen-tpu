"""Export a completed, released comparison with raw tuning and quality evidence."""
import argparse,hashlib,json,shutil
from pathlib import Path


def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def one(rows,kind):
    matches=[r for r in rows if r.get('kind')==kind]
    if len(matches)!=1:raise ValueError(f'Expected one {kind}, found {len(matches)}')
    return matches[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();completion=json.loads((a.run/'completion.json').read_text())
    if completion['status']!='completed' or completion['release_exit_code']!=0:raise ValueError('Run and release must be complete')
    data=a.run/'remote'/a.run.name/'artifacts';a.output.mkdir(parents=True,exist_ok=False);manifest={}
    def copy(source,name):
        shutil.copyfile(source,a.output/name)
        manifest[name]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    summary=dict(source_run=str(a.run),completion=completion,arms={},scope='Qwen3-32B B8 S1024; resident 64-layer compute sum, not end-to-end serving latency')
    for title in ('tune','refine'):
        for name in ('profile.json','summary.json','events.jsonl'):copy(data/title/name,title+'-'+name)
    profile=json.loads((data/'refine/profile.json').read_text());summary['profile']=profile
    summary['confirmation']=one(records(data/'refine/events.jsonl'),'confirmation')
    copy(data/'refinement-handoff.json','refinement-handoff.json')
    expected=dict(streamed='70b86198317db1dee0c351568cf420d78242aa74e0b9a6cbb14925eaa55ca2a5',
                  quality='4aa8e3628d16c9a26876aedb8c99525c6a15ee601c88d93deadae15855a84d74')
    for dtype in ('bfloat16','float32'):
        for family in ('s1','s2'):
            arm=family+'-'+dtype;summary['arms'][arm]={}
            for stage in ('streamed','quality'):
                title=arm+'-'+stage;rows=records(data/title/'events.jsonl');copy(data/title/'events.jsonl',title+'.jsonl')
                layers=[r for r in rows if r.get('kind')=='layer']
                if sorted(r['layer'] for r in layers)!=list(range(64)):raise ValueError(f'Incomplete layers: {title}')
                tokens=one(rows,'tokens')
                if tokens['sha256']!=expected[stage]:raise ValueError('Token hash mismatch')
                metadata=one(rows,'metadata')
                if metadata['profile_sha256']!=hashlib.sha256((data/'refine/profile.json').read_bytes()).hexdigest():raise ValueError('Unexpected profile')
                result=dict(environment=one(rows,'environment'),metadata=metadata,tokens=tokens,
                            task=one(rows,'task'),verdict=one(rows,'verdict'),layers=64,
                            layer_errors=[{k:v for k,v in r.items() if k in ('layer','errors_vs_regular_xla','finite')} for r in layers])
                if stage=='streamed':result['performance']=one(rows,'aggregate_performance')
                summary['arms'][arm][stage]=result
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Our fused kernels in the parent Qwen3-32B experiment','',summary['scope']+'.','',
           'Final profiles use the separately archived interleaved v005 refinement. The v004 screen remains intact; its normalized scores are not reported runtimes.','',
           '| Custom store | Family | Native 64-layer ms | Cubic ms | Candidate ms | Speedup vs Native | Speedup vs cubic |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,r in summary['arms'].items():
        s=r['streamed']['performance']['aggregate_ms'];family,dtype=name.split('-')
        lines.append(f'| {dtype} | {family.upper()} | {s["regular_xla"]:.3f} | {s["gated_cubic"]:.3f} | {s["gated_strassen"]:.3f} | {s["regular_xla"]/s["gated_strassen"]:.4f}x | {s["gated_cubic"]/s["gated_strassen"]:.4f}x |')
    lines+=['','| Custom store | Family | WikiText Native PPL | Candidate PPL | Absolute NLL delta | KL | Top-1 agreement | Candidate quality gate |',
            '|---|---|---:|---:|---:|---:|---:|---|']
    for name,r in summary['arms'].items():
        q=r['quality']['task']['results']['gated_strassen'];family,dtype=name.split('-')
        lines.append(f'| {dtype} | {family.upper()} | {q["native_perplexity"]:.6f} | {q["candidate_perplexity"]:.6f} | {q["absolute_loss_delta"]:.8f} | {q["mean_kl_nats"]:.8f} | {100*q["top1_agreement"]:.4f}% | {"Pass" if q["passes_task_gate"] else "Fail"} |')
    lines+=['','Native fallback is eligible at each site; family names describe tuned policies, not a claim that every projection uses Strassen.',
            'FP32/BF16 label custom kernel stores. Both feed the BF16 parent model, with consumer conversion included in timing. This is not an all-FP32 model comparison.',
            'Our Q/K normalization retains HIGHEST dot precision, while the unchanged parent uses DEFAULT. Parent-vs-ours differences therefore include this precision choice, packing/layout and tile policy, not only the multiplication algorithm.',
            'Downloads, compilation, weight packing/transfer, embedding lookup and final scoring are excluded from resident compute timing.',
            'Quality is teacher-forced agreement against the parent Native implementation, over 32,736 WikiText positions and 8,184 repetitive-text positions. It is not an independent official-model qualification.',
            'All execution outcomes, quality/speed verdicts, raw samples, per-layer errors and profile details remain in summary.json and the JSONL files. Execution completion does not imply every benchmark gate passed.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    for name in ('summary.json','README.md'):manifest[name]=dict(sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),arms=list(summary['arms']))))


if __name__=='__main__':main()
