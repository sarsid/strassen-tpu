"""Export completed original/current-stack parent runs without model weights."""
import argparse,hashlib,json,math,shutil
from pathlib import Path

def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def one(rows,kind):
    values=[r for r in rows if r.get('kind')==kind]
    if len(values)!=1:raise ValueError(f'Expected one {kind}, got {len(values)}')
    return values[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    completion=json.loads((a.run/'completion.json').read_text())
    if completion.get('status')!='completed' or completion.get('release_exit_code')!=0:raise ValueError('Run or release incomplete')
    a.output.mkdir(parents=True,exist_ok=False)
    data=a.run/'remote'/a.run.name/'artifacts';summary={};sources={}
    expected_tokens=dict(streamed='70b86198317db1dee0c351568cf420d78242aa74e0b9a6cbb14925eaa55ca2a5',
                         quality='4aa8e3628d16c9a26876aedb8c99525c6a15ee601c88d93deadae15855a84d74')
    for stack in ('parent','current'):
        summary[stack]={}
        for stage in ('layer','streamed','quality'):
            source=data/(stack+'-'+stage)/'events.jsonl';rows=read(source)
            target=a.output/(stack+'-'+stage+'.jsonl');shutil.copyfile(source,target)
            sources[target.name]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            environment=one(rows,'replication_environment');verdict=one(rows,'verdict')
            result=dict(environment=environment,verdict=verdict)
            if stage=='layer':
                result.update(performance=one(rows,'performance'),accuracy=one(rows,'accuracy'))
            else:
                layers=[r for r in rows if r.get('kind')=='layer']
                if sorted(r['layer'] for r in layers)!=list(range(64)):raise ValueError('Incomplete layer sequence')
                tokens=one(rows,'tokens')
                if tokens['sha256']!=expected_tokens[stage]:raise ValueError('Token identity mismatch')
                result.update(tokens=tokens,task=one(rows,'task'),layers=64)
                if stage=='streamed':result['performance']=one(rows,'aggregate_performance')
            summary[stack][stage]=result
    parent=root/'third_party/strassen_tpu_public_95be1fb_v001/evidence/qwen3'
    published=read(parent/'strassen_qwen3_32b_streamed_product_inference_v6e_fusedqk.jsonl')
    reproduced=read(a.output/'parent-streamed.jsonl')
    errors=lambda rows:{r['layer']:r['errors_vs_regular_xla'] for r in rows if r.get('kind')=='layer'}
    summary['published_agreement']=dict(all_64_layer_error_records_equal=errors(published)==errors(reproduced),
                                        repetitive_task_metrics_equal=one(published,'task')==one(reproduced,'task'),
                                        published_block_speedup=1.2093,published_streamed_speedup=1.2949971494360175)
    summary['scope']='Qwen3-32B B8 S1024; resident compute excludes downloads, packing, transfer, embedding lookup and final scoring'
    summary['source_run']=str(a.run);summary['completion']=completion
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Parent Qwen3-32B reproduction on v6e','',summary['scope']+'.','',
           '| Environment | Native block ms | S1 block ms | Block speedup | Native 64-layer ms | S1 64-layer ms | Streamed speedup |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for stack in ('parent','current'):
        block=summary[stack]['layer']['performance'];stream=summary[stack]['streamed']['performance']
        b=block['mean_ms'];s=stream['aggregate_ms'];env=summary[stack]['layer']['environment']['versions']
        lines.append(f'| JAX {env["jax"]}, libtpu {env["libtpu"]} | {b["regular_xla"]:.3f} | {b["product_strassen_fused"]:.3f} | {block["speedup_vs_xla"]:.4f}x | {s["regular_xla"]:.3f} | {s["gated_strassen"]:.3f} | {stream["speedup_strassen_vs_xla"]:.4f}x |')
    lines+=['','The published targets are 1.2093x for the block and 1.2950x for streamed resident compute.',
            'The single-block and streamed parent scripts use different down-projection tiles; both are preserved as written.','',
            '| Environment | WikiText Native PPL | S1 PPL | Absolute NLL delta | KL | Top-1 agreement | Quality gate |',
            '|---|---:|---:|---:|---:|---:|---|']
    for stack in ('parent','current'):
        r=summary[stack]['quality']['task']['results']['gated_strassen'];v=summary[stack]['quality']['verdict']
        lines.append(f'| {stack} | {math.exp(r["native_loss"]):.6f} | {math.exp(r["candidate_loss"]):.6f} | {r["absolute_loss_delta"]:.8f} | {r["mean_kl_nats"]:.8f} | {100*r["top1_agreement"]:.4f}% | {"Pass" if v["passes"] else "Fail"} |')
    lines+=['','All 64 original-stack per-layer error records and repetitive-text task metrics match the published artifact exactly.' if all(summary['published_agreement'][k] for k in ('all_64_layer_error_records_equal','repetitive_task_metrics_equal')) else 'See summary.json for numeric agreement checks.',
            '', 'These are parent-code replications. A newer-stack improvement is a software-environment effect, not evidence of an improvement in our kernels.',
            'WikiText covers 32,736 scored positions; repetitive text covers 8,184. Agreement with the parent Native implementation is not independent official-model qualification.',
            '', 'Detailed timing samples, error metrics, token hashes and metadata are in the six JSONL files. Source, setup and release evidence are in the linked source run in summary.json.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    sources['summary.json']=dict(sha256=hashlib.sha256((a.output/'summary.json').read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(dict(source_run=str(a.run),files=sources),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),published_agreement=summary['published_agreement'])))

if __name__=='__main__':main()
