"""Export the declared timing repeat and memory-cap audit as separate evidence."""
import argparse,hashlib,json,shutil
from pathlib import Path
from export_parent_comparison_v001 import records,one


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--main-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.run/'completion.json').read_text());main_receipt=json.loads((a.main_run/'completion.json').read_text())
    if receipt['status']!='completed' or main_receipt['status']!='completed' or main_receipt['release_exit_code']!=0:raise ValueError('Follow-up/main/release not complete')
    data=a.run/'remote'/a.run.name/'artifacts';a.output.mkdir(parents=True,exist_ok=False);manifest={}
    def copy(source,name):
        shutil.copyfile(source,a.output/name);manifest[name]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for name in ('input-provenance.json','changed-profiles.json'):copy(data/name,name)
    for name in ('profile.json','summary.json','events.jsonl'):copy(data/'budget'/name,'budget-'+name)
    budget=json.loads((data/'budget/summary.json').read_text());changed=json.loads((data/'changed-profiles.json').read_text())['changed']
    summary=dict(source_run=str(a.run),main_run=str(a.main_run),completion=receipt,release_evidence=main_receipt,
                 budget=budget,changed=changed,stages={})
    titles=['s1-bfloat16-repeat']+[r['family']+'-'+r['dtype']+'-'+stage for r in changed for stage in ('streamed','quality')]
    for title in titles:
        source=data/title/'events.jsonl';rows=records(source);copy(source,title+'.jsonl')
        layers=[r for r in rows if r.get('kind')=='layer']
        if sorted(r['layer'] for r in layers)!=list(range(64)):raise ValueError('Incomplete follow-up layers')
        result=dict(metadata=one(rows,'metadata'),environment=one(rows,'environment'),tokens=one(rows,'tokens'),
                    task=one(rows,'task'),verdict=one(rows,'verdict'))
        if not title.endswith('quality'):result['performance']=one(rows,'aggregate_performance')
        summary['stages'][title]=result
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Declared timing repeat and v6e down-cap audit','',
           'This follow-up reuses the same owned v6e, current software stack and checkpoint cache after all eight main evaluation stages. Main-study results remain unchanged.','',
           'Exactly one unchanged S1/BF16 streamed repeat tests the isolated 197.153625 ms sample in the first pass. No sample is deleted and neither pass is substituted for the other.','',
           'The down-only audit raises the (2048,2560,1024), two-buffer kernel cap from 112 MiB to the parent allowances (120 MiB Strassen, 124 MiB cubic). It retains current settings unless the interleaved score improves by more than 1%. Other sites and norm precision stay fixed.','',
           '| Stage | Native ms | Cubic ms | S1/S2 ms | Speedup vs Native | Performance gate |',
           '|---|---:|---:|---:|---:|---|']
    for name,r in summary['stages'].items():
        if 'performance' not in r:continue
        q=r['performance'];s=q['aggregate_ms']
        lines.append(f'| {name} | {s["regular_xla"]:.3f} | {s["gated_cubic"]:.3f} | {s["gated_strassen"]:.3f} | {q["speedup_strassen_vs_xla"]:.4f}x | {"Pass" if q["passes"] else "Fail"} |')
    lines+=['','Changed profile groups: '+(', '.join(r['family']+'/'+r['dtype'] for r in changed) or 'none')+'.',
            'Changed profiles receive separate 64-layer and WikiText evaluations. Unchanged profiles retain the main-study quality evidence; they are not counted as new quality measurements.',
            'Timing remains the sum of resident layer compute, excluding checkpoint I/O, compilation, weight preparation/transfer, embedding lookup and final scoring. Both output-store variants feed a BF16 model.',
            'Raw samples, gates, profile changes, compilation failures and numerical errors are preserved in the JSONL files and summary.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    for name in ('summary.json','README.md'):manifest[name]=dict(sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),changed=changed)))


if __name__=='__main__':main()
