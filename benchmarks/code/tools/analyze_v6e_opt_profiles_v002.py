"""Extract actual device module times; explicitly avoid utilization claims."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def analyze(file):
    events=json.loads(gzip.decompress(file.read_bytes()))['traceEvents']
    processes={e['pid']:e.get('args',{}).get('name','') for e in events
               if e.get('ph')=='M' and e.get('name')=='process_name'}
    device=[pid for pid,name in processes.items() if name.startswith('/device:TPU:')]
    if len(device)!=1:raise ValueError('Expected one actual TPU process')
    pid=device[0]
    threads={e['tid']:e.get('args',{}).get('name','') for e in events
             if e.get('ph')=='M' and e.get('name')=='thread_name' and e.get('pid')==pid}
    tracks=[tid for tid,name in threads.items() if name=='XLA Modules']
    if len(tracks)!=1:raise ValueError('Expected one TPU module track')
    modules=sorted([e for e in events if e.get('ph')=='X' and e.get('pid')==pid and e.get('tid')==tracks[0]],key=lambda e:e['ts'])
    if any(e.get('dur',0)<=0 for e in modules):raise ValueError('Invalid module duration')
    if any(a['ts']+a['dur']>b['ts']+.05 for a,b in zip(modules,modules[1:])):
        raise ValueError('Overlapping serialized module intervals')
    # Compilation metadata filenames identify exactly the emitted host annotations.
    root=file.parents[3]
    arms={p.name.removesuffix('.hlo.txt') for p in root.glob('*.hlo.txt')}
    annotations=[e for e in events if e.get('ph')=='X' and e.get('pid')!=pid and e.get('name') in arms]
    if len(annotations)!=20*len(arms):raise ValueError('Incomplete host annotations')
    # TPU and host timestamp origins are offset in these traces. The runner
    # blocks after every invocation, so chronological order is the mapping.
    # Require full cardinality and one stable executable identity per arm.
    annotations.sort(key=lambda e:e['ts'])
    if len(modules)!=len(annotations):raise ValueError('Incomplete serialized device coverage')
    grouped=defaultdict(list);identities=defaultdict(set)
    for module,annotation in zip(modules,annotations):
        grouped[annotation['name']].append(module['dur']/1000)
        identities[annotation['name']].add(module['name'])
    if any(len(v)!=1 for v in identities.values()):
        raise ValueError('Executable identity changed inside one serialized arm')
    if set(grouped)!=arms or any(len(v)!=20 for v in grouped.values()):
        raise ValueError('Expected 20 device module observations for every arm')
    return dict(group=root.name,trace=str(file),trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        device_process=processes[pid],device_tracks=list(threads.values()),
        possible_drop_indicators=sorted({e.get('name','') for e in events if any(s in e.get('name','').lower() for s in ('dropped','overflow','truncated'))}),
        arms={k:dict(sample_count=len(v),mean_ms=statistics.mean(v),median_ms=statistics.median(v),
                     min_ms=min(v),max_ms=max(v),samples_ms=v) for k,v in grouped.items()},
        mapping='Serialized invocation order with full coverage and stable module identity; host/device clocks are not assumed aligned.',
        scope='Instrumented device module durations. No direct MXU utilization, memory bandwidth, stall or intra-kernel overlap counters.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    results=[];errors=[]
    for file in sorted(a.cohort.glob('phases/*/artifacts/profiles/**/*.trace.json.gz')):
        try:results.append(analyze(file))
        except Exception as exc:errors.append(dict(trace=str(file),error_type=type(exc).__name__,message=str(exc)))
    (a.output_dir/'profiles.json').write_text(json.dumps(dict(results=results,errors=errors),indent=2)+'\n')
    lines=['# v6e device module profiles','',
        'These are profiler-instrumented device durations. Ordinary synchronized call latencies are reported separately. No direct utilization, stall, bandwidth or overlap counters were captured.','']
    for r in results:
        lines += ['## '+r['group'],'','| Kernel | Device mean ms | Observations |','|---|---:|---:|']
        for arm,v in r['arms'].items():lines.append(f'| {arm} | {v["mean_ms"]:.6f} | {v["sample_count"]} |')
        lines.append('')
    for e in errors:lines.append('- Unavailable trace analysis: '+e['message'])
    (a.output_dir/'PROFILES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(profiles=len(results),errors=errors,output=str(a.output_dir))))
    return 1 if errors or not results else 0


if __name__=='__main__':raise SystemExit(main())
