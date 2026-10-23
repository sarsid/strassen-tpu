"""Attribute saved phase wall time without changing the running experiment."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    phases = []
    for receipt in sorted(a.cohort.glob('arch-*-finished.json')):
        if not receipt.name.endswith(('-screen-finished.json', '-confirm-finished.json')):
            continue
        info = read(receipt)
        if info['status'] != 'completed':
            continue
        run = Path(info['run'])
        completion = read(run / 'completion.json')
        events_path = run / 'artifacts/results.jsonl'
        raw = events_path.read_bytes()
        hashes[str(events_path)] = hashlib.sha256(raw).hexdigest()
        events = [json.loads(line) for line in raw.splitlines()]
        remote = completion['remote_completion']
        compiles = []
        starts = {}
        input_seconds = 0.0
        group_start = None
        first_compile = False
        for e in events:
            if e['event'] == 'group_start':
                group_start = e['monotonic_ns']
                first_compile = True
            if e['event'] == 'compile_start':
                key = (e['group_id'], e['arm_id'])
                if key in starts:
                    raise ValueError('Duplicate pending compilation')
                starts[key] = e
                if first_compile:
                    input_seconds += (e['monotonic_ns'] - group_start) / 1e9
                    first_compile = False
            terminal = e['event'] == 'compilation' or (e['event'] == 'error' and e.get('during') == 'compile')
            if terminal:
                start = starts.pop((e['group_id'], e['arm_id']))
                seconds = (e['monotonic_ns'] - start['monotonic_ns']) / 1e9
                assert seconds >= 0
                compiles.append(dict(family=start.get('family'), status=e['status'], seconds=seconds))
        assert not starts
        total_compile = sum(x['seconds'] for x in compiles)
        sample_seconds = sum(e['elapsed_ms'] / 1000 for e in events if e['event'] == 'sample')
        wall = remote['elapsed_seconds']
        assert total_compile + sample_seconds <= wall + 1
        cases = [e for e in events if e['event'] == 'case_result']
        families = {}
        for family in sorted({x['family'] for x in compiles}):
            rows = [x for x in compiles if x['family'] == family]
            families[family] = dict(count=len(rows), seconds=sum(x['seconds'] for x in rows),
                                   failed_seconds=sum(x['seconds'] for x in rows if x['status'] != 'ok'))
        phases.append(dict(phase=receipt.name.removesuffix('-finished.json'),
            shape_mkn=next(e['shape_mkn'] for e in events if e['event'] == 'group_start'),
            started_utc=remote['started_utc'], finished_utc=remote['finished_utc'],
            remote_wall_seconds=wall, local_orchestration_seconds=completion['orchestration_wall_seconds'],
            compilation_seconds=total_compile,
            failed_compilation_seconds=sum(x['seconds'] for x in compiles if x['status'] != 'ok'),
            measured_call_seconds=sample_seconds,
            other_remote_seconds=wall-total_compile-sample_seconds,
            group_start_to_first_compile_seconds=input_seconds,
            compilation_count=len(compiles), sample_count=sum(e['event']=='sample' for e in events),
            status_counts=dict(Counter(e['status'] for e in cases)), compilation_by_family=families,
            source_run=str(run)))
    def aggregate(rows):
        fields = ['remote_wall_seconds', 'local_orchestration_seconds', 'compilation_seconds',
                  'failed_compilation_seconds', 'measured_call_seconds', 'other_remote_seconds',
                  'group_start_to_first_compile_seconds', 'compilation_count', 'sample_count']
        result = {k:sum(r[k] for r in rows) for k in fields}
        status = Counter()
        for r in rows:
            status.update(r['status_counts'])
        result['status_counts'] = dict(status)
        result['phases'] = len(rows)
        return result
    events = [json.loads(line) for line in (a.cohort/'controller-events.jsonl').read_text().splitlines()]
    active = {}
    stages = []
    for e in events:
        if e['event'] == 'stage_started':
            active[e['stage']] = e['utc']
        elif e['event'] == 'stage_finished' and e['stage'] in active:
            start = active.pop(e['stage'])
            seconds = (datetime.fromisoformat(e['utc']) - datetime.fromisoformat(start)).total_seconds()
            stages.append(dict(stage=e['stage'], seconds=seconds, state=e['state'], started_utc=start, finished_utc=e['utc']))
    pairs = defaultdict(list)
    for r in phases:
        pairs[r['phase'].rsplit('-',1)[0]].append(r)
    whole_pairs = [v for v in pairs.values() if len(v) == 2]
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), cohort=str(a.cohort),
        totals=aggregate(phases), by_phase={stage:aggregate([r for r in phases if r['phase'].endswith('-'+stage)]) for stage in ['screen','confirm']},
        completed_pair_count=len(whole_pairs),
        pair_worker_seconds_median=statistics.median(sum(r['remote_wall_seconds'] for r in pair) for pair in whole_pairs),
        export_stage_seconds=sum(s['seconds'] for s in stages if s['stage'].startswith('export-')),
        measurement_stage_seconds=sum(s['seconds'] for s in stages if not s['stage'].startswith('export-')),
        phases=phases, stages=stages, input_sha256=hashes,
        interpretation='Completed phases only on this cohort. Compile durations use paired compile_start and compilation/error events, including failed compilation. Measured calls exclude warmups. Other remote time includes input generation, transfer, reference and output checks, warmups, metadata, logging and cleanup; these components are not separately timed. Group-start to first-compile is a subset of other time, chiefly first-input generation. No retry samples are merged for performance claims.')
    (a.output_dir/'duration.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['phases','stages','input_sha256']}),flush=True)


if __name__ == '__main__':
    main()
