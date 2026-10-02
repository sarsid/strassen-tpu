#!/usr/bin/env python3
"""Read compact progress from a phase without importing JAX or altering files."""
import argparse
import json
import os
from pathlib import Path
import re

RUNS = Path('/content/Strassen_MM_Focus/runs')


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {'unavailable': True, 'reason': 'file is not readable as complete JSON yet'}


def tail_from(path, offset, limit):
    if not path.exists():
        return {'offset': offset, 'next_offset': offset, 'bytes': 0, 'text': '', 'missing': True}
    size = path.stat().st_size
    if offset > size:
        return {'offset': offset, 'next_offset': offset, 'bytes': size, 'text': '',
                'error': 'file_shrank_or_wrong_offset'}
    with path.open('rb') as stream:
        stream.seek(offset)
        data = stream.read(limit)
    return {'offset': offset, 'next_offset': offset + len(data), 'bytes': size,
            'more': offset + len(data) < size, 'text': data.decode('utf-8', errors='replace')}


def alive(pid):
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--log-offset', type=int, default=0)
    parser.add_argument('--result-offset', type=int, default=0)
    parser.add_argument('--worker-log-offset', type=int, default=0)
    parser.add_argument('--max-bytes', type=int, default=8192)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        parser.error('Invalid run ID')
    if min(args.log_offset, args.result_offset, args.worker_log_offset) < 0:
        parser.error('Offsets must be nonnegative')
    if not 1 <= args.max_bytes <= 65536:
        parser.error('--max-bytes must be between 1 and 65536')
    run = RUNS / args.run_id
    if not run.is_dir():
        raise FileNotFoundError('Run directory does not exist')
    worker = read_json(run / 'worker-start.json') or read_json(run / 'launcher-started.json') or {}
    benchmark = read_json(run / 'benchmark-start.json') or {}
    completion = read_json(run / 'completion.json')
    ready = read_json(run / 'archive-ready.json')
    raw_summary = read_json(run / 'artifacts' / 'summary.json')
    summary = None
    if raw_summary:
        summary = {key: raw_summary.get(key) for key in ('phase', 'completed', 'status',
                   'wall_seconds', 'planned_group_count', 'case_status_counts', 'error')}
        summary['completed_group_count'] = len(raw_summary.get('completed_groups', []))
        summary['not_completed_group_count'] = len(raw_summary.get('not_completed_group_ids', []))
    result = {'kind': 'phase_progress', 'run_id': args.run_id,
              'phase': (read_json(run / 'launch.json') or {}).get('phase'),
              'worker_alive': alive(worker.get('worker_pid')),
              'benchmark_alive': alive(benchmark.get('pid')),
              'completion': completion, 'archive_ready': ready,
              'archive_error': read_json(run / 'archive-error.json'),
              'summary': summary,
              'log': tail_from(run / 'benchmark.log', args.log_offset, args.max_bytes),
              'results': tail_from(run / 'artifacts' / 'results.jsonl', args.result_offset, args.max_bytes),
              'worker_log': tail_from(run / 'worker.log', args.worker_log_offset, args.max_bytes)}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
