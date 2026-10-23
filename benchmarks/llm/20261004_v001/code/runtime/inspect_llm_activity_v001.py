"""Read-only process and file-metadata progress for a single owned LLM run."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', args.run_id):
        raise ValueError('Invalid run ID')
    root = Path('/content/Strassen_MM_Focus')
    run = root / 'runs' / args.run_id
    private = root / '.runtime_private' / args.run_id
    result = {'kind': 'llm_activity', 'utc_epoch': time.time(),
              'disk_free_bytes': shutil.disk_usage(run).free, 'processes': [], 'directories': {}}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if args.run_id not in command or not any(name in command for name in
                    ('llm_model_session_v001.py --worker', 'tools/run_llm_case_v003.py', 'tools/prepare_llm_case_v003.py')):
                continue
            status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            result['processes'].append({'pid': int(proc.name),
                'role': 'measurement' if 'tools/run_llm_case_v003.py' in command else 'worker_or_inputs',
                'state': status['State'].strip(), 'ppid': int(status['PPid']),
                'rss': status.get('VmRSS', '').strip(), 'peak_rss': status.get('VmHWM', '').strip(),
                'cpu_seconds': (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'),
                'io': (proc / 'io').read_text(), 'wchan': (proc / 'wchan').read_text()})
        except (OSError, ProcessLookupError):
            continue
    # Only count file sizes and mtimes; do not read checkpoint, cache or auth data.
    for label, folder in [('artifacts', run / 'artifacts'),
                          ('layout_cache', private / 'model/layout-cache'),
                          ('pilot_logits', private / 'model/pilot-original-bfloat16'),
                          ('native_logits', private / 'model/native-logits-bfloat16')]:
        stats = []
        for path in folder.rglob('*'):
            try:
                if path.is_file():
                    stats.append(path.stat())
            except FileNotFoundError:
                pass
        result['directories'][label] = {'files': len(stats), 'bytes': sum(s.st_size for s in stats),
                                       'last_mtime': max((s.st_mtime for s in stats), default=None)}
    result['memory'] = {line.split(':')[0]: line.split(':')[1].strip()
                        for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:'))}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
