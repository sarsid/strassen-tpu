"""Read-only worker and archive diagnostics; never inspect private model files."""
import argparse
import json
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
    run = Path('/content/Strassen_MM_Focus/runs') / args.run_id
    result = {'kind': 'llm_finalization_diagnostic', 'time': time.time(),
              'disk_free': shutil.disk_usage(run).free, 'files': {}, 'workers': []}
    for path in [run / 'status.json', run / 'archive-ready.json',
                 run / 'worker.log', run.with_suffix('.tar.gz')]:
        if path.exists():
            stat = path.stat()
            result['files'][path.name] = {'bytes': stat.st_size, 'mtime': stat.st_mtime}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if '--worker' in command and str(run) in command:
                result['workers'].append({'pid': int(proc.name),
                    'status': (proc / 'status').read_text(),
                    'io': (proc / 'io').read_text(),
                    'wchan': (proc / 'wchan').read_text()})
        except (OSError, ProcessLookupError):
            pass
    result['artifact_bytes'] = sum(path.stat().st_size for path in (run / 'artifacts').rglob('*')
                                   if path.is_file() and not path.name.endswith('.tar.gz'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
