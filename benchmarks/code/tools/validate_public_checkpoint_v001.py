"""Archive credential-pattern scanning and full CPU report replay for publication."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    run = Path(os.environ['STRASSEN_EXECUTION_DIR'])
    checkpoint = args.checkpoint.resolve()
    here = Path(__file__).resolve().parent
    artifacts = run / 'artifacts'
    artifacts.mkdir()
    subprocess.run([sys.executable, str(here / 'scan_public_checkpoint_v001.py'),
                    '--root', str(checkpoint), '--output', str(artifacts / 'credential-scan.json')], check=True)
    subprocess.run([sys.executable, str(checkpoint / 'replay.py'), '--root', str(checkpoint),
                    '--output', str(artifacts / 'report-replay.json')], check=True)
    result = dict(passed=True, scope='Local export integrity, credential-pattern scan and scientific report replay; no new accelerator measurements.',
                  manifest_sha256=__import__('hashlib').sha256((checkpoint / 'MANIFEST.json').read_bytes()).hexdigest(),
                  source_commit=json.loads((checkpoint / 'checkpoint.json').read_text())['source_commit'])
    (artifacts / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
