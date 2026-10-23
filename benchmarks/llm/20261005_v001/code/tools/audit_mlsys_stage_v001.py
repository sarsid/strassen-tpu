"""Resolve a sealed phase receipt and run the independent scientific audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--stage',required=True)
    args=p.parse_args();receipt=json.loads((args.cohort/(args.stage+'-finished.json')).read_text())
    if receipt['status']!='completed':raise ValueError('Cannot audit incomplete phase')
    artifacts=Path(receipt['run'])/'artifacts'
    return subprocess.call([sys.executable,str(Path(__file__).with_name('audit_mlsys_shapes_v001.py')),
        '--artifacts',str(artifacts),'--output',str(Path(os.environ['STRASSEN_EXECUTION_DIR'])/'audit.json')])

if __name__=='__main__':raise SystemExit(main())
