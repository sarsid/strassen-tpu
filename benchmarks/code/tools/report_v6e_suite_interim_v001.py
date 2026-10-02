"""Archive a clearly marked interim report from sealed confirmation batches."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from report_v6e_suite_v001 import build
p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();started=datetime.now(timezone.utc).isoformat()
code=build(a.cohort,a.output_dir)
result=json.loads((a.output_dir/'results.json').read_text())
assert code in (0,1) and result['results']
notice=(f'INTERIM SNAPSHOT started {started}. {len(result["results"])}/168 shapes from sealed confirmation batches. '
        'The live campaign is unchanged; absent batches are still pending or require inspection. '
        'The suite is ordered by increasing matrix volume, so this subset is not representative of the full suite.\n\n')
report=a.output_dir/'RESULTS.md';report.write_text(notice+report.read_text())
(a.output_dir/'interim.json').write_text(json.dumps(dict(analysis_completed=True,campaign_completed=result['completed'],
    shapes_reported=len(result['results']),snapshot_started_utc=started,missing_batches=result['missing_batches']),indent=2)+'\n')
print('Interim analysis archived; the experiment was not modified')
