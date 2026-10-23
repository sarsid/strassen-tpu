"""Finish this one running campaign with an audited report and figures."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--wait-seconds', type=int, default=18000)
    args = parser.parse_args()
    root = Path(os.environ['STRASSEN_PROJECT_ROOT']).resolve()
    run = Path(os.environ['STRASSEN_EXECUTION_DIR']).resolve()
    cohort = args.cohort.resolve()
    if not cohort.is_relative_to(root / 'runs'):
        raise ValueError('Input cohort must be within the study')
    artifacts = run / 'artifacts'
    artifacts.mkdir(exist_ok=False)
    tools = Path(__file__).resolve().parent

    def progress(state, title, detail, evidence):
        row = {'kind': 'activity', 'id': 'current', 'name': title, 'title': title,
               'state': state, 'detail': detail, 'utc': datetime.now(timezone.utc).isoformat(),
               'evidence': [{'label': p.name, 'path': str(p.relative_to(root))} for p in evidence]}
        with (root / 'status/work_progress_v001.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(row) + '\n')

    deadline = time.monotonic() + args.wait_seconds
    completion = cohort / 'controller-completion.json'
    print('Waiting for the existing campaign; no device jobs are launched by this reporter.', flush=True)
    try:
        while not completion.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError('Existing campaign did not finish within the reporting wait budget')
            time.sleep(10)
        result = json.loads(completion.read_text())
        if result['status'] != 'completed' or result.get('released') is not True:
            raise RuntimeError('Campaign needs attention; consult its preserved completion receipt')
        progress('working', 'TPU run complete; auditing and plotting results',
                 'All phases retrieved and the v5e allocation released. Replaying selection and confidence intervals.', [completion])
        report = artifacts / 'report'
        figures = artifacts / 'figures'
        subprocess.run([sys.executable, str(tools / 'summarize_llm_mini_v001.py'),
                        '--cohort', str(cohort), '--output-dir', str(report)], check=True)
        subprocess.run([sys.executable, str(tools / 'plot_llm_mini_v001.py'),
                        '--report-dir', str(report), '--output-dir', str(figures)], check=True)
        data = json.loads((report / 'comparisons.json').read_text())
        progress('complete', 'Larger LLM MM study complete',
                 '18 shapes confirmed; evidence audit and comparison figures complete. '
                 + str(len(data['failures'])) + ' non-ok outcomes retained. TPU released.',
                 [report / 'RESULTS.md', figures / 'latency_by_model.png', figures / 'strassen_speedup_ci.png'])
        result = {'completed': True, 'report': str(report), 'figures': str(figures)}
    except Exception as error:
        result = {'completed': False, 'error_type': type(error).__name__, 'error': str(error)}
        progress('attention', 'Larger LLM MM report needs attention',
                 str(error), [completion] if completion.exists() else [cohort / 'cohort.json'])
    with (artifacts / 'reporting-completion.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
