"""Check the new evidence reader against the completed, immutable TPU smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import summarize_llm_mini_v001 as report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    output = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    output.mkdir(exist_ok=False)
    checks = report.Checks()
    error = None
    try:
        frozen = report.read(cohort / 'frozen.json')
        runner, numpy_version = report.frozen_runner(cohort, frozen, checks)
        config = cohort / 'source' / frozen['campaign_relative']
        campaign = report.read(config)
        shape = config.parent / campaign['shape_manifest']
        distribution = config.parent / campaign['distribution_manifest']
        artifacts = report.phase_artifacts(cohort, 'GRID-smoke', frozen,
                                           (config, shape, distribution), checks)
        observed = report.inspect_phase(artifacts, 'GRID-smoke', campaign,
                                       report.read(shape), runner, checks)
        report.ensure_clean(checks)
        assert observed.groups == 2 and observed.scoped_cases == 12
        assert observed.status_counts == {'ok': 12}
        assert 'jax' not in sys.modules
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    result = {'passed': error is None and not checks.issues, 'scope': 'Read-only audit of sealed smoke evidence; no kernels',
              'check_counts': dict(checks.counts), 'issues': checks.issues, 'error': error}
    report.write(output / 'summary.json', result)
    report.seal(output)
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
