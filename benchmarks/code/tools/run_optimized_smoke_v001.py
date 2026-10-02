"""Run the small frozen ablation on its own bounded allocation, then release it.

Requires a receipt from allocate_bounded_optimized_v001.py. This controller
does not allocate, interrupt, share, or release any other task's TPU. Existing
archive/launch/retrieval logic is reused with commits scoped to owned paths.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import signal
import sys
import tarfile
import time
import uuid
import archive_v002 as archive
import setup_allocation_v001 as setup
import run_phase_v004 as phase_runner

ROOT = Path(__file__).resolve().parents[1]
OWNED = [
    'configs/strassen_optimized_hardware_smoke_v001.json',
    'runtime/allocate_bounded_optimized_v001.py',
    'src/strassen_mm/benchmark_optimized_hardware_v001.py',
    'tests/test_optimized_hardware_adapter_v001.py',
    'tools/run_optimized_smoke_v001.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    parser.add_argument('--controller-python', required=True)
    args = parser.parse_args()
    receipt_dir = args.receipt_dir.resolve()
    receipt = json.loads((receipt_dir / 'allocation.json').read_text())
    session, endpoint = receipt['session'], receipt['endpoint']
    if endpoint == receipt['coexisting_endpoint_untouched'] or not receipt['created']:
        raise ValueError('Expected a newly owned separate allocation')
    receipt_relative = str(receipt_dir.relative_to(ROOT))
    owned = OWNED + [receipt_relative]
    current = None
    phases = []
    frozen_source_commit = None

    def commit(message):
        paths = owned + ([str(current.relative_to(ROOT))] if current else [])
        subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
        if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=ROOT).returncode:
            subprocess.run(['git', 'commit', '--only', '-m', message, '--', *paths], cwd=ROOT, check=True)
        return archive.git('rev-parse', 'HEAD')

    def begin(label, command=None, scope='local_validation'):
        nonlocal current, frozen_source_commit
        current = None
        revision = commit('Freeze source before ' + label)
        frozen_source_commit = frozen_source_commit or revision
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + label + '-' + uuid.uuid4().hex[:6]
        current = ROOT / 'runs' / run_id
        current.mkdir(exist_ok=False)
        source = current / 'source'
        source.mkdir()
        roots = [p for p in archive.git('ls-tree', '--name-only', frozen_source_commit).splitlines()
                 if p not in ('runs', 'status')]
        package = current / 'source.tar'
        with package.open('xb') as stream:
            subprocess.run(['git', 'archive', frozen_source_commit, '--', *roots], cwd=ROOT, stdout=stream, check=True)
        with tarfile.open(package) as packed:
            packed.extractall(source, filter='data')
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source.rglob('*')) if p.is_file()}
        archive.exclusive_json(current / 'source-manifest.json', {'source_commit': frozen_source_commit,
              'source_sha256': hashes, 'archive_sha256': hashlib.sha256(package.read_bytes()).hexdigest()})
        archive.exclusive_json(current / 'execution.json', {'run_id': run_id, 'label': label,
              'created_utc': archive.utc(), 'source_commit': frozen_source_commit,
              'scope': scope, 'command': command,
              'source_policy': 'Same frozen source commit for setup, screen and confirmation despite concurrent unrelated commits'})
        phases.append(current)
        return current

    archive.commit, archive.begin = commit, begin
    result = {'allocation_id': endpoint, 'session': session, 'completed': False, 'phases': []}
    def deadline_alarm(signum, frame):
        raise TimeoutError('Campaign monitoring stopped early to reserve allocation-release time')
    prior_handler = signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1, receipt['deadline_unix'] - time.time() - 360))
    try:
        if receipt['deadline_unix'] - time.time() < 600:
            raise RuntimeError('Insufficient bounded allocation time for setup and benchmark')
        sys.argv = ['setup_allocation_v001.py', '--session', session, '--endpoint', endpoint,
                    '--controller-python', args.controller_python]
        with (receipt_dir / 'setup-controller.log').open('x') as log, contextlib.redirect_stdout(log):
            code = setup.main()
        if code:
            raise RuntimeError('TPU setup failed; preserved setup archive identifies the cause')
        setup_run = current
        identity = f'/content/Strassen_MM_Focus/{setup_run.name}/identity.json'
        screen_run = None
        for phase in ('SOPT-screen', 'SOPT-confirm'):
            remaining = receipt['deadline_unix'] - time.time()
            # Existing launcher reserves 300s inside its worker timeout.
            # 900s therefore provides at most 600s for actual benchmarking.
            timeout = min(900, int(remaining - 420))
            if timeout < 600:
                result['stopped_reason'] = 'Remaining budget reserved for retrieval and release'
                break
            argv = ['run_phase_v004.py', '--phase', phase, '--session', session, '--endpoint', endpoint,
                    '--expected-identity', identity, '--controller-python', args.controller_python,
                    '--campaign-relative', 'configs/strassen_optimized_hardware_smoke_v001.json',
                    '--benchmark-module', 'strassen_mm.benchmark_optimized_hardware_v001',
                    '--timeout-seconds', str(timeout), '--archive-grace-seconds', '120']
            if screen_run:
                argv.append('--runner-arg=--prior-run=/content/Strassen_MM_Focus/runs/' + screen_run.name + '/artifacts')
            sys.argv = argv
            print(json.dumps({'phase': phase, 'timeout_seconds': timeout, 'allocation_id': endpoint}), flush=True)
            with (receipt_dir / (phase + '-controller.log')).open('x') as log, contextlib.redirect_stdout(log):
                code = phase_runner.main()
            result['phases'].append({'phase': phase, 'run': str(current.relative_to(ROOT)), 'exit_code': code})
            if code:
                raise RuntimeError(phase + ' did not complete; inspect the archived failure')
            if phase == 'SOPT-screen':
                screen_run = current
                identity = f'/content/Strassen_MM_Focus/runs/{screen_run.name}/artifacts/environment.json'
            else:
                result['completed'] = True
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        try:
            with (receipt_dir / 'release.log').open('x') as log:
                release = subprocess.run([args.controller_python, str(ROOT / 'runtime/release_allocation_v001.py'),
                           '--session', session, '--expect-endpoint', endpoint], stdout=log, stderr=subprocess.STDOUT,
                           timeout=150)
            result['release_exit_code'] = release.returncode
        except Exception as error:
            result['release_exit_code'] = None
            result['release_error'] = {'type': type(error).__name__, 'message': str(error)}
            result['watchdog_remains_responsible_for_budget_release'] = True
        result['all_archive_paths'] = [str(p.relative_to(ROOT)) for p in phases]
        archive.exclusive_json(receipt_dir / 'campaign-completion.json', result)
        current = None
        result['commit'] = commit('Archive bounded Strassen optimization hardware campaign lifecycle')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['completed'] and result['release_exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
