"""Persistently supervise v6e then historical v5e, reconciling failures automatically.

No allocation occurs while the provider's state is unknown. Existing workers
are inspected before replacement. Only whole verified comparisons are reused.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from arch_resilience_v001 import BASE, RESULTS, ROOT, collect, units, check_contract
from run_large_real_v002 import atomic, commit, read, utc, write, headline_results
from run_phase_v006 import remote_object
from run_region_cohort_v001 import sha


class RetryLater(RuntimeError):
    pass


class Supervisor:
    def __init__(self, folder):
        self.folder = folder.resolve()
        self.source = Path(__file__).resolve().parents[1]
        self.controller = str(ROOT.parent / '.venv-colab/bin/python')
        self.analysis = str(ROOT.parent / '.venv-reconcile/bin/python')
        self.statefile = self.folder / 'supervisor-state.json'
        self.historyfile = self.folder / 'history.json'
        self.state = read(self.statefile)
        self.history = read(self.historyfile)
        self.child = None
        self.baseline = []
        self.baseline_results = []
        self.complete = {}

    def save(self):
        atomic(self.statefile, self.state)
        atomic(self.historyfile, self.history)

    def event(self, name, **details):
        with (self.folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=utc(), event=name, **details)) + '\n')

    def publish(self, detail=None, child=None):
        hardware = self.state['hardware']
        value = dict(child or {})
        additional = sum(s.get('state') == 'succeeded' and s['id'].endswith('-confirm')
            and s['id'].removesuffix('-confirm') not in self.complete for s in value.get('stages', []))
        value.update(schema_version=1, campaign_id=self.folder.name,
            title=f'{hardware}: automatic recovery; {len(self.complete)+additional}/{len(units(hardware))} comparison units saved',
            cohort_path=str(self.folder.relative_to(ROOT)), heartbeat_utc=utc(), updated_utc=utc(),
            state='running' if not self.state.get('finished') else 'succeeded',
            stage=value.get('stage') or self.state.get('action', 'supervising'),
            detail=detail or value.get('detail') or self.state.get('detail', ''),
            next_step='Recover routine failures automatically; preserve whole comparisons and allocation identities.',
            stages=self.baseline + value.get('stages', []),
            selected_results=self.baseline_results + value.get('selected_results', []))
        atomic(self.folder / 'progress.json', value)

    def wait(self, seconds, detail):
        self.state.update(action='automatic-recovery', detail=detail)
        self.save()
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.publish(detail)
            time.sleep(min(15, max(0, until - time.monotonic())))

    def call(self, label, command, timeout=300):
        operation = self.folder / 'operations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + label + '-' + uuid.uuid4().hex[:6])
        operation.mkdir(parents=True)
        write(operation / 'execution.json', dict(command=command, started_utc=utc()))
        try:
            with (operation / 'execution.log').open('x') as log:
                result = subprocess.run(command, cwd=self.source, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, env=dict(os.environ, STRASSEN_PROJECT_ROOT=str(ROOT),
                    PYTHONPATH=str(self.source / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
            code, error = result.returncode, None
        except subprocess.TimeoutExpired:
            code, error = -1, 'TimeoutExpired; inspect existing remote state before any retry'
        write(operation / 'completion.json', dict(exit_code=code, error=error, finished_utc=utc()))
        return code, (operation / 'execution.log').read_text(), operation

    def control(self, hardware, *args, timeout=300):
        version = 'v003' if hardware == 'v6e' else 'v002'
        return self.call('control-' + args[0], [self.controller,
            str(self.source / ('runtime/colab_control_' + version + '.py')), *args], timeout)

    def assignments(self):
        code, output, _ = self.control('v6e', 'list', timeout=90)
        if code:
            raise RetryLater('Colab allocation status is unavailable; waiting without allocating another TPU.')
        return next(json.loads(line)['assignments'] for line in output.splitlines()
                    if json.loads(line).get('kind') == 'allocations')

    def ensure_runtime(self):
        hardware = self.state['hardware']
        assignments = self.assignments()
        runtime = self.state.get('runtime')
        if runtime and any(a['endpoint'] == runtime['endpoint'] for a in assignments):
            return runtime
        if runtime:
            self.event('allocation_absent', runtime=runtime)
            self.state['runtime'] = None
            self.save()
        request = self.state.get('pending_request')
        if assignments:
            if not request or len(assignments) != 1:
                raise RetryLater('An unrelated or ambiguous allocation is present; waiting without creating another.')
            code, output, _ = self.call('adopt-late', [self.controller,
                str(self.source / 'runtime/adopt_supervisor_assignment_v001.py'),
                '--request', request['path'], '--expect-endpoint', assignments[0]['endpoint']])
            if code:
                raise RetryLater('Late allocation could not be attributed safely; waiting and retaining evidence.')
            allocation = next(json.loads(line) for line in output.splitlines() if json.loads(line).get('kind') == 'allocation')
        else:
            if request and time.time() - request['epoch'] < 600:
                raise RetryLater('Allocation request response was uncertain; allowing provisioning to settle before retrying.')
            recent = [t for t in self.state.get('allocation_requests', []) if time.time() - t < 3600]
            if len(recent) >= 4:
                raise RetryLater('Runtime replacement rate limit reached; automatically waiting before the next attempt.')
            session = 'strassen-auto-' + hardware + '-' + datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
            requestfile = self.folder / ('allocation-request-' + session + '.json')
            write(requestfile, dict(session=session, hardware=hardware, before_assignments=[], utc=utc()))
            self.state.update(pending_request=dict(path=str(requestfile), epoch=time.time(), session=session),
                              allocation_requests=recent + [time.time()])
            self.save()
            code, output, _ = self.call('allocate', [self.controller, str(self.source / 'runtime/allocate_tradeoff_v002.py'),
                hardware, 'create', '--session', session], timeout=600)
            if code:
                raise RetryLater('Colab did not acknowledge allocation; automatically checking for a late assignment.')
            allocation = next(json.loads(line) for line in output.splitlines() if json.loads(line).get('kind') == 'allocation')
        runtime = dict(hardware=hardware, session=allocation['session'], endpoint=allocation['endpoint'], identity=None)
        self.state.update(runtime=runtime, pending_request=None)
        self.save()
        self.event('allocation_owned', runtime=runtime)
        return runtime

    def release(self):
        runtime = self.state.get('runtime')
        if not runtime:
            return
        if any(a['endpoint'] == runtime['endpoint'] for a in self.assignments()):
            version = 'v002' if runtime['hardware'] == 'v6e' else 'v001'
            code, _, _ = self.call('release', [self.controller, str(self.source / ('runtime/release_allocation_' + version + '.py')),
                '--session', runtime['session'], '--expect-endpoint', runtime['endpoint']], 180)
            if code and any(a['endpoint'] == runtime['endpoint'] for a in self.assignments()):
                raise RetryLater('Waiting to verify release of the owned runtime before replacement.')
        self.event('allocation_released_or_absent', runtime=runtime)
        self.state['runtime'] = None
        self.state['runtime_needs_release'] = False
        self.save()

    def setup(self, runtime):
        if runtime.get('identity'):
            return
        # archive_scoped locates its repository from its own path; check the
        # root copies match this supervisor's committed source before using it.
        for name in ('tools/archive_scoped_v001.py', 'tools/setup_allocation_v005.py'):
            if (ROOT / name).read_bytes() != (self.source / name).read_bytes():
                raise ValueError('Frozen setup source changed: ' + name)
        self.publish('Preparing the replacement TPU with the original architecture-specific software pins.')
        code, output, _ = self.call('setup', [self.analysis, str(ROOT / 'tools/archive_scoped_v001.py'),
            '--label', 'auto-' + runtime['hardware'] + '-setup-v001', '--source', 'runtime',
            '--source', 'tools/setup_allocation_v005.py', '--timeout-seconds', '3300', '--',
            self.controller, 'tools/setup_allocation_v005.py', '--session', runtime['session'],
            '--endpoint', runtime['endpoint'], '--hardware', runtime['hardware'], '--controller-python', self.controller], 3400)
        found = [line.split('=', 1)[1] for line in output.splitlines() if line.startswith('EXECUTION_DIR=')]
        if not code and found and read(Path(found[-1]) / 'artifacts/summary.json')['completed']:
            runtime['identity'] = read(Path(found[-1]) / 'artifacts/summary.json')['remote_identity']
            runtime['setup_run'] = found[-1]
            self.save()
            return
        self.state['runtime_needs_release'] = True
        self.save()
        self.release()
        raise RetryLater('Runtime setup did not complete; evidence retained and a fresh setup will be attempted automatically.')

    def refresh_completed(self):
        self.complete = collect(self.history, self.state['hardware'])
        self.baseline = []
        self.baseline_results = []
        for pair in self.complete.values():
            for item in pair.values():
                # A controller can fail in the export step after successfully
                # committing the phase. Repair that dedicated-folder export.
                if Path(item['evidence']).name != 'reconciliation.json':
                    cohort = Path(item['cohort'])
                    exported = ROOT / RESULTS[self.state['hardware']] / 'phases' / cohort.name / (item['phase'] + '.json')
                    if not exported.exists():
                        code, _, _ = self.call('repair-export', [self.analysis,
                            str(self.source / 'tools/export_arch_study_v001.py'), '--cohort', str(cohort), '--phase', item['phase']], 1800)
                        if code:
                            raise RetryLater('Saved measurements need their dedicated result export repaired; retrying automatically.')
                summary = read(Path(item['run']) / 'artifacts/summary.json')
                counts = summary['case_status_counts']
                total = sum(counts.values())
                self.baseline.append(dict(id='saved-' + item['phase'], label='Saved ' + item['phase'],
                    kind='measurement', state='succeeded', expected=total, completed=total,
                    succeeded=counts.get('ok', 0), failed=total-counts.get('ok', 0), measured=counts.get('ok', 0),
                    detail='Verified complete comparison; preserved without rerunning.', evidence=[]))
                if item['phase'].endswith('-confirm'):
                    rows = [json.loads(line) for line in (Path(item['run']) / 'artifacts/results.jsonl').read_text().splitlines()]
                    self.baseline_results.extend(headline_results(rows, item['phase'], Path(item['run'])))

    def launch(self, runtime):
        cohort = ROOT / 'runs' / ('auto-' + runtime['hardware'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:4])
        code, _, _ = self.call('freeze', [self.analysis, str(self.source / 'tools/prepare_arch_resilient_v001.py'),
            'freeze', '--cohort', str(cohort), '--history', str(self.historyfile), '--hardware', runtime['hardware'],
            '--session', runtime['session'], '--endpoint', runtime['endpoint'], '--identity', runtime['identity'],
            '--controller-python', self.controller, '--analysis-python', self.analysis], 900)
        if code:
            raise RetryLater('Continuation freeze failed; retrying only after preserving its diagnostics.')
        self.history['cohorts'].append(dict(path=str(cohort), hardware=runtime['hardware']))
        self.state.update(current_cohort=str(cohort), action='measuring')
        self.save()
        command = [self.controller, str(cohort / 'source/tools/run_large_real_v003.py'), 'run', '--cohort', str(cohort)]
        with (cohort / 'controller.log').open('x') as log:
            self.child = subprocess.Popen(command, cwd=cohort / 'source', stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, env=dict(os.environ, STRASSEN_PROJECT_ROOT=str(ROOT),
                PYTHONPATH=str(cohort / 'source/src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
        write(cohort / 'controller-process.json', dict(pid=self.child.pid, command=command, supervised_by=str(self.folder), utc=utc()))
        self.event('controller_started', cohort=str(cohort), pid=self.child.pid)
        commit([self.historyfile, self.statefile, cohort / 'controller-process.json'], 'Checkpoint automatic architecture continuation')

    def monitor(self, cohort):
        heartbeat = 0
        while not (cohort / 'controller-completion.json').exists():
            if (cohort / 'progress.json').exists():
                self.publish(child=read(cohort / 'progress.json'))
            process = read(cohort / 'controller-process.json')
            if self.child is not None and self.child.pid == process['pid'] and self.child.poll() is not None:
                return
            try:
                os.kill(process['pid'], 0)
            except ProcessLookupError:
                self.event('controller_exited_without_completion', cohort=str(cohort))
                return
            command = subprocess.run(['ps', '-p', str(process['pid']), '-o', 'state=,command='], capture_output=True, text=True).stdout.strip()
            if not command or command.startswith('Z') or str(cohort / 'source/tools/run_large_real_v003.py') not in command:
                self.event('controller_process_no_longer_matches', cohort=str(cohort), pid=process['pid'])
                return
            if time.monotonic() - heartbeat > 300:
                runtime = self.state['runtime']
                # Existing-session create only renews a dead keep-alive process.
                self.control(runtime['hardware'], 'create', '--session', runtime['session'], '--expect-endpoint', runtime['endpoint'], timeout=90)
                heartbeat = time.monotonic()
            time.sleep(15)

    def reconcile(self, cohort):
        progress = read(cohort / 'progress.json')
        stage = progress.get('stage')
        receipt = cohort / (str(stage) + '-started.json')
        runtime = self.state['runtime']
        assignments = self.assignments()
        if not any(a['endpoint'] == runtime['endpoint'] for a in assignments):
            self.event('lost_runtime', cohort=str(cohort), stage=stage, endpoint=runtime['endpoint'])
            self.state['runtime'] = None
            self.save()
            return
        if not receipt.exists():
            # A local command failed between archived phases; no new worker.
            return
        run = Path(read(receipt)['run'])
        code, output, _ = self.control(runtime['hardware'], 'exec-file', '--session', runtime['session'],
            '--expect-endpoint', runtime['endpoint'], '--file', str(self.source / 'runtime/probe_supervisor_v001.py'),
            '--timeout', '60', '--script-arg=--run-id', '--script-arg=' + run.name, timeout=90)
        if code:
            raise RetryLater('Remote worker state is uncertain; checking again without relaunching it.')
        probe = remote_object(output, 'supervisor_probe')
        if not probe['execution_lock_free']:
            raise RetryLater('The existing remote worker is still running; waiting for it instead of duplicating work.')
        ready = probe.get('archive_ready')
        local_done = read(run / 'completion.json') if (run / 'completion.json').exists() else {}
        if ready and ready['status'] == 'completed' and local_done.get('status') != 'completed':
            recovery = self.folder / 'recoveries' / (run.name + '-' + uuid.uuid4().hex[:6])
            recovery.mkdir(parents=True)
            code, _, _ = self.control(runtime['hardware'], 'download', '--session', runtime['session'],
                '--expect-endpoint', runtime['endpoint'], '--remote', ready['archive'],
                '--local', str(recovery / 'remote-run.tar.gz'), timeout=300)
            if code:
                raise RetryLater('Completed archive download interrupted; automatically retrying retrieval.')
            code, _, _ = self.call('recover-archive', [self.analysis, str(self.source / 'tools/reconcile_arch_transport_v001.py'),
                '--folder', str(recovery), '--prior-cohort', str(cohort), '--phase', stage,
                '--archive-sha256', ready['sha256']], 180)
            if code:
                raise ValueError('Recovered archive failed verification; inspect preserved evidence')
            self.history.setdefault('recoveries', {})[str(cohort) + ':' + stage] = str(recovery)
            self.save()
            destination = ROOT / RESULTS[runtime['hardware']] / 'recovery' / recovery.name
            destination.mkdir(parents=True, exist_ok=False)
            for name in ('remote-run.tar.gz', 'reconciliation.json', 'verified.json'):
                shutil.copyfile(recovery / name, destination / name)
            commit([recovery, destination, self.historyfile], 'Recover completed architecture phase after transport failure')
        self.event('worker_reconciled_idle', cohort=str(cohort), stage=stage, recovered=bool(ready))

    def finish_hardware(self):
        hardware = self.state['hardware']
        # All paired measurements and phase exports are verified before this
        # method runs. Release idle hardware before CPU-only final reporting.
        self.release()
        destination = ROOT / RESULTS[hardware]
        marker = destination / 'COMPLETE.json'
        if not marker.exists():
            out = self.folder / ('final-' + hardware + '-' + uuid.uuid4().hex[:6]) / 'artifacts'
            code, _, _ = self.call('report-' + hardware, [self.analysis, str(self.source / 'tools/report_arch_resilient_v001.py'),
                '--history', str(self.historyfile), '--hardware', hardware, '--output-dir', str(out)], 3600)
            if code or not read(out / 'results.json')['completed']:
                raise RetryLater('All measurements are saved; final reporting needs recovery, without new TPU measurements.')
            if (destination / 'report').exists():
                raise ValueError('Unsealed prior report exists; preserve and reconcile it before replacement')
            shutil.copytree(out, destination / 'report')
            write(marker, dict(completed=True, hardware=hardware, history=str(self.historyfile),
                results_sha256=sha(destination / 'report/results.json'), completed_units=list(self.complete), utc=utc()))
            commit([destination, out.parent], 'Finalize automatically recovered ' + hardware + ' architecture study')
        elif sha(destination / 'report/results.json') != read(marker)['results_sha256']:
            raise ValueError('Final result checksum mismatch')
        self.release()
        self.event('hardware_completed', hardware=hardware)
        self.state.setdefault('completed_hardware', []).append(hardware)
        self.state['current_cohort'] = None
        if hardware == 'v6e':
            self.state['hardware'] = 'v5e'
        else:
            self.state['finished'] = True
        self.save()

    def run(self):
        lock = (self.folder / 'supervisor.lock').open('ab')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_contract(self.source)
        guard = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        write(self.folder / ('process-' + str(os.getpid()) + '-' + uuid.uuid4().hex[:6] + '.json'), dict(pid=os.getpid(), sleep_guard_pid=guard.pid, source=str(self.source), utc=utc()))
        backoff = 30
        try:
            self.refresh_completed()
            while not self.state.get('finished'):
                try:
                    current = self.state.get('current_cohort')
                    if current:
                        cohort = Path(current)
                        self.monitor(cohort)
                        self.reconcile(cohort)
                        self.state['current_cohort'] = None
                        self.save()
                    self.refresh_completed()
                    self.publish('Verified saved comparisons; automatically continuing unfinished work.')
                    if len(self.complete) == len(units(self.state['hardware'])):
                        self.finish_hardware()
                        continue
                    if self.state.get('runtime_needs_release'):
                        self.release()
                    runtime = self.ensure_runtime()
                    self.setup(runtime)
                    self.launch(runtime)
                    backoff = 30
                except Exception as error:
                    self.event('automatic_retry', type=type(error).__name__, message=str(error), delay_seconds=backoff)
                    self.wait(backoff, str(error))
                    backoff = min(900, backoff * 2)
            self.publish('Both architecture studies are complete; results committed and TPU released.')
            write(self.folder / 'completion.json', dict(completed=True, utc=utc(), hardware=self.state['completed_hardware']))
            commit([self.folder / 'completion.json', self.statefile, self.historyfile], 'Complete automatically supervised architecture studies')
        finally:
            guard.terminate()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    Supervisor(args.folder).run()
