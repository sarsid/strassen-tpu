"""Local handoff guards and owned-process integration; no TPU/network access.

Run through the scoped archive runner. Signals outside the one explicitly owned
subprocess fixture are mocked. No production cohort or production PID is read.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import prepare_mlsys_actual_recovery_v001 as recovery_builder
import prepare_mlsys_campaign_v001 as original_builder
import supervise_mlsys_handoff_v001 as supervisor


def identity(pid=12345, state='S', command='fixture-controller'):
    return dict(pid=pid, started='Mon Sep 21 12:00:00 2026', state=state, command=command)


def original_plan(original):
    return original_builder.build_plan(original, 'fixture-session', 'tpu-v5e1-fixture',
        '/content/fixture/identity.json', sys.executable, sys.executable)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream)


def eventually(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.025)
    raise AssertionError('Owned subprocess fixture did not reach the expected state')


class HandoffGuards(unittest.TestCase):
    def test_identity_and_signal_reject_pid_start_or_command_changes(self):
        expected = identity()
        with patch.object(supervisor, 'process', return_value=expected), patch.object(supervisor.os, 'kill') as kill:
            supervisor.checked_signal(expected, signal.SIGSTOP)
            kill.assert_called_once_with(expected['pid'], signal.SIGSTOP)
        for observed in (None, dict(expected, pid=999), dict(expected, started='new start'),
                         dict(expected, command='other-controller')):
            with self.subTest(observed=observed), patch.object(supervisor, 'process', return_value=observed), \
                    patch.object(supervisor.os, 'kill') as kill:
                self.assertFalse(supervisor.matches(observed, expected))
                with self.assertRaises(RuntimeError):
                    supervisor.checked_signal(expected, signal.SIGKILL)
                kill.assert_not_called()

    def test_inactive_accepts_missing_zombie_and_reused_pid_but_not_stopped_parent(self):
        expected = identity()
        cases = [(None, True), (dict(expected, state='Z', command='<defunct>'), True),
                 (dict(expected, started='new start', command='unrelated'), True),
                 (dict(expected, state='T'), False), (dict(expected, state='S'), False),
                 (dict(expected, command='changed without a new start'), False)]
        for observed, answer in cases:
            with self.subTest(observed=observed), patch.object(supervisor, 'process', return_value=observed):
                self.assertEqual(supervisor.inactive(expected), answer)
        zombie = dict(expected, state='Z', command='<defunct>')
        self.assertTrue(supervisor.matches(zombie, expected, zombie=True))
        self.assertFalse(supervisor.matches(zombie, expected))

    def test_ps_parser_preserves_full_command_and_rejects_bad_output(self):
        completed = subprocess.CompletedProcess([], 0, 'Mon Sep 21 12:00:00 2026 T /fixture/python app.py --cohort /a path\n', '')
        with patch.object(supervisor.subprocess, 'run', return_value=completed) as run:
            value = supervisor.process(12345)
            self.assertEqual(value, identity(state='T', command='/fixture/python app.py --cohort /a path'))
            self.assertNotIn('shell', run.call_args.kwargs)
        for code, text, absent in ((1, '', True), (0, 'malformed', False), (2, '', False)):
            with self.subTest(code=code), patch.object(supervisor.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], code, text, 'fixture error')):
                if absent:
                    self.assertIsNone(supervisor.process(12345))
                else:
                    with self.assertRaises(RuntimeError):
                        supervisor.process(12345)

    def test_watchdog_only_resumes_verified_parent_before_takeover_fence(self):
        parent = identity(); watcher = identity(45678, command='fixture-watchdog-owner')
        with tempfile.TemporaryDirectory(prefix='mlsys-handoff-watchdog-') as tmp:
            recovery = Path(tmp).resolve()
            for state in ('observing', 'pause_pending', 'paused', 'takeover_intent', 'retired',
                          'recovery_started', 'completed', 'failed'):
                def fake_read(path):
                    if Path(path).name == 'handoff-verification.json':
                        raise FileNotFoundError('Fixture deliberately lacks proven idle evidence')
                    return {'handoff-plan.json': {'parent': parent, 'original': '/fixture/original'},
                            'handoff-supervisor.json': watcher, 'handoff-state.json': {'state': state}}[Path(path).name]
                with self.subTest(state=state), patch.object(supervisor.ctl, 'read', side_effect=fake_read), \
                        patch.object(supervisor.ctl, 'write') as write, patch.object(supervisor.ctl, 'atomic'), \
                        patch.object(supervisor.ctl, 'commit'), patch.object(supervisor, 'combined', return_value={}), \
                        patch.object(supervisor, 'inactive', return_value=True), \
                        patch.object(supervisor, 'process', return_value=parent), patch.object(supervisor, 'checked_signal') as send, \
                        patch.object(supervisor, 'start_recovery_once') as start:
                    self.assertEqual(supervisor.watchdog(recovery), 0)
                    self.assertEqual(write.call_args.args[1]['resumed'], state in supervisor.PRE_TRANSFER)
                    if state in supervisor.PRE_TRANSFER:
                        send.assert_called_once_with(parent, signal.SIGCONT)
                    else:
                        send.assert_not_called()
                    if state in ('takeover_intent', 'retired', 'recovery_started'):
                        self.assertIn('error', write.call_args.args[1])
                    start.assert_not_called()

    def test_watchdog_does_not_resume_reused_parent_or_signal_after_completion(self):
        parent = identity(); watcher = identity(45678)
        def fake_read(path):
            return {'handoff-plan.json': {'parent': parent}, 'handoff-supervisor.json': watcher,
                    'handoff-state.json': {'state': 'paused'}}[Path(path).name]
        with patch.object(supervisor.ctl, 'read', side_effect=fake_read), patch.object(supervisor.ctl, 'write') as write, \
                patch.object(supervisor.ctl, 'commit'), \
                patch.object(supervisor, 'inactive', return_value=True), \
                patch.object(supervisor, 'process', return_value=dict(parent, started='reused')), \
                patch.object(supervisor, 'checked_signal') as send:
            supervisor.watchdog(Path('/fixture/recovery'))
            send.assert_not_called(); self.assertFalse(write.call_args.args[1]['resumed'])
        def completed_read(path):
            return {'handoff-plan.json': {'parent': parent}, 'handoff-supervisor.json': watcher,
                    'handoff-state.json': {'state': 'completed'}}[Path(path).name]
        with patch.object(supervisor.ctl, 'read', side_effect=completed_read), patch.object(supervisor.ctl, 'write') as write, \
                patch.object(supervisor.ctl, 'commit'), \
                patch.object(supervisor, 'inactive', return_value=False), patch.object(supervisor, 'checked_signal') as send, \
                patch.object(supervisor.time, 'sleep', side_effect=AssertionError('Unexpected watchdog wait')):
            self.assertEqual(supervisor.watchdog(Path('/fixture/recovery')), 0)
            write.assert_not_called(); send.assert_not_called()

    def test_existing_or_ambiguous_recovery_start_never_launches_duplicate(self):
        with tempfile.TemporaryDirectory(prefix='mlsys-handoff-launch-') as tmp:
            recovery = Path(tmp).resolve()
            plan = {'allocation_id': 'tpu-v5e1-fixture', 'controller_python': sys.executable}
            save(recovery/'plan.json', plan)
            expected_command = str(recovery/'source/tools/run_mlsys_unattended_v001.py')+' run --cohort '+str(recovery)
            running = identity(command=expected_command)
            save(recovery/'controller-started.json', {'pid': running['pid']})
            with patch.object(supervisor, 'process', return_value=running), patch.object(supervisor.subprocess, 'Popen') as spawn:
                self.assertEqual(supervisor.start_recovery_once(recovery), running)
                spawn.assert_not_called()
            with patch.object(supervisor, 'process', return_value=None), patch.object(supervisor.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'no automatic duplicate'):
                    supervisor.start_recovery_once(recovery)
                spawn.assert_not_called()
            (recovery/'controller-started.json').unlink()
            save(recovery/'recovery-launch-intent.json', {'allocation_id': plan['allocation_id']})
            with patch.object(supervisor.time, 'sleep'), patch.object(supervisor.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'no receipt'):
                    supervisor.start_recovery_once(recovery)
                spawn.assert_not_called()

    def test_recovery_plan_preserves_allocation_protocol_and_original(self):
        original = Path('/fixture/runs/original'); new = Path('/fixture/runs/recovery')
        old = original_plan(original); before = deepcopy(old)
        link = {'cohort_id': original.name, 'cohort_absolute_path': str(original)}
        plan = recovery_builder.build_plan(new, old, link)
        self.assertEqual(old, before)
        self.assertEqual(len(plan['stages']), 11)
        self.assertEqual([s['id'] for s in plan['stages'][:3]],
                         ['original-final-audit', 'original-report', 'MODEL-tools-v002'])
        for field in ('allocation_id', 'session', 'remote_identity', 'private_root'):
            self.assertEqual(plan[field], old[field])
        self.assertEqual(plan['recovery_of']['cohort_absolute_path'], str(original))
        first, report, tools = plan['stages'][:3]
        self.assertEqual(first['command'][-4:], ['--cohort', str(original), '--stage', 'MAIN-12-confirm'])
        self.assertFalse(first.get('continue_on_failure', False))
        self.assertEqual(report['command'][-1], '{cohort}/operations/original-report/artifacts')
        self.assertEqual(tools['requires'], ['original-final-audit', 'original-report'])
        original_stages = {s['id']: s for s in old['stages']}
        for stage in plan['stages']:
            if stage.get('kind') == 'preparation' and stage.get('operation') == 'phase':
                self.assertEqual(stage['module'], recovery_builder.PREPARER)
                args = stage['runner_args']
                self.assertEqual(args[args.index('--venv-dir') + 1], old['private_root'] + '/cpu-tools-v002')
                self.assertTrue(stage['continue_on_failure'])
                if stage['id'].endswith('-inputs'):
                    self.assertEqual(stage['requires'], ['MODEL-tools-v002'])
            if stage['id'].endswith('-actual'):
                self.assertEqual(stage, original_stages[stage['id']])
        self.assertEqual(sum(s.get('releases_allocation') is True for s in plan['stages']), 1)
        with self.assertRaises(ValueError):
            recovery_builder.build_plan(original, old, link)
        with self.assertRaises(ValueError):
            recovery_builder.build_plan(new, dict(old, allocation_id='tpu-v6e1-fixture'), link)

    def test_combined_feed_keeps_old_results_queues_models_and_has_one_release(self):
        with tempfile.TemporaryDirectory(prefix='mlsys-handoff-feed-') as tmp:
            root = Path(tmp).resolve(); original = root/'runs/original'; recovery = root/'runs/recovery'
            old = original_plan(original)
            plan = recovery_builder.build_plan(recovery, old, {'cohort_absolute_path': str(original)})
            old_result = {'stage': 'MAIN-11-confirm', 'shape_id': 'fixture', 'method': 'cubic', 'mean_ms': 1.0}
            old_state = dict(state='running', stage='MAIN-12-screen', stages=[
                {'id': s['id'], 'state': 'not_started'} for s in old['stages']], selected_results=[old_result])
            save(original/'progress.json', old_state); save(recovery/'plan.json', plan)
            save(recovery/'progress.json', {'state': 'preparing', 'stages': [], 'selected_results': []})
            with patch.object(supervisor, 'ROOT', root):
                queued = supervisor.combined(original, recovery, {'state': 'observing'})
                self.assertEqual(queued['selected_results'], [old_result])
                self.assertEqual(sum(s['id'].endswith('/release') for s in queued['stages']), 1)
                self.assertNotIn('release', [s['id'] for s in queued['stages']])
                for model in ('qwen', 'mistral', 'gemma'):
                    stage = next(s for s in queued['stages'] if s['id'] == 'recovery/'+model+'-actual')
                    self.assertEqual(stage['state'], 'not_started')
                    self.assertEqual(stage['completed'], 0)
                    self.assertIsNone(stage['expected'])
                current = dict(state='running', stage='qwen-actual', selected_results=[{'stage': 'qwen-actual'}],
                               stages=[dict(s, state='succeeded' if s['id'] == 'original-final-audit' else 'not_started')
                                       for s in plan['stages']])
                with patch.object(supervisor.ctl, 'read', side_effect=lambda path:
                        current if Path(path) == recovery/'progress.json' else json.loads(Path(path).read_text())):
                    active = supervisor.combined(original, recovery, {'state': 'recovery_started'})
                self.assertEqual(active['selected_results'], [old_result, {'stage': 'qwen-actual'}])
                self.assertEqual(next(s for s in active['stages'] if s['id'] == supervisor.FINAL+'-audit')['state'], 'succeeded')
                self.assertEqual(sum(s['id'].endswith('/release') for s in active['stages']), 1)
            self.assertEqual(json.loads((original/'progress.json').read_text()), old_state)

    def test_terminal_cleanup_markers_forbid_handoff(self):
        with tempfile.TemporaryDirectory(prefix='mlsys-handoff-markers-') as tmp:
            root = Path(tmp)
            self.assertFalse(supervisor.forbidden(root))
            for name in ('operations/release', 'failure-cleanup', 'controller-completion.json'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
                self.assertTrue(supervisor.forbidden(root)); path.unlink()

    @unittest.skipUnless(os.name == 'posix' and Path('/bin/ps').exists(), 'POSIX process-control fixture')
    def test_owned_parent_pause_does_not_pause_independent_child_and_continue_resumes(self):
        ticker = "import sys,time\nfrom pathlib import Path\np=Path(sys.argv[1])\nfor n in range(1500):\n with p.open('a') as f:f.write(str(n)+'\\n')\n time.sleep(.02)\n"
        parent_code = """import json,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]); alive=True
def stop(signum,frame):
 global alive
 alive=False
signal.signal(signal.SIGTERM,stop)
child=subprocess.Popen([sys.executable,'-c',sys.argv[2],str(root/'child.txt')],start_new_session=True)
(root/'child.json').write_text(json.dumps({'pid':child.pid}))
try:
 n=0
 while alive and n<1500:
  with (root/'parent.txt').open('a') as f:f.write(str(n)+'\\n')
  n+=1;time.sleep(.02)
finally:
 child.terminate()
 try:child.wait(timeout=3)
 except subprocess.TimeoutExpired:child.kill();child.wait()
"""
        with tempfile.TemporaryDirectory(prefix='mlsys-handoff-owned-') as tmp:
            root = Path(tmp).resolve(); child_identity = None; parent_identity = None
            parent = subprocess.Popen([sys.executable, '-c', parent_code, str(root), ticker],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            def size(name):
                path = root/name
                return path.stat().st_size if path.exists() else 0
            try:
                eventually(lambda: size('child.txt') > 0 and size('parent.txt') > 0)
                parent_identity = supervisor.process(parent.pid)
                child_identity = supervisor.process(json.loads((root/'child.json').read_text())['pid'])
                self.assertIsNotNone(parent_identity); self.assertIsNotNone(child_identity)
                supervisor.checked_signal(parent_identity, signal.SIGSTOP)
                eventually(lambda: 'T' in supervisor.process(parent.pid)['state'])
                stopped_size = size('parent.txt'); child_size = size('child.txt')
                eventually(lambda: size('child.txt') > child_size + 5)
                self.assertEqual(size('parent.txt'), stopped_size)
                self.assertFalse(supervisor.inactive(parent_identity))
                supervisor.checked_signal(parent_identity, signal.SIGCONT)
                eventually(lambda: size('parent.txt') > stopped_size)
            finally:
                if parent.poll() is None:
                    # This Popen handle belongs exclusively to this test.
                    os.kill(parent.pid, signal.SIGCONT); parent.terminate()
                    try:parent.wait(timeout=5)
                    except subprocess.TimeoutExpired:parent.kill(); parent.wait()
                if child_identity and not supervisor.inactive(child_identity):
                    supervisor.checked_signal(child_identity, signal.SIGTERM)
                    eventually(lambda: supervisor.inactive(child_identity))


def main():
    out = Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'; out.mkdir(exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HandoffGuards)
    with (out/'unittest.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print((out/'unittest.log').read_text())
    summary = dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        failures=[{'test': str(test), 'traceback': text} for test, text in result.failures],
        errors=[{'test': str(test), 'traceback': text} for test, text in result.errors],
        skipped=[{'test': str(test), 'reason': reason} for test, reason in result.skipped],
        network_used=False, remote_operations=False, production_processes_signaled=False,
        limitation='Local guard/plan/feed checks and one owned-process pause/resume integration; no live campaign handoff or TPU continuation executed.')
    save(out/'summary.json', summary)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
