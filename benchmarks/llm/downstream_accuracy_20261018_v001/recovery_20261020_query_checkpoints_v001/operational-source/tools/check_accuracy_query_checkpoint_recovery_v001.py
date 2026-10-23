"""Verify repaired-source recovery, inherited archive safety and exact dispatch."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import resume_accuracy_query_remote_v001 as remote
from start_accuracy_study_v006 import observer_command, owned


def make_tar(path, files):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.source = self.base / 'source.tar.gz'
        self.worker_bytes = b'def worker(run, previous):\n    pass\n'
        make_tar(self.source, {remote.WORKER: self.worker_bytes})
        self.source_sha = remote.sha(self.source)
        q14 = self.base / 'q14.tar.gz'
        make_tar(q14, {'qwen3_14b/summary.json': b'{"complete":true}\n'})
        self.q14 = q14.read_bytes()
        self.contract = {'bundle_sha256': self.source_sha, 'depth': 1}
        self.contract_hash = remote.digest(self.contract)
        self.files = {
            'completed/qwen3_14b.tar.gz': self.q14,
            'artifacts/qwen3_32b/manifest.json': json.dumps({'execution_contract_sha256': self.contract_hash}).encode(),
            'artifacts/qwen3_32b/execution-contract.json': json.dumps(self.contract).encode(),
            'artifacts/qwen3_32b/batches/gsm8k-000.json': json.dumps({'execution_contract_sha256': self.contract_hash}).encode(),
            'artifacts/qwen3_32b/provenance/original-runtime-admission.json': b'{"status":"passed","all_logits_finite":true}',
        }
        self.seed = self.base / 'seed.tar.gz'
        for name, value in [('BASE', self.base), ('Q14_SHA', hashlib.sha256(self.q14).hexdigest()),
                            ('WORKER_SHA', hashlib.sha256(self.worker_bytes).hexdigest())]:
            context = patch.object(remote, name, value)
            context.start()
            self.addCleanup(context.stop)

    def seed_archive(self):
        receipt = dict(source_sha256=self.source_sha, saved_gsm_pairs=8,
            files={k: hashlib.sha256(v).hexdigest() for k, v in self.files.items()})
        make_tar(self.seed, dict(self.files, **{'seed-manifest.json': json.dumps(receipt).encode()}))
        return remote.sha(self.seed)

    def test_exact_restore_and_duplicate_refusal(self):
        seed_sha = self.seed_archive()
        run, environment = remote.prepare(self.source, self.source_sha, self.seed, seed_sha, 'recovery')
        for name, data in self.files.items():
            self.assertEqual((run / name).read_bytes(), data)
        self.assertEqual((run / 'qwen3_14b.tar.gz').read_bytes(), self.q14)
        self.assertTrue((run / 'artifacts/qwen3_14b/summary.json').exists())
        self.assertNotEqual(run, environment)
        self.assertEqual(json.loads((run / 'launch.json').read_text())['frozen_worker'], remote.WORKER)
        with self.assertRaises(FileExistsError):
            remote.prepare(self.source, self.source_sha, self.seed, seed_sha, 'recovery')

    def test_hash_and_contract_refusals(self):
        seed_sha = self.seed_archive()
        with self.assertRaises(ValueError):
            remote.inspect_seed(self.seed, 'changed', self.source_sha)
        with self.assertRaises(ValueError):
            remote.inspect_seed(self.seed, seed_sha, 'changed')
        self.files['artifacts/qwen3_32b/batches/gsm8k-000.json'] = b'{"execution_contract_sha256":"changed"}'
        with self.assertRaisesRegex(ValueError, 'Batch contract'):
            remote.inspect_seed(self.seed, self.seed_archive(), self.source_sha)

    def test_unexpected_and_unsafe_files_rejected(self):
        for name in ['artifacts/qwen3_32b/summary.json', 'artifacts/../../outside', 'artifacts/gemma4_31b/manifest.json']:
            self.files[name] = b'{}'
            with self.assertRaises(ValueError):
                remote.inspect_seed(self.seed, self.seed_archive(), self.source_sha)
            del self.files[name]

    def test_inherited_archive_and_frozen_worker_refusals(self):
        self.files['completed/qwen3_14b.tar.gz'] = b'changed'
        with self.assertRaisesRegex(ValueError, 'inherited'):
            remote.inspect_seed(self.seed, self.seed_archive(), self.source_sha)
        self.files['completed/qwen3_14b.tar.gz'] = self.q14
        seed_sha = self.seed_archive()
        with patch.object(remote, 'WORKER_SHA', 'changed'):
            with self.assertRaisesRegex(ValueError, 'worker changed'):
                remote.prepare(self.source, self.source_sha, self.seed, seed_sha, 'recovery')

    def test_exact_worker_is_called_synchronously(self):
        run, environment = self.base / 'run', self.base / 'environment'
        worker = Mock()
        with patch.object(remote, 'install_environment') as install, patch.object(remote.runpy, 'run_path', return_value={'worker': worker}) as loader:
            remote.execute_frozen(run, environment)
        install.assert_called_once_with(run, environment)
        loader.assert_called_once_with(str(run / 'source' / remote.WORKER), run_name='frozen_query_recovery')
        worker.assert_called_once_with(run, environment)

    def test_install_failure_never_starts_scoring_and_retains_state(self):
        run, environment = self.base / 'run', self.base / 'environment'
        worker, save, archive = Mock(), Mock(), Mock(return_value={'verified': True})
        with patch.object(remote, 'install_environment', side_effect=RuntimeError('failure')), patch.object(remote.runpy, 'run_path', return_value={'worker': worker, 'save': save, 'archive': archive}):
            with self.assertRaises(RuntimeError):
                remote.execute_frozen(run, environment)
        worker.assert_not_called()
        self.assertFalse(save.call_args_list[0].args[1]['release_attempted'])
        archive.assert_called_once()

    def test_allocation_ownership_and_real_observer_cli(self):
        self.assertIsNone(owned([], 'new'))
        self.assertEqual(owned([{'name': 'new', 'endpoint': 'owned'}], 'new'), 'owned')
        with self.assertRaises(RuntimeError):
            owned([{'name': 'other', 'endpoint': 'other'}], 'new')
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(observer_command(root, 'recovery', 'owned') + ['--validate-cli'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PreparedSeedTests(unittest.TestCase):
    def test_actual_saved_contract_and_scientific_bundle(self):
        root = Path(os.environ['STRASSEN_PROJECT_ROOT'])
        run = root / 'runs/20261020-accuracy-study-v009'
        source = json.loads((run / 'source-manifest.json').read_text())
        seed = json.loads((run / 'seed-receipt.json').read_text())
        receipt = remote.inspect_seed(run / 'resume-seed.tar.gz', seed['sha256'], source['sha256'])
        self.assertEqual(receipt['saved_gsm_pairs'], 16)
        self.assertEqual(receipt['completed_task_evaluations'], 5)
        self.assertEqual(len([x for x in receipt['files'] if '/batches/' in x]), 21)
        self.assertEqual(remote.sha(run / 'source.tar.gz'), source['sha256'])
        with tarfile.open(run / 'source.tar.gz') as archive:
            self.assertEqual(hashlib.sha256(archive.extractfile(remote.WORKER).read()).hexdigest(), remote.WORKER_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
