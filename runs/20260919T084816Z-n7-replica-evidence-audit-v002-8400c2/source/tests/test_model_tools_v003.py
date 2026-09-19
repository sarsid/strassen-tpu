"""Offline v002 recovery guards with canonical temporary fixture roots.

macOS temporary paths can use /var while Path.resolve() returns /private/var.
Production ROOT already resolves __file__; the Linux installer PROJECT is a
canonical /content path. Fixture constants must preserve that same invariant.
The frozen v002 suite is retained, and these tests still exercise v002 code.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


installer = module('install_model_tools_v002')
resume = module('resume_n9_inputs_v002')


class RecoveryGuards(unittest.TestCase):
    def test_every_pip_operation_targets_new_interpreter(self):
        commands = installer.installation_commands('/usr/bin/python3', Path('/fresh/model-tools-v002'), Path('/evidence'))
        self.assertEqual(commands[0], ['/usr/bin/python3', '-m', 'venv', '--without-pip', '/fresh/model-tools-v002'])
        for command in commands[1:]:
            self.assertEqual(command[:6], ['/usr/bin/python3', '-m', 'pip', '--isolated', '--python', '/fresh/model-tools-v002/bin/python'])
        installations = [command for command in commands if 'install' in command]
        self.assertEqual(len(installations), 2)
        self.assertIn('torch==2.8.0', installations[0])
        self.assertEqual(installations[1][-len(installer.PINS):], installer.PINS)
        for command in installations:
            self.assertIn('--only-binary=:all:', command)
            self.assertIn('--no-cache-dir', command)
            self.assertNotIn('--upgrade', command)

    def test_existing_partial_and_escaped_venvs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'model-tools-v001').mkdir()
            with patch.object(installer, 'PROJECT', root):
                self.assertEqual(installer.validate_new_target(root / 'model-tools-v002'), root / 'model-tools-v002')
                with self.assertRaisesRegex(FileExistsError, 'existing or partial environment'):
                    installer.validate_new_target(root / 'model-tools-v001')
                with self.assertRaisesRegex(ValueError, 'inside the authorized project'):
                    installer.validate_new_target(root.parent / 'escaped-model-tools')
                (root / 'model-tools-v002').mkdir()
                with self.assertRaisesRegex(FileExistsError, 'existing or partial environment'):
                    installer.validate_new_target(root / 'model-tools-v002')

    def make_inputs(self, root):
        endpoint = 'test-v5e-endpoint'
        models, steps = [], []

        def write_json(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, sort_keys=True))

        def seal(run):
            prep = run / 'artifacts/preparation-00'
            files = {str(path.relative_to(prep)): {'bytes': path.stat().st_size, 'sha256': resume.digest(path)}
                     for path in prep.iterdir() if path.name != 'artifact_manifest.json'}
            write_json(prep / 'artifact_manifest.json', {'files': files})
            files = {str(path.relative_to(run)): {'bytes': path.stat().st_size, 'sha256': resume.digest(path)}
                     for path in run.rglob('*') if path.is_file() and path.name != 'artifact-manifest.json'}
            write_json(run / 'artifact-manifest.json', files)

        def run(label, filename, manifest):
            directory = root / 'runs' / label
            write_json(directory / 'completion.json', {'status': 'completed', 'allocation_id': endpoint,
                                                       'remote_may_still_be_running': False})
            write_json(directory / 'artifacts/preparation-00/status.json', {'status': 'completed'})
            write_json(directory / 'artifacts/preparation-00' / filename, manifest)
            seal(directory)
            steps.append({'run_id': label, 'run_path': str(directory), 'status': 'completed',
                          'phase': 'N9-corpus' if label == 'corpus' else 'N9-download-' + label,
                          'completion_sha256': resume.digest(directory / 'completion.json')})
            return directory

        for slug, model_id, revision in resume.MODELS:
            cache = resume.BASE + '/models/' + model_id.replace('/', '--') + '/' + revision
            manifest = {'model_id': model_id, 'revision': revision, 'cache_dir': cache,
                        'files': [{'path': 'model.safetensors', 'bytes': 16, 'sha256': 'a' * 64}]}
            directory = run(slug, 'model_manifest.json', manifest)
            models.append({'model_id': model_id, 'revision': revision, 'status': 'checkpoint_ready',
                           'download_run_id': slug, 'model_manifest': cache + '/manifest.json',
                           'local_model_manifest': str(directory / 'artifacts/preparation-00/model_manifest.json')})
        models.append({'model_id': 'google/gemma-3-1b-pt', 'status': 'blocked_access', 'reason': 'No permission'})
        corpus = run('corpus', 'corpus_manifest.json', {'dataset_id': 'Salesforce/wikitext',
            'revision': resume.CORPUS_REVISION, 'configuration': 'wikitext-2-raw-v1', 'split': 'test',
            'cache_dir': resume.BASE + '/models/corpus/' + resume.CORPUS_REVISION,
            'files': [{'path': 'test-000.parquet', 'sha256': 'b' * 64, 'bytes': 16}]})
        bundle = root / 'input_bundle.json'
        write_json(bundle, {'cohort': endpoint, 'models': models, 'steps': steps})
        return endpoint, bundle, corpus, seal, write_json

    def test_valid_reuse_preserves_blocked_model_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            endpoint, bundle, corpus, _, _ = self.make_inputs(root)
            with patch.object(resume, 'ROOT', root):
                _, models, corpus_info, sources = resume.validate_reuse(bundle, corpus, endpoint)
            self.assertEqual([item['status'] for item in models], ['checkpoint_ready', 'checkpoint_ready', 'blocked_access'])
            self.assertEqual(len(sources), 3)
            self.assertEqual(corpus_info['corpus_manifest_sha256'], resume.digest(corpus / 'artifacts/preparation-00/corpus_manifest.json'))
            self.assertEqual(models[0]['model_manifest_sha256'], sources[0]['sha256'])

    def test_changed_checkpoint_bytes_or_original_completion_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            endpoint, bundle, corpus, seal, write = self.make_inputs(root)
            model = root / 'runs/qwen/artifacts/preparation-00/model_manifest.json'
            original = model.read_bytes()
            model.write_bytes(original + b' ')
            with patch.object(resume, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Execution artifact hash/size mismatch'):
                    resume.validate_reuse(bundle, corpus, endpoint)
            model.write_bytes(original)
            completion = root / 'runs/qwen/completion.json'
            value = json.loads(completion.read_text()); value['extra'] = 'changed despite reseal'
            write(completion, value); seal(root / 'runs/qwen')
            with patch.object(resume, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Original bundle completion SHA256 changed'):
                    resume.validate_reuse(bundle, corpus, endpoint)

    def test_wrong_corpus_split_and_endpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            endpoint, bundle, corpus, seal, write = self.make_inputs(root)
            with patch.object(resume, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Original input bundle allocation'):
                    resume.validate_reuse(bundle, corpus, 'different-endpoint')
            path = corpus / 'artifacts/preparation-00/corpus_manifest.json'
            value = json.loads(path.read_text()); value['split'] = 'train'
            write(path, value); seal(corpus)
            with patch.object(resume, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Corpus differs from frozen dataset/revision/split'):
                    resume.validate_reuse(bundle, corpus, endpoint)


if __name__ == '__main__': unittest.main()
