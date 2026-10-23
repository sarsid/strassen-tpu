"""Check Gemma recovery integrity, admission gates and unchanged study commands."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import resume_accuracy_gemma_remote_v002 as remote
import accuracy_gemma_recovery_v001 as worker
from start_accuracy_study_v008 import owned, observer_command, validate_launch


def tar(path, files):
    with tarfile.open(path, 'w:gz') as out:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            out.addfile(member, io.BytesIO(data))


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.source = self.base / 'source.tar.gz'
        self.frozen_worker = b'def worker(run, previous): pass\n'
        tar(self.source, {remote.WORKER: self.frozen_worker})
        self.source_sha = remote.sha(self.source)
        self.operational = self.base / 'operational.py'
        self.operational.write_text('def worker(run, previous): pass\n')
        self.files = {}
        completed = {}
        for model in ('qwen3_14b', 'qwen3_32b'):
            path = self.base / (model + '.tar.gz')
            tar(path, {model + '/summary.json': b'{"done":true}'})
            self.files['completed/' + path.name] = path.read_bytes()
            completed[model] = remote.sha(path)
        self.contract = {'bundle_sha256': self.source_sha}
        self.digest = remote.digest(self.contract)
        self.manifest = {'execution_contract_sha256': self.digest, 'admission_status': 'passed',
            'examples': [{'task': task, 'example_id': task+str(i)} for task,n in [('mmlu',114),('ppl',32),('gsm8k',32)] for i in range(n)]}
        self.files['artifacts/gemma4_31b/manifest.json'] = json.dumps(self.manifest).encode()
        self.files['artifacts/gemma4_31b/execution-contract.json'] = json.dumps(self.contract).encode()
        self.files['artifacts/gemma4_31b/provenance/original-runtime-evidence.json'] = b'{"original_admission_receipt_available":false}'
        self.files['artifacts/gemma4_31b/provenance/admission-20261021T065510Z-20a85c.json'] = b'{"status":"passed","all_logits_finite":true}'
        self.files['artifacts/gemma4_31b/provenance/query-block-preflight-0.json'] = b'{"status":"passed"}'
        for task,count in remote.SAVED_COUNTS.items():
            for first in range(0,count,8):
                n=min(8,count-first)
                self.files[f'artifacts/gemma4_31b/batches/{task}-{first//8:03d}.json'] = json.dumps(dict(
                    task=task, example_ids=[task+str(k) for k in range(first,first+n)],
                    execution_contract_sha256=self.digest,
                    results={'native':[{}]*n,'strassen':[{}]*n})).encode()
        self.seed = self.base / 'seed.tar.gz'
        for name, value in [('BASE', self.base), ('COMPLETED', completed),
                            ('WORKER_SHA', hashlib.sha256(self.frozen_worker).hexdigest())]:
            p = patch.object(remote, name, value)
            p.start(); self.addCleanup(p.stop)

    def seed_archive(self):
        batches={}
        for name,data in self.files.items():
            if '/batches/' in name:
                filename=name.rsplit('/',1)[1]
                original=filename.startswith('mmlu-') and int(filename[5:8])<13
                batches[filename]=dict(sha256=hashlib.sha256(data).hexdigest(),
                    execution_run='20261020-accuracy-study-v010' if original else '20261021-accuracy-study-v011')
        admission='admission-20261021T065510Z-20a85c.json'
        lineage=dict(batches=batches,v011_admission=dict(file=admission,
            sha256=hashlib.sha256(self.files['artifacts/gemma4_31b/provenance/'+admission]).hexdigest()),
            v011_preflight_sha256=hashlib.sha256(self.files['artifacts/gemma4_31b/provenance/query-block-preflight-0.json']).hexdigest())
        self.files['artifacts/gemma4_31b/provenance/resume-lineage-v002.json']=json.dumps(lineage).encode()
        receipt = dict(source_sha256=self.source_sha, saved_pairs=remote.SAVED_COUNTS,completed_task_evaluations=8,
            files={k:hashlib.sha256(v).hexdigest() for k,v in self.files.items()})
        tar(self.seed, dict(self.files, **{'seed-manifest.json':json.dumps(receipt).encode()}))
        return remote.sha(self.seed)

    def test_exact_restore_of_both_models_and_prefix(self):
        run, env = remote.prepare(self.source,self.source_sha,self.seed,self.seed_archive(),
            'resume',self.operational,remote.sha(self.operational))
        for name,data in self.files.items():self.assertEqual((run/name).read_bytes(),data)
        state=json.loads((run/'status.json').read_text())
        self.assertEqual(state['completed_models'],['qwen3_14b','qwen3_32b'])
        self.assertEqual(state['test_evaluations_completed'],6)
        progress=json.loads((run/'progress.json').read_text())
        self.assertEqual(progress['completed_tasks'],['mmlu','ppl'])
        self.assertEqual(progress['completed_examples'],16)
        self.assertEqual(progress['task'],'gsm8k')
        self.assertFalse(json.loads((run/'inherited-results.json').read_text())['original_gemma_admission_receipt_available'])
        self.assertNotEqual(run,env)
        with self.assertRaises(FileExistsError):
            remote.prepare(self.source,self.source_sha,self.seed,remote.sha(self.seed),'resume',self.operational,remote.sha(self.operational))

    def test_changed_contract_order_and_missing_pair_refused(self):
        name='artifacts/gemma4_31b/batches/mmlu-000.json'; original=self.files[name]
        for field,value in [('execution_contract_sha256','changed'),('example_ids',['wrong']*8),('results',{'native':[{}]*8,'strassen':[{}]*7})]:
            row=json.loads(original);row[field]=value;self.files[name]=json.dumps(row).encode()
            with self.assertRaises(ValueError):remote.inspect_seed(self.seed,self.seed_archive(),self.source_sha)
        self.files[name]=original

    def test_changed_hash_unknown_paths_and_receipt_fabrication_refused(self):
        digest=self.seed_archive()
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,'wrong',self.source_sha)
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,digest,'wrong')
        for name in ['artifacts/../../outside','artifacts/gemma4_31b/summary.json']:
            self.files[name]=b'{}'
            with self.assertRaises(ValueError):remote.inspect_seed(self.seed,self.seed_archive(),self.source_sha)
            del self.files[name]
        self.files['artifacts/gemma4_31b/provenance/original-runtime-evidence.json']=b'{"original_admission_receipt_available":true}'
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,self.seed_archive(),self.source_sha)

    def test_missing_saved_batch_or_failed_prior_admission_refused(self):
        name='artifacts/gemma4_31b/batches/gsm8k-001.json'
        old=self.files.pop(name)
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,self.seed_archive(),self.source_sha)
        self.files[name]=old
        admission='artifacts/gemma4_31b/provenance/admission-20261021T065510Z-20a85c.json'
        self.files[admission]=b'{"status":"failed","all_logits_finite":true}'
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,self.seed_archive(),self.source_sha)

    def test_operational_hash_and_install_gate(self):
        digest=self.seed_archive()
        with self.assertRaises(ValueError):remote.prepare(self.source,self.source_sha,self.seed,digest,'bad',self.operational,'wrong')
        call,save,archive=Mock(),Mock(),Mock(return_value={})
        with patch.object(remote,'install_environment',side_effect=RuntimeError('failed')),patch.object(remote.runpy,'run_path',return_value={'worker':call,'save':save,'archive':archive}):
            with self.assertRaises(RuntimeError):remote.execute_operational(self.base/'run',self.base/'env',self.operational,remote.sha(self.operational))
        call.assert_not_called()
        self.assertEqual(save.call_args_list[0].args[1]['test_evaluations_completed'],6)
        self.assertFalse(save.call_args_list[0].args[1]['release_attempted'])

    def test_frozen_study_and_preflight_commands_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        def commands(path):
            tree=ast.parse(path.read_text());result={}
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('cmd','check'):
                    result[node.targets[0].id]=ast.dump(node.value)
            return result
        self.assertEqual(commands(root/'runtime/accuracy_query_block_recovery_v001.py'),commands(root/'runtime/accuracy_gemma_recovery_v001.py'))
        self.assertEqual(worker.MODELS,('gemma4_31b',))

    def test_supervisor_ownership_and_launch_identity(self):
        self.assertIsNone(owned([],'new'))
        with self.assertRaises(RuntimeError):owned([{'name':'other','endpoint':'other'}],'new')
        op={'runtime/resume_accuracy_gemma_remote_v002.py':'boot','runtime/accuracy_gemma_recovery_v001.py':'worker'}
        launch=dict(source_sha256='source',seed_sha256='seed',launch_mode='foreground_notebook',bootstrap_sha256='boot',operational_worker_sha256='worker')
        validate_launch(launch,{'sha256':'source'},{'sha256':'seed'},op)
        with self.assertRaises(RuntimeError):validate_launch(dict(launch,operational_worker_sha256='wrong'),{'sha256':'source'},{'sha256':'seed'},op)
        root=Path(__file__).resolve().parents[1]
        checked=subprocess.run(observer_command(root,'resume','owned')+['--validate-cli'],capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stderr)

    def test_failed_preflight_never_scores_or_releases(self):
        run=self.base/'run';run.mkdir();(run/'source/runtime').mkdir(parents=True)
        (run/'launch.json').write_text(json.dumps({'source_sha256':self.source_sha}))
        previous=self.base/'environment';(previous/'artifacts').mkdir(parents=True)
        private=self.base/'.runtime_private'/previous.name/'environment/bin';private.mkdir(parents=True);(private/'python').touch()
        execute=Mock(return_value=1)
        with patch.object(worker,'BASE',self.base),patch.object(worker.shutil,'disk_usage',return_value=types.SimpleNamespace(free=200*1024**3)),patch.dict(sys.modules,{'parent_exact_remote_v001':types.SimpleNamespace(execute=execute)}):
            worker.worker(run,previous)
        self.assertEqual(execute.call_count,2)
        self.assertTrue(all('accuracy_query_block_preflight_v001.py' in c.args[0][1] for c in execute.call_args_list))
        state=json.loads((run/'status.json').read_text())
        self.assertEqual(state['status'],'failed');self.assertFalse(state['release_attempted'])
        self.assertEqual(state['completed_models'],['qwen3_14b','qwen3_32b'])


if __name__=='__main__':unittest.main(verbosity=2)
