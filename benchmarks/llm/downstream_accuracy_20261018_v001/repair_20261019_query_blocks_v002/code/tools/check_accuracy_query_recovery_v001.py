"""Validate retained-runtime recovery gates without remote execution."""
import importlib.util
import json
from pathlib import Path
import sys,tempfile,types,unittest
from unittest.mock import patch
from check_accuracy_engine_v006 import Tests as EngineTests

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('query_recovery',ROOT/'runtime/accuracy_query_block_recovery_v001.py')
recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)

class RecoveryTests(unittest.TestCase):
    def execute_worker(self,fail_preflight):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);run=base/'runs/new';run.mkdir(parents=True);(run/'source').mkdir()
            previous=base/'runs/20261019-accuracy-study-v006';previous.mkdir()
            private=base/'.runtime_private'/previous.name
            python=private/'environment/bin/python';python.parent.mkdir(parents=True);python.touch()
            cache=private/'weights-qwen3_32b/canonical';cache.mkdir(parents=True);(cache/'keep-on-error').touch()
            (run/'launch.json').write_text(json.dumps(dict(source_sha256='fixture')))
            commands=[]
            def execute(command,*args):
                commands.append(command)
                output=Path(command[command.index('--output')+1]);is_preflight=command[1].endswith('accuracy_query_block_preflight_v001.py')
                if is_preflight and fail_preflight:return 1
                output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(dict(status='passed')))
                return 0
            helper=types.ModuleType('parent_exact_remote_v001');helper.execute=execute
            with patch.object(recovery,'BASE',base),patch.dict(sys.modules,{'parent_exact_remote_v001':helper}),patch.object(recovery.shutil,'disk_usage',return_value=types.SimpleNamespace(free=500*1024**3)):
                recovery.worker(run,previous)
            state=json.loads((run/'status.json').read_text())
            self.assertFalse(state['release_attempted'])
            self.assertTrue((run/'archive-ready.json').exists())
            if fail_preflight:
                self.assertEqual(state['status'],'failed');self.assertEqual(state['completed_models'],['qwen3_14b'])
                self.assertEqual(len(commands),2);self.assertTrue(all(c[1].endswith('accuracy_query_block_preflight_v001.py') for c in commands))
                self.assertTrue((cache/'keep-on-error').exists())
            else:
                self.assertEqual(state['status'],'completed');self.assertEqual(state['test_evaluations_completed'],9)
                self.assertEqual([Path(c[1]).name for c in commands],['accuracy_query_block_preflight_v001.py','accuracy_study_v002.py']*2)
                self.assertTrue(all(c[c.index('--model')+1] in ('qwen3_32b','gemma4_31b') for c in commands))
                self.assertFalse(cache.exists())
                for model in ('qwen3_32b','gemma4_31b'):self.assertTrue((run/(model+'.tar.ready.json')).exists())

    def test_failed_preflight_retains_cache_and_never_scores(self):self.execute_worker(True)
    def test_completed_qwen14_is_not_reexecuted(self):self.execute_worker(False)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(EngineTests),unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)])
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
