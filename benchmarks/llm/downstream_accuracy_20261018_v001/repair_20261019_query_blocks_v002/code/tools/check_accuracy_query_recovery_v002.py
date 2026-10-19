"""Validate repaired engine, recovery and preflight event dispatch."""
import json
from pathlib import Path
import sys,tempfile,types,unittest
from unittest.mock import patch
from check_accuracy_query_recovery_v001 import EngineTests,RecoveryTests
import accuracy_query_block_preflight_v001 as preflight

class PreflightTests(unittest.TestCase):
    def test_engine_contract_event_and_failed_admission(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);pilot=root/'pilot.json';out=root/'output'
            model=preflight.MODELS['qwen3_32b']
            pilot.write_text(json.dumps(dict(scope='train_generation_timing_pilot',model=model[1],revision=model[2])))
            def engine(model_name,cache,emit):
                emit(dict(kind='engine_contract',model=model[1]))
                return types.SimpleNamespace(prepare=lambda:None)
            with patch.object(sys,'argv',['check','--model','qwen3_32b','--pilot',str(pilot),'--output',str(out),'--cache',str(root/'cache')]),patch.object(preflight.jax,'default_backend',return_value='tpu'),patch.object(preflight.jax,'devices',return_value=[types.SimpleNamespace(device_kind='TPU v6e')]),patch.object(preflight,'Engine',side_effect=engine),patch.object(preflight,'qualify',return_value=dict(status='failed')):
                with self.assertRaisesRegex(RuntimeError,'Original Native admission failed'):preflight.main()
            self.assertFalse((out/'summary.json').exists())
            event=json.loads((out/'events.jsonl').read_text());self.assertEqual(event['model'],model[1]);self.assertEqual(event['kind'],'engine_contract')

if __name__=='__main__':
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)for c in (EngineTests,RecoveryTests,PreflightTests))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
