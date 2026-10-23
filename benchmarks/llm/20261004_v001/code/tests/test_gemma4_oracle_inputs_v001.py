"""Independent oracle export and immutable input-template checks; no downloads."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import torch
import test_gemma4_checkpoint_v002 as fixture
from strassen_mm.checkpoint_gemma4_v001 import Checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import gemma4_oracle_inputs_v001 as inputs


class Gemma4OracleInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.Gemma4CheckpointTests.setUpClass()
        helper = fixture.Gemma4CheckpointTests()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        checkpoint_dir = cls.root/'checkpoint'
        checkpoint_dir.mkdir()
        cls.cp = Checkpoint(helper.write_checkpoint(checkpoint_dir, helper.shard_state()))
        cls.ids = (np.arange(64)[None] % 96 + 1).astype(np.int32)
        cls.events = []
        cls.export = inputs.export_oracle(checkpoint_dir, cls.ids, cls.root/'official', cls.events.append)
        cls.model = helper.model
        cls.identity = inputs.source_identity()
        cls.model_id = dict(id='synthetic_gemma4', repo_id='synthetic/gemma4-test', revision='a'*40)
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if os.environ.get('STRASSEN_EXECUTION_DIR'):
            out = Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
            out.mkdir(exist_ok=True)
            (out/'gemma4-oracle-input-checks.json').write_text(json.dumps(cls.records, indent=2)+'\n')

    def template(self, target):
        target.mkdir()
        shutil.copytree(self.root/'official', target/'official')
        np.save(target/'oracle-tokens.npy', self.ids)
        (target/'checkpoint-provenance.json').write_text(json.dumps({k:v for k,v in self.cp.manifest.items() if k!='cache_dir'}))
        (target/'summary.json').write_text(json.dumps(dict(model=self.model_id, actual_checkpoint=False,
            synthetic=True, oracle=self.export, batch=1, sequence=64)))
        for split in ('train','test'):
            tokens = (np.arange(256) % 96 + 1).astype(np.int32)
            if split == 'test':tokens = tokens[::-1].copy()
            np.save(target/(split+'-all.npy'), tokens)
            (target/(split+'-tokens.json')).write_text(json.dumps(dict(split=split, synthetic=True,
                selection='synthetic distinct train/test streams', tokenizer_revision='a'*40)))
        return inputs.cache_receipt(target, self.model_id, self.identity)

    def test_exported_layers_logits_and_source_identity(self):
        captures = []
        hooks = [layer.register_forward_hook(lambda m,args,out: captures.append((args[0].detach().float().numpy(),out.detach().float().numpy())))
                 for layer in self.model.model.layers]
        try:
            with torch.inference_mode():
                logits = self.model(torch.tensor(self.ids,dtype=torch.long),use_cache=False).logits.float().numpy()
        finally:
            for hook in hooks:hook.remove()
        np.testing.assert_array_equal(logits,np.load(self.root/'official/logits.npy'))
        for index,(before,after) in enumerate(captures):
            np.testing.assert_array_equal(before,np.load(self.root/'official'/f'layer-{index:03d}-input.npy'))
            np.testing.assert_array_equal(after,np.load(self.root/'official'/f'layer-{index:03d}-output.npy'))
        self.assertEqual(self.export['status'],'reference_exported')
        self.assertNotIn('passed',self.export)
        self.assertEqual(self.export['tokens'],64)
        self.assertEqual(self.export['layers'],2)
        self.assertEqual([e['layer'] for e in self.events],[0,1])
        self.assertEqual(self.identity['official_source_sha256'],inputs.OFFICIAL_SOURCES)
        for name,row in self.export['artifacts'].items():
            self.assertEqual(inputs.digest(self.root/'official'/name),row['sha256'])
        self.records.append(dict(kind='official_export',synthetic=True,exact=True,reference=self.export))

    def test_cached_windows_keep_oracle_and_provenance_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.template(folder/'template')
            for batch,sequence in ((1,17),(4,32),(8,16)):
                output=folder/f'b{batch}s{sequence}'
                summary=inputs.reuse_inputs(folder/'template',output,self.model_id,self.identity,batch,sequence)
                self.assertTrue(summary['reused_model_inputs'])
                self.assertFalse(summary['actual_checkpoint'])
                for split in ('train','test'):
                    original=np.load(folder/'template'/(split+'-all.npy'))
                    selected=np.load(output/(split+'-tokens.npy'))
                    np.testing.assert_array_equal(selected,original[:batch*sequence].reshape(batch,sequence))
                    meta=json.loads((output/(split+'-tokens.json')).read_text())
                    self.assertEqual(meta['shape'],[batch,sequence])
                    self.assertEqual(meta['split'],split)
                for name in ('oracle-tokens.npy','checkpoint-provenance.json','official/logits.npy'):
                    self.assertEqual(inputs.digest(output/name),inputs.digest(folder/'template'/name))
                self.records.append(dict(kind='cached_window',batch=batch,sequence=sequence,synthetic=True,exact=True))

    def test_cache_identity_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'template';self.template(folder)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                inputs.cache_receipt(folder,self.model_id,self.identity)
            for changed_model,changed_source in ((dict(self.model_id,revision='b'*40),self.identity),
                                                 (self.model_id,dict(self.identity,version='changed'))):
                with self.assertRaisesRegex(ValueError,'identity'):
                    inputs.verify_cache(folder,changed_model,changed_source)
            target=folder/'oracle-tokens.npy';original=target.read_bytes()
            target.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaisesRegex(ValueError,'checksum'):
                inputs.verify_cache(folder,self.model_id,self.identity)
            target.write_bytes(original)
            (folder/'unlisted').write_text('unexpected')
            with self.assertRaisesRegex(ValueError,'file set'):
                inputs.verify_cache(folder,self.model_id,self.identity)
            (folder/'unlisted').unlink()
            target.unlink();target.symlink_to(self.root/'official/logits.npy')
            with self.assertRaisesRegex(ValueError,'escapes'):
                inputs.verify_cache(folder,self.model_id,self.identity)

    def test_invalid_tokens_and_insufficient_corpus(self):
        for array,batch,sequence in ((np.arange(3,dtype=np.int32),1,4),
                                    (np.arange(8,dtype=np.int64),1,4),
                                    (np.arange(8,dtype=np.int32).reshape(2,4),1,4),
                                    (np.array([-1,1],np.int32),1,2),
                                    (np.arange(8,dtype=np.int32),0,4)):
            with self.assertRaises(ValueError):inputs.token_window(array,batch,sequence)
        for ids in (self.ids.astype(np.int64),self.ids[:,:1],np.tile(self.ids,(2,1))):
            with self.assertRaisesRegex(ValueError,'oracle sequence'):
                inputs.export_oracle(self.cp.root,ids,self.root/'invalid')


if __name__=='__main__':unittest.main(verbosity=2)
