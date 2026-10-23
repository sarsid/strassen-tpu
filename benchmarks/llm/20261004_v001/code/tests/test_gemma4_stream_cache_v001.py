"""Gemma4 full streamed semantics and exact private cache, CPU only.

Synthetic checkpoint state is temporary; only metrics enter the run archive.
The original strict predictive gates also apply to this native oracle check.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
import jax
import numpy as np
import torch
import test_gemma4_checkpoint_v002 as fixture
from strassen_mm import llm_stream_gemma4_v001 as stream
from strassen_mm import llm_weight_cache_gemma4_v001 as cache
from strassen_mm import model_gemma4_fused_v001 as fused


class Gemma4StreamCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.Gemma4CheckpointTests.setUpClass()
        cls.fixture = fixture.Gemma4CheckpointTests()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.cp = stream.checkpoint(cls.fixture.write_checkpoint(cls.folder, cls.fixture.shard_state()))
        cls.tokens = (np.arange(34).reshape(2, 17) % 96 + 1).astype(np.int32)
        with torch.no_grad():
            cls.official = cls.fixture.model(torch.tensor(cls.tokens, dtype=torch.long), use_cache=False).logits.float().numpy()
        cls.reference = cls.folder / 'reference'
        cls.reference.mkdir()
        flat = cls.official.reshape(-1, cls.cp.config['vocab_size'])
        for lo in range(0, len(flat), 7):
            np.save(cls.reference / f'{lo:06d}.npy', flat[lo:lo+7])
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        if os.environ.get('STRASSEN_EXECUTION_DIR'):
            folder = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
            folder.mkdir(exist_ok=True)
            (folder / 'gemma4-stream-cache-checks.json').write_text(json.dumps(cls.records, indent=2) + '\n')

    def test_native_full_forward_official_and_scoring_boundaries(self):
        for backend in ('explicit', 'xla'):
            with self.subTest(backend=backend):
                model = stream.StreamedModel(self.cp, 2, 17, logit_chunk=7, attention_backend=backend)
                events = []
                result = model.forward(self.tokens, capture=True, score=True,
                                       reference_logits=self.reference, emit=events.append)
                quality = result['quality']
                self.records.append(dict(kind='official_native_forward', backend=backend, quality=quality,
                                         synthetic=True, cpu=True))
                self.assertTrue(quality['finite'])
                self.assertEqual(quality['positions'], 32)  # Excludes each sequence's last token.
                self.assertLessEqual(abs(quality['delta_nll']), .01)
                self.assertLessEqual(quality['mean_kl'], .02)
                self.assertGreaterEqual(quality['top1_agreement'], .97)
                self.assertLessEqual(quality['logit_relative_l2'], .03)
                self.assertFalse(result['timing']['eligible_for_inference_timing'])
                self.assertEqual(len(result['hidden']), 2)
                self.assertEqual([e['layer'] for e in events], [0, 1])
                self.assertEqual(len(model.builders), 2)
                self.assertEqual(len(model.metadata[0]['matrix_product_sites']), 6)
                self.assertEqual(len(model.metadata[1]['matrix_product_sites']), 5)
                self.assertTrue(model.warmup(self.tokens)['eligible_for_inference_timing'])
                np.testing.assert_array_equal(model.embeddings(self.tokens), self.cp.embeddings(self.tokens))

    def test_cached_original_logits_hidden_and_scalar_dtype_exact(self):
        identity = cache.cache_identity(self.cp)
        self.assertEqual(identity['cache_version'], 'gemma4-v001')
        changed = deepcopy(identity)
        changed['config']['rms_norm_eps'] *= 2
        self.assertNotEqual(cache.key(identity), cache.key(changed))
        store = cache.LayoutCache(self.folder / 'cache-native', identity, max_bytes=8*1024**2)
        cached = cache.CachedCheckpoint(self.cp, store)
        self.assertNotIn('v', cached.layer(1))
        self.assertIn('v', cached.layer(0))
        self.assertEqual(cached.layer(0)['layer_scalar'].dtype, np.float32)
        original = stream.StreamedModel(self.cp, 2, 17, logit_chunk=7)
        restored = stream.StreamedModel(cached, 2, 17, logit_chunk=7,
            prepared_provider=lambda index, metadata: cache.packed_layer(cached, index, metadata))
        first = original.forward(self.tokens, capture=True, score=True, save_logits=self.folder / 'uncached')
        second = restored.forward(self.tokens, capture=True, score=True,
            save_logits=self.folder / 'cached', reference_logits=self.folder / 'uncached')
        for actual, expected in zip(second['hidden'], first['hidden']):
            np.testing.assert_array_equal(actual, expected)
        for name in first['logit_files']:
            np.testing.assert_array_equal(np.load(self.folder / 'uncached' / name),
                                          np.load(self.folder / 'cached' / name))
        self.assertEqual(second['quality']['logit_relative_l2'], 0)
        self.assertEqual(second['quality']['top1_agreement'], 1)
        self.assertEqual(second['quality']['delta_nll'], 0)
        self.records.append(dict(kind='cached_native_equivalence', exact=True, quality=second['quality'],
                                 stats=store.stats, synthetic=True, cpu=True))

    def test_custom_packing_exact_and_cache_reopens_verifies(self):
        identity = cache.cache_identity(self.cp)
        root = self.folder / 'cache-packed'
        store = cache.LayoutCache(root, identity, max_bytes=32*1024**2)
        cached = cache.CachedCheckpoint(self.cp, store)
        for index in range(2):
            for depth in (0, 1, 2):
                policy = fused.native_policy(self.cp.config, index)
                for site in policy:
                    policy[site] = dict(policy[site], implementation='cubic' if depth == 0 else 'current',
                                        depth=depth, accumulator='outputs', tile=[32, 512, 512])
                fn = fused.build_layer(self.cp.config, 17, policy, batch_size=2, layer_index=index,
                                       interpret=True, rounding='accumulator')
                expected = fn.prepare_weights({k: jax.numpy.asarray(v) for k, v in self.cp.layer(index).items()})
                actual = stream.pack_host(self.cp.layer(index), fn.metadata)
                packed = cache.packed_layer(cached, index, fn.metadata)
                for name in expected:
                    np.testing.assert_array_equal(actual[name], np.asarray(expected[name]))
                    np.testing.assert_array_equal(packed[name], actual[name])
                    self.assertFalse(packed[name].flags.writeable)
                    self.assertEqual(packed[name].dtype, actual[name].dtype)
                self.records.append(dict(kind='host_packing', layer=index, depth=depth,
                                         exact=True, names=list(expected), synthetic=True, cpu=True))
        reopened = cache.LayoutCache(root, identity, max_bytes=32*1024**2)
        twice = cache.CachedCheckpoint(self.cp, reopened)
        np.testing.assert_array_equal(twice.layer(1)['k'], self.cp.layer(1)['k'])
        self.assertGreater(reopened.stats['verified_bytes'], 0)
        # New opener must reject a byte mutation; no unverified cache reuse.
        descriptor = dict(kind='canonical', layer=1, name='k')
        data = reopened.root / (cache.key(descriptor) + '.bin')
        with data.open('r+b') as f:
            byte = f.read(1)
            f.seek(0)
            f.write(bytes([byte[0] ^ 1]))
        corrupted = cache.CachedCheckpoint(self.cp, cache.LayoutCache(root, identity))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            corrupted.layer(1)

    def test_invalid_workload_and_output_contract(self):
        with self.assertRaisesRegex(ValueError, 'BF16'):
            stream.StreamedModel(self.cp, 2, 17, output_dtype='float32')
        with self.assertRaisesRegex(ValueError, 'At least two'):
            stream.StreamedModel(self.cp, 2, 1)
        model = stream.StreamedModel(self.cp, 2, 17)
        for tokens in (self.tokens[:1], self.tokens.astype(np.float32),
                       np.full_like(self.tokens, self.cp.config['vocab_size'])):
            with self.assertRaises(ValueError):
                model.forward(tokens)


if __name__ == '__main__':
    unittest.main(verbosity=2)
