"""Static v6e menu audit plus real CPU native-fallback tuning orchestration.

CPU timings are validation diagnostics only, never kernel selection evidence for
the TPU campaign. The tiny workload deliberately prunes custom padded tiles.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import test_gemma4_checkpoint_v002 as fixture
from strassen_mm import llm_campaign_gemma4_v001 as campaign
from strassen_mm import tuner_llm_gemma4_v001 as tuner
from strassen_mm.checkpoint_gemma4_v001 import Checkpoint

ROOT = Path(__file__).resolve().parents[1]


class Gemma4TunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'results/v6e/llm_campaign_20261003_v001/inputs_gemma4_v001/gemma4_31b.config.json'
        cls.official = json.loads(path.read_text())
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('STRASSEN_EXECUTION_DIR'):
            folder = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
            folder.mkdir(exist_ok=True)
            (folder / 'gemma4-tuner-checks.json').write_text(json.dumps(cls.records, indent=2) + '\n')

    def test_pinned_geometry_and_all_candidate_decisions(self):
        for index, dimension in ((0, 256), (5, 512)):
            c, shapes, specs = campaign.geometry(self.official, 8, 1024, index)
            self.assertEqual(specs['q'].head_dim, dimension)
            self.assertEqual(specs['q'].norm, 'gemma4')
            self.assertEqual(shapes['q'], [8192, 5376, 32*dimension])
            self.assertEqual(shapes['k'], [8192, 5376, 4096 if index == 0 else 2048])
            self.assertEqual('v' in shapes, index == 0)
            self.assertEqual(specs['k'].kind, 'qk_norm_rope' if index == 0 else 'none')
            for site, shape in shapes.items():
                for family in ('native_default', 'native_tuned', 'cubic_tuned', 's1', 's2'):
                    menu = list(campaign.candidates(shape, specs[site], 'bfloat16', family))
                    self.assertEqual(len({r['candidate_id'] for r in menu}), len(menu))
                    if family.startswith('native'):
                        self.assertEqual(len(menu), 1 if family == 'native_default' else 6)
                        continue
                    for row in menu:
                        self.assertIn(row['arm']['depth'], (0, 1, 2))
                        self.assertEqual(row['arm']['architecture'], 'v6e')
                        if row['disposition'] == 'offered':
                            self.assertFalse(row['reasons'])
                            self.assertEqual(row['metadata']['kernel_version'], 'kernels_fused_v005')
                            self.assertLessEqual(row['padded_work_ratio'], 2)
                            if specs[site].rotary:
                                self.assertEqual(row['arm']['tile'][1] % (2*dimension), 0)
                        else:
                            self.assertTrue(row['reasons'])
                    chosen = tuner.shortlist(menu, 12)
                    again = tuner.shortlist(list(campaign.candidates(shape, specs[site], 'bfloat16', family)), 12)
                    self.assertEqual([r['candidate_id'] for r in chosen], [r['candidate_id'] for r in again])
                    self.assertLessEqual(len(chosen), 12)
                    self.assertTrue(all(r['search_disposition'] == 'selected_for_measurement' for r in chosen))
                    categories = lambda rows: {(r['arm']['implementation'], r['arm']['accumulator'],
                                                r['arm']['buffers'], r['metadata']['full_contraction'])
                                               for r in rows if r['disposition'] == 'offered'}
                    self.assertEqual(categories(chosen), categories(menu))
                    self.records.append(dict(kind='candidate_menu', layer=index, site=site, family=family,
                        candidates=len(menu), offered=sum(r['disposition']=='offered' for r in menu),
                        measured=0, selected_for_future_measurement=len(chosen)))

    def test_real_cpu_tuning_orchestration_with_explicit_native_fallbacks(self):
        fixture.Gemma4CheckpointTests.setUpClass()
        helper = fixture.Gemma4CheckpointTests()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cp = Checkpoint(helper.write_checkpoint(folder, helper.shard_state()))
            tokens = (np.arange(17)[None] % 96 + 1).astype(np.int32)
            events = []
            # Keep source-linked dispositions/profiles as diagnostics, no weights.
            output = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts/tuner-smoke'
            profiles = tuner.tune(cp, tokens, 'bfloat16', output, events.append, limit=1)
            saved = json.loads((output / 'profiles.json').read_text())
            self.assertEqual(saved['representative_layers'], {'full_attention': 1, 'sliding_attention': 0})
            self.assertNotIn('v', saved['calibration_geometry']['full_attention']['projection_shapes_mkn'])
            self.assertEqual(set(profiles), {'native_default', 'native_tuned', 'cubic_tuned', 's1', 's2'})
            self.assertEqual(len(list(output.glob('*-menu.json'))), 33)
            self.assertFalse(list(output.glob('full_attention-v-*')))
            for name, profile in profiles.items():
                for kind, entry in profile['by_layer_type'].items():
                    self.assertEqual(len(entry['policy']), 6 if kind == 'sliding_attention' else 5)
                    self.assertTrue(all(a['implementation']=='native' for a in entry['policy'].values()))
                    if name not in ('native_default', 'native_tuned'):
                        self.assertEqual(set(entry['fallbacks']), set(entry['policy']))
                        self.assertEqual(entry['selected_custom_sites'], [])
            rows = [r for r in events if r['kind']=='native_screen' and r['status']=='eligible']
            self.assertTrue(rows)
            self.assertTrue(all(len(r['samples_ms']) == 8 for r in rows))
            gates = [r for r in events if r['kind']=='tuned_layer_gate']
            self.assertEqual(len(gates), 6)
            self.assertTrue(all(r['errors']['finite'] and r['errors']['relative_l2']<=.1 for r in gates))
            (output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
            self.records.append(dict(kind='native_fallback_orchestration', synthetic=True, cpu=True,
                representative_layers=saved['representative_layers'], custom_sites_selected=0,
                final_layer_gates=len(gates), eligible_native_screens=len(rows),
                not_device_tuning_evidence=True))

    def test_reject_fp32_unknown_families_and_invalid_budget(self):
        _, shapes, specs = campaign.geometry(self.official, 1, 512, 0)
        for dtype, family in (('float32','s1'), ('bfloat16','s3'), ('bfloat16','s4')):
            with self.assertRaises(ValueError):
                list(campaign.candidates(shapes['q'], specs['q'], dtype, family))
        for dtype, limit in (('float32',12), ('bfloat16',0), ('bfloat16',13)):
            with self.assertRaises(ValueError):
                tuner.tune(None, None, dtype, None, None, limit=limit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
