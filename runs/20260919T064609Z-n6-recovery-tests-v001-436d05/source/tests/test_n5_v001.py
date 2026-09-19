"""Offline N5 budget/selection guards and full-tile accumulator algebra."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from strassen_mm import benchmark_v001 as base
from strassen_mm import benchmark_n5_v001 as tuning
from strassen_mm.kernels_v002 import make_matmul


ROOT = Path(__file__).resolve().parents[1]


class N5Protocol(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(base, "np", np, create=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.campaign = json.loads((ROOT / "configs/campaign_n5_n9_v1.json").read_text())
        self.shapes = json.loads((ROOT / "configs/shapes_v1.json").read_text())

    def test_equal_attempt_budget_native_once_and_both_cost_scopes(self):
        groups = tuning.build_groups(self.campaign, self.shapes, "N5-screen")
        shape_ids = self.campaign["experiments"]["N5-screen"]["shape_ids"]
        self.assertEqual(len(shape_ids), 16)
        self.assertEqual(len(groups), 16 * 20)
        for shape_id in shape_ids:
            selected = [g for g in groups if g["shape"]["id"] == shape_id]
            arms = [arm for group in selected for arm in group["arms"]]
            self.assertEqual(sum(a["family"] == "native" for a in arms), 1)
            for family in tuning.FAMILIES:
                candidates = [a for a in arms if a["family"] == family]
                self.assertEqual(len(candidates), 20)
                self.assertEqual(len({a["candidate_id"] for a in candidates}), 20)
            self.assertTrue(all(g["scopes"] == ["call", "prepared_kernel"] for g in selected))
        total = sum(len(g["arms"]) * len(g["scopes"]) for g in groups)
        self.assertEqual(total, 1312)
        changed = copy.deepcopy(self.campaign["experiments"]["N5-screen"])
        changed["candidate_families"]["cubic"].pop()
        with self.assertRaisesRegex(ValueError, "equal candidate"):
            tuning.validate_candidate_space(changed)

    def selection_fixture(self):
        campaign = copy.deepcopy(self.campaign)
        experiment = campaign["experiments"]["N5-screen"]
        shape_id = experiment["shape_ids"][0]
        experiment["shape_ids"] = [shape_id]
        campaign["experiments"]["N5-confirm"]["shape_ids"] = [shape_id]
        rows = []
        for family in tuning.FAMILIES:
            for index, candidate in enumerate(experiment["candidate_families"][family]):
                for scope in ("call", "prepared_kernel"):
                    mean = float(index + 1)
                    status, correct = "ok", True
                    if family == "cubic" and index == 0 and scope == "prepared_kernel":
                        status = "compile_error"
                    if family == "strassen" and index == 0:
                        correct = False
                    rows.append({**candidate, "family": family, "shape_id": shape_id,
                                 "arm_id": candidate["candidate_id"], "scope": scope, "status": status,
                                 "correctness_pass": correct, "relative_l2": .001,
                                 "reference_scope": "sampled_cross_product", "group_id": "fixture",
                                 "case_id": "fixture", "timing": {"mean_ms": mean, "sample_count": 7}})
        selection = tuning.select_winners(rows, campaign, self.shapes, "N5-screen")
        selection["environment_identity"] = {"allocation_id": "test-cohort"}
        return campaign, selection, shape_id

    def test_selection_requires_accuracy_both_scopes_and_all_samples(self):
        campaign, selection, shape_id = self.selection_fixture()
        selected = selection["by_shape"][shape_id]
        for family in tuning.FAMILIES:
            expected = campaign["experiments"]["N5-screen"]["candidate_families"][family][1]
            self.assertEqual(selected[family]["candidate_id"], expected["candidate_id"])
            self.assertEqual(selected["attempt_counts"][family], 20)
            self.assertEqual(selected["eligible_candidate_counts"][family], 19)
        groups = tuning.build_groups(campaign, self.shapes, "N5-confirm", selection)
        self.assertEqual(len(groups), 1)
        self.assertEqual({a["public_arm_id"] for a in groups[0]["arms"]},
                         {"native_xla", "cubic_selected", "strassen_selected"})
        self.assertTrue(tuning.verify_selection(selection, campaign, self.shapes, "N5-confirm",
                                               {"allocation_id": "test-cohort"}))
        changed = copy.deepcopy(selection)
        changed["by_shape"][shape_id]["cubic"]["tile"] = [16, 256, 256]
        with self.assertRaisesRegex(ValueError, "outside the preregistered"):
            tuning.verify_selection(changed, campaign, self.shapes, "N5-confirm", changed["environment_identity"])

    def test_heldout_selector_rejects_seen_shapes_and_identity_changes(self):
        manifest = {"shapes": [{"id": "unseen", "m": 768, "k": 1024, "n": 3072}]}
        identity = {"allocation_id": "test"}
        selector = {"training_shapes": [[512, 512, 512]],
                    "reserved_heldout_shapes": [[768, 1024, 3072]], "source_identity": identity}
        self.assertTrue(tuning.verify_heldout_selector(selector, manifest, ["unseen"], identity))
        seen = copy.deepcopy(selector)
        seen["training_shapes"].append([768, 1024, 3072])
        with self.assertRaisesRegex(ValueError, "training geometry"):
            tuning.verify_heldout_selector(seen, manifest, ["unseen"], identity)
        with self.assertRaisesRegex(ValueError, "identities differ"):
            tuning.verify_heldout_selector(selector, manifest, ["unseen"], {"allocation_id": "other"})

    def test_confirmation_journal_preserves_classical_paired_contrast(self):
        campaign, selection, _ = self.selection_fixture()
        group = tuning.build_groups(campaign, self.shapes, "N5-confirm", selection)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            journal = tuning.TuningJournal(path, "N5-confirm")
            journal.groups = {group["group_id"]: group}
            journal.emit("case_result", group_id=group["group_id"], arm_id="strassen_basic",
                         scope="call", status="ok", comparisons=[{"reference_arm": "cubic_basic",
                                                                    "speedup_ratio_of_means": 1.05}])
            journal.close()
            row = json.loads((path / "results.jsonl").read_text())
            self.assertEqual(row["arm_id"], "strassen_selected")
            self.assertEqual(row["comparisons"][0]["reference_arm"], "cubic_selected")
            self.assertEqual(row["family"], "strassen")


class FullTileOutputAccumulator(unittest.TestCase):
    def test_full_tile_variant_preserves_exact_algebra_across_grid_and_padding(self):
        rng = np.random.default_rng(311)
        shape = (17, 259, 257)
        a_np = rng.integers(-2, 3, (shape[0], shape[1])).astype(np.float32)
        b_np = rng.integers(-2, 3, (shape[1], shape[2])).astype(np.float32)
        a, b = jnp.asarray(a_np, jnp.bfloat16), jnp.asarray(b_np, jnp.bfloat16)
        expected = a_np @ b_np
        for variant in ("plain", "output_accumulator"):
            with self.subTest(variant=variant):
                fn = make_matmul("cubic_full", shape, (16, 256, 256), variant=variant, interpret=True)
                np.testing.assert_array_equal(np.asarray(jax.jit(fn)(a, b)), expected)
                self.assertEqual(fn.metadata["accumulation_dtype"], "float32")
        with self.assertRaises(ValueError):
            make_matmul("cubic_full", shape, (16, 256, 256), variant="interleaved", interpret=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
