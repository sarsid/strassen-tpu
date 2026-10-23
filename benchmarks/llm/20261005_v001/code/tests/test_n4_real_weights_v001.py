"""Offline artifact and plan checks; no TPU execution or timing."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ml_dtypes
import numpy as np

from strassen_mm import benchmark_v001 as base
from strassen_mm import n4_real_weights_v001 as supplement


ROOT = Path(__file__).resolve().parents[1]


class RealWeightSupplement(unittest.TestCase):
    def setUp(self):
        numpy_patch = mock.patch.object(base, "np", np, create=True)
        dtype_patch = mock.patch.object(base, "ml_dtypes", ml_dtypes, create=True)
        numpy_patch.start()
        dtype_patch.start()
        self.addCleanup(numpy_patch.stop)
        self.addCleanup(dtype_patch.stop)
        self.campaign_path = ROOT / "configs/campaign_real_weights_v1.json"
        self.campaign, self.config, _ = supplement.load_campaign(self.campaign_path)
        self.manifest_path = (self.campaign_path.parent / self.config["weights_manifest"]).resolve()

    def test_loaded_transposes_preserve_every_checkpoint_bf16_bit(self):
        weights, provenance = supplement.load_verified_weights(
            self.manifest_path, self.config["expected_model_id"], self.config["expected_revision"])
        self.assertEqual(len(weights), 2)
        manifest = json.loads(self.manifest_path.read_text())
        enriched = {item["name"]: item for item in provenance["tensors"]}
        for item in manifest["tensors"]:
            with self.subTest(tensor=item["name"]):
                stored_bits = np.fromfile(self.manifest_path.parent / item["path"], dtype="<u2").reshape(item["shape"])
                b = weights[item["name"]]
                self.assertEqual(b.dtype, np.dtype(ml_dtypes.bfloat16))
                self.assertTrue(b.flags.c_contiguous)
                np.testing.assert_array_equal(b.view(np.uint16), stored_bits.T)
                expected_hash = hashlib.sha256(memoryview(b.view(np.uint8))).hexdigest()
                self.assertEqual(enriched[item["name"]]["b_sha256_after_transpose"], expected_hash)
        self.assertEqual(provenance["revision"], self.config["expected_revision"])

    def test_plan_is_48_accuracy_cases_with_unchanged_gates(self):
        shapes_path = self.campaign_path.parent / self.config["shape_manifest"]
        shapes = json.loads(shapes_path.read_text())
        groups = base.planned_groups(self.campaign, shapes, "N4")
        self.assertEqual(len(groups), 4)
        count = sum(len(g["arms"]) * len(g["inputs"]) * len(g["scopes"]) for g in groups)
        self.assertEqual(count, 48)
        core = json.loads((ROOT / "configs/campaign_v1.json").read_text())
        self.assertEqual(self.campaign["correctness"], core["correctness"])
        self.assertEqual(self.campaign["precision"], core["precision"])
        for group in groups:
            self.assertEqual(group["scopes"], ["call"])
            self.assertIsNone(group["timing"])
            self.assertEqual(group["tile"], [1024, 1024, 512])
            self.assertEqual(len(group["inputs"]), 3)
            self.assertEqual({a["variant"] for a in group["arms"]}, {"plain"})
            self.assertIn(group["shape"]["tensor_name"], supplement.EXPECTED_TENSORS)
        with tempfile.TemporaryDirectory() as folder:
            journal = supplement.WeightJournal(Path(folder))
            journal.emit("case_result", status="ok", eligible_for_speedup_claim=True)
            journal.close()
            row = json.loads((Path(folder) / "results.jsonl").read_text())
            self.assertTrue(row["numerical_eligible"])
            self.assertFalse(row["eligible_for_speedup_claim"])

    def test_bad_file_hash_and_different_checkpoint_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tensor.bf16"
            path.write_bytes(b"\x80\x3f")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                supplement.verified_file(Path(folder), path.name, "0" * 64)
        with self.assertRaisesRegex(ValueError, "revision does not match"):
            supplement.load_verified_weights(self.manifest_path, self.config["expected_model_id"], "0" * 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
