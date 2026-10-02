"""Cohort integrity and honest validation checks; synthetic data, no TPU."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("region_margin", ROOT / "tools/fit_region_margin_v001.py")
margin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(margin)


class MarginChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = [
            {"candidate_id": "native_default", "family": "native", "algorithm": "native", "variant": "plain", "tile": None, "compiler_options": {}},
            {"candidate_id": "native_tuned", "family": "native", "algorithm": "native", "variant": "plain", "tile": None, "compiler_options": {"xla_tpu_scoped_vmem_limit_kib": 65536}},
            {"candidate_id": "strassen_a", "family": "strassen", "algorithm": "strassen", "variant": "interleaved_output_accumulator", "tile": [256, 256, 256], "compiler_options": {}},
            {"candidate_id": "strassen_b", "family": "strassen", "algorithm": "strassen", "variant": "interleaved_output_accumulator", "tile": [512, 512, 512], "compiler_options": {}}]
        self.shapes = [{"id": f"s{i}", "m": 256 * 2 ** i, "n": 512, "k": 1024,
                        "sampling_role": ("broad", "focused", "anchor")[i % 3]} for i in range(12)]
        self.manifest = {"cohort_id": "new_cohort", "shapes": self.shapes,
                         "exploratory_shape_ids": [s["id"] for s in self.shapes],
                         "holdout_shape_ids": ["protected"]}
        self.config = {"cohort_id": "new_cohort", "campaign_id": "new_campaign", "seed": 42,
                       "timing": {"screen": {"repeats": 7}}, "experiments": {"GRID-screen": {
                           "shape_ids": self.manifest["exploratory_shape_ids"], "timing": "screen",
                           "candidate_families": {f: [c for c in self.candidates if c["family"] == f]
                                                  for f in ("native", "strassen")}}}}
        self.environment = {"campaign_id": "new_campaign", "identity": {"allocation_id": "allocation_a", "boot_id": "boot_a"}}
        for name, obj in (("manifest", self.manifest), ("config", self.config), ("environment", self.environment)):
            self.save_json(name, obj)
        self.rows = [{"event": "run_start", "phase": "GRID-screen", "allocation_id": "allocation_a",
                      "source_manifest": {"sha256": {"manifest": margin.sha(self.path("manifest")),
                                                       "config": margin.sha(self.path("config"))}}}]
        for i, shape in enumerate(self.shapes):
            for j, candidate in enumerate(self.candidates):
                self.rows.append({"event": "case_result", "phase": "GRID-screen", "scope": "call", **candidate,
                                  "shape_id": shape["id"], "shape_mkn": [shape[d] for d in ("m", "k", "n")],
                                  "group_id": f"{shape['id']}__batch{j}", "status": "ok",
                                  "correctness": {"pass": True}, "eligible_for_speedup_claim": True,
                                  "distribution": "gaussian", "seed": 42,
                                  "timing": {"mean_ms": math.exp(-j * .04 + i * .002 * j), "sample_count": 7}})
        self.rows.append({"event": "run_complete", "phase": "GRID-screen", "completed": True})
        self.save_rows()

    def path(self, name):
        return self.root / f"{name}.json"

    def save_json(self, name, value):
        self.path(name).write_text(json.dumps(value))

    def save_rows(self, rows=None, name="screen"):
        self.path(name).write_text("\n".join(json.dumps(r) for r in (self.rows if rows is None else rows)))

    def load(self):
        return margin.load_dataset(self.path("manifest"), self.path("config"), self.path("screen"),
                                   self.path("environment"), "new_cohort", "allocation_a")

    def test_complete_cohort_loads_with_unpaired_point_ratio_label(self):
        data = self.load()
        self.assertEqual(data["y"].shape, (12, 4))
        np.testing.assert_array_equal(data["y"][:, 0], 0.)
        self.assertEqual(data["provenance"]["screen_ratio_pairing"], "unpaired_cross_candidate_group_point_estimates")
        self.assertFalse(data["provenance"]["cohort_pooling"])

    def test_mixed_allocations_cohorts_and_concatenated_runs_rejected(self):
        original = copy.deepcopy(self.rows)
        variants = [original + [original[0]], copy.deepcopy(original), copy.deepcopy(original)]
        variants[1][1]["allocation_id"] = "allocation_b"
        variants[2][1]["cohort_id"] = "old_cohort"
        for rows in variants:
            with self.subTest(kind=rows[1].get("allocation_id", rows[1].get("cohort_id", "concat"))):
                self.save_rows(rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_old_artifacts_cannot_be_relabelled_by_cli(self):
        del self.manifest["cohort_id"]
        self.save_json("manifest", self.manifest)
        with self.assertRaisesRegex(ValueError, "explicitly declare"):
            self.load()

    def test_wrong_snapshot_or_environment_cannot_be_attached(self):
        self.environment["identity"]["allocation_id"] = "other"
        self.save_json("environment", self.environment)
        with self.assertRaisesRegex(ValueError, "Environment allocation"):
            self.load()
        self.environment["identity"]["allocation_id"] = "allocation_a"
        self.save_json("environment", self.environment)
        self.config["seed"] = 43
        self.save_json("config", self.config)
        with self.assertRaisesRegex(ValueError, "snapshot hashes"):
            self.load()

    def test_missing_duplicate_failed_and_different_seed_rows_rejected(self):
        original = copy.deepcopy(self.rows)
        cases = [original[:1] + original[2:], original + [original[1]], copy.deepcopy(original), copy.deepcopy(original)]
        cases[2][1]["correctness"]["pass"] = False
        cases[3][1]["seed"] = 999
        for rows in cases:
            self.save_rows(rows)
            with self.assertRaises(ValueError):
                self.load()

    def test_holdout_rows_are_rejected_even_with_an_ordinary_call_result(self):
        self.rows[1]["shape_id"] = "protected"
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "holdout"):
            self.load()

    def test_holdout_geometry_cannot_be_renamed_into_training(self):
        self.manifest["protected_legacy_holdout_shapes_mnk"] = [[self.shapes[0][d] for d in ("m", "n", "k")]]
        self.save_json("manifest", self.manifest)
        self.rows[0]["source_manifest"]["sha256"]["manifest"] = margin.sha(self.path("manifest"))
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "under an alias"):
            self.load()

    def test_tile_features_and_axis_order_are_shape_only(self):
        c = self.candidates[2]
        f = dict(zip(margin.FEATURE_NAMES, margin.features(1025, 513, 257, c)))
        self.assertAlmostEqual(f["log2_output_tiles"], math.log2(5 * 3))
        self.assertEqual(f["log2_k_panels"], 1.)
        self.assertAlmostEqual(f["log2_padding_volume_ratio"], math.log2(1280 * 768 * 512 / (1025 * 513 * 257)))
        altered = dict(c, winner=True, measured_ms=0., candidate_id="unused_label")
        self.assertEqual(margin.features(1025, 513, 257, c), margin.features(1025, 513, 257, altered))

    def test_blocks_keep_boundaries_and_permutations_together(self):
        variants = [{"m": 4095, "n": 8192, "k": 2048}, {"m": 8192, "n": 2048, "k": 4097}]
        self.assertEqual(margin.block_key(variants[0]), margin.block_key(variants[1]))
        folds = margin.blocked_folds(self.shapes + variants, 3)
        self.assertEqual(folds[-2], folds[-1])

    def test_test_labels_do_not_change_fit_or_native_baseline(self):
        data = self.load()
        x = np.array([[margin.features(s["m"], s["n"], s["k"], c) for c in data["candidates"]] for s in data["shapes"]])
        train = np.arange(8)
        first = margin.fit(x, data["y"], train)
        native = margin.fixed_native(data["y"], train, data["candidates"])
        data["y"][8:, 0] = 1000
        self.assertEqual(first, margin.fit(x, data["y"], train))
        self.assertEqual(native, margin.fixed_native(data["y"], train, data["candidates"]))

    def test_native_fallback_requires_margin_over_selected_native(self):
        p = np.array([[0., .10, .11, .09], [0., .10, .20, .19]])
        selected, shortlist = margin.choose(p, 1, self.candidates)
        self.assertEqual(selected.tolist(), [1, 2])
        self.assertEqual(shortlist.tolist(), [[2, 3], [2, 3]])

    def test_evaluation_keeps_sampling_roles_separate_and_holdouts_unused(self):
        model, report = margin.evaluate(self.load(), nfold=3)
        self.assertEqual(set(report["by_sampling_role"]), {"anchor", "broad", "focused"})
        self.assertEqual(report["heldout_shapes_used"], [])
        self.assertFalse(report["confirmation_used_for_fit"])
        self.assertEqual(len(report["out_of_fold_selections"]), 12)
        self.assertEqual(len(model["models"]), 4)
        json.dumps(report, allow_nan=False)

    def test_frozen_model_can_predict_without_observed_timing_lookup(self):
        model, _ = margin.evaluate(self.load(), nfold=3)
        result = margin.select_shape(model, 768, 512, 1024)
        self.assertEqual(result["status"], "experimental_prediction")
        self.assertGreater(result["nearest_training_log2_distance"], 0.)
        self.assertEqual(len(result["strassen_shortlist"]), 2)
        model["memory_budget_bytes"] = 1
        result = margin.select_shape(model, 768, 512, 1024)
        self.assertIsNone(result["selected_candidate"])

    def test_metrics_distinguish_dispatched_performance_and_shortlist_oracle(self):
        y = np.array([[0., 0., math.log(2.), math.log(.5)]])
        result = margin.metrics(y, np.array([3]), np.array([0]), np.array([[2, 3]]), self.candidates)
        self.assertAlmostEqual(result["geomean_speedup_vs_training_selected_fixed_native"], .5)
        self.assertAlmostEqual(result["shortlist_oracle_geomean_speedup_vs_fixed_native"], 2.)
        self.assertAlmostEqual(result["geomean_regret_vs_screen_catalog_oracle"], 4.)

    def test_confirmation_does_not_compare_across_timed_groups(self):
        dataset = self.load()
        rows = [dict(r, phase="GRID-confirm") for r in copy.deepcopy(self.rows)]
        self.save_rows(rows, "confirm")
        action = {"shape_id": "s0", "selected_candidate_id": "strassen_a", "fixed_native_candidate_id": "native_tuned"}
        result = margin.confirmation_check(self.path("confirm"), self.path("environment"), self.path("manifest"),
                                            self.path("config"), dataset, [action])
        self.assertFalse(result["rows"][0]["covered"])
        next(r for r in rows if r.get("shape_id") == "s0" and r.get("candidate_id") == "strassen_a")["group_id"] = "s0__batch1"
        self.save_rows(rows, "confirm")
        result = margin.confirmation_check(self.path("confirm"), self.path("environment"), self.path("manifest"),
                                            self.path("config"), dataset, [action])
        self.assertTrue(result["rows"][0]["covered"])

    def test_outputs_are_never_overwritten(self):
        target = self.path("result")
        margin.write_new(target, {"original": 1})
        with self.assertRaises(FileExistsError):
            margin.write_new(target, {"replacement": 2})
        self.assertEqual(json.loads(target.read_text()), {"original": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
