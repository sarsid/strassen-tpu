"""Offline checks of probability sampling, sealed holdouts and new-only outputs."""
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_region_grid_v001 as region


class RegionGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.population = region.build_manifest()
        cls.history, cls.old_holdouts, _ = region.historical_inventory()

    def test_uniform_draw_matches_declared_population_seed_and_probability(self):
        design = self.manifest["broad_sampling_design"]
        values = design["dimension_values"]
        self.assertEqual(values[0], 256)
        self.assertEqual(values[-1], 131072)
        self.assertEqual(len(self.population), len(values) ** 3)
        points = [region.point(row) for row in self.population]
        self.assertEqual(points, sorted(set(points)))
        eligible = [region.point(row) for row in self.population if row["eligible_for_broad"]]
        drawn = [region.point(row) for row in self.manifest["shapes"] if row["sampling_role"] == "broad"]
        self.assertEqual(drawn, random.Random(region.SEED).sample(eligible, 60))
        self.assertEqual(design["eligible_population_size"], len(eligible))
        self.assertEqual(design["inclusion_probability"], 60 / len(eligible))
        self.assertAlmostEqual(design["inclusion_probability"] * design["inverse_probability_weight"], 1)
        for row in self.population:
            p = region.point(row)
            expected = (row["conservative_pair_buffer_bytes"] <= region.geometry.BUDGET_BYTES
                        and p[0] * p[1] * p[2] <= region.MAX_BROAD_VOLUME
                        and p not in self.history)
            self.assertEqual(row["eligible_for_broad"], expected)
        for row in self.manifest["shapes"]:
            if row["sampling_role"] == "broad":
                self.assertEqual(row["broad_inclusion_probability"], 60 / len(eligible))
            else:
                self.assertIsNone(row["broad_inclusion_probability"])

    def test_roles_have_unique_geometries_and_distinct_untouched_holdouts(self):
        shapes = self.manifest["shapes"]
        self.assertEqual(len(shapes), 140)
        self.assertEqual(len({row["id"] for row in shapes}), 140)
        self.assertEqual(len({region.point(row) for row in shapes}), 140)
        self.assertEqual([len(self.manifest[key]) for key in (
            "broad_shape_ids", "focused_shape_ids", "anchor_shape_ids", "holdout_shape_ids")], [60, 48, 12, 20])
        self.assertEqual(len(self.manifest["exploratory_shape_ids"]), 120)
        self.assertFalse(set(self.manifest["exploratory_shape_ids"]) & set(self.manifest["holdout_shape_ids"]))
        self.assertFalse({region.point(row) for row in shapes} & {region.point(row) for row in self.old_holdouts})
        new_points = {region.point(row) for row in shapes if row["sampling_role"] != "anchor"}
        self.assertFalse(new_points & self.history)
        self.assertEqual({region.point(row) for row in shapes if row["sampling_role"] == "anchor"}, set(region.ANCHORS))
        self.assertTrue({(8192, 8192, 8192), (16384, 16384, 16384),
                         (2048, 2048, 16384), (2048, 2048, 131072)} <= set(region.ANCHORS))
        self.assertEqual(self.manifest["new_holdout_shape_ids"], self.manifest["holdout_shape_ids"])
        self.assertEqual({tuple(p) for p in self.manifest["protected_legacy_holdout_shapes_mnk"]},
                         {region.point(row) for row in self.old_holdouts})
        remaining = [region.point(row) for row in self.population if row["eligible_for_broad"]
                     and region.point(row) not in {region.point(s) for s in shapes if s["stage"] == "exploratory"}]
        self.assertEqual([region.point(row) for row in shapes if row["stage"] == "holdout"],
                         random.Random(region.SEED + 1).sample(remaining, 20))

    def test_actual_candidate_padding_is_within_budget_and_boundaries_are_covered(self):
        expected_tiles = {tuple(c["tile"]) for family in ("cubic", "strassen")
                          for c in region.configuration.candidates()[family]}
        self.assertEqual(set(region.TILES), expected_tiles)
        self.assertEqual(sum(len(v) for v in region.configuration.candidates().values()), 52)
        shapes = self.manifest["shapes"]
        for row in shapes:
            estimate = region.geometry.pair_buffer_estimate(*region.point(row), tiles=region.TILES)
            self.assertEqual(row["conservative_pair_buffer_bytes"], estimate)
            self.assertLessEqual(estimate, region.geometry.BUDGET_BYTES)
        counts = region.Counter(row["family"] for row in shapes if row["sampling_role"] == "focused")
        self.assertEqual(dict(counts), {"focused_square_scale": 8, "focused_axis_permutations": 12,
                                       "focused_long_k": 12, "focused_tile_boundaries": 16})
        boundaries = {region.point(row) for row in shapes if row["family"] == "focused_tile_boundaries"}
        for axis in range(3):
            for delta in (-1, 1):
                p = [8192] * 3
                p[axis] += delta
                self.assertIn(tuple(p), boundaries)
        self.assertFalse(any(row["on_lattice"] for row in shapes if row["family"] == "focused_tile_boundaries"))

    def test_campaign_runs_only_exploratory_geometries_in_its_fresh_cohort(self):
        campaign = region.build_campaign(self.manifest)
        self.assertEqual(campaign["cohort_id"], self.manifest["cohort_id"])
        self.assertNotEqual(campaign["campaign_id"], "power_midpoint_sample_v001")
        self.assertTrue(campaign["region_study"]["new_cohort"])
        self.assertFalse(campaign["region_study"]["pooled_with_prior"])
        self.assertEqual(campaign["shape_manifest"], "sampled_shapes.json")
        for phase in ("GRID-smoke", "GRID-screen", "GRID-confirm"):
            chosen = campaign["experiments"][phase]["shape_ids"]
            self.assertTrue(set(chosen) <= set(self.manifest["exploratory_shape_ids"]))
            self.assertFalse(set(chosen) & set(self.manifest["holdout_shape_ids"]))
            self.assertEqual(campaign["experiments"][phase]["candidate_families"], region.configuration.candidates())
        self.assertEqual(campaign["experiments"]["GRID-screen"]["shape_ids"], self.manifest["exploratory_shape_ids"])
        self.assertEqual(campaign["experiments"]["GRID-confirm"]["shape_ids"], self.manifest["exploratory_shape_ids"])

    def test_generated_bundle_reproduces_hashes_and_refuses_any_overwrite(self):
        old_path = ROOT / "configs/generated_power_grid_v001/sampled_shapes.json"
        old_bytes = old_path.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fresh"
            summary = region.generate(output)
            self.assertEqual(summary["planned_screen_shape_candidate_attempts"], 6240)
            self.assertEqual(summary["new_geometry_count"], 128)
            self.assertEqual(summary["protected_legacy_holdout_count"], 12)
            manifest = json.loads((output / "sampled_shapes.json").read_text())
            bundle = json.loads((output / "bundle_manifest.json").read_text())
            for name, expected in bundle["sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            self.assertEqual((output / manifest["old_holdout_source_file"]).read_bytes(), old_bytes)
            self.assertEqual(hashlib.sha256(old_bytes).hexdigest(), manifest["old_holdout_source_sha256"])
            with (output / "broad_population_preflight.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(self.population))
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                region.generate(output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(old_path.read_bytes(), old_bytes)
            second = Path(temp) / "second"
            region.generate(second)
            self.assertEqual(before, {p.name: p.read_bytes() for p in second.iterdir()})


if __name__ == "__main__":
    unittest.main()
