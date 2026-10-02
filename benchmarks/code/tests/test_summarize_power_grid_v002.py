"""Synthetic-only tests for independent near-interval replay in version 002."""
import importlib.util
import copy
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("grid_descriptive_v002", ROOT / "tools/summarize_power_grid_v002.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
BASE_SPEC = importlib.util.spec_from_file_location("grid_descriptive_v001_tests", ROOT / "tests/test_summarize_power_grid_v001.py")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


class DescriptiveV002RegressionTests(base.DescriptiveGridTests):
    """Rerun the six meaningful v001 invariants against the new implementation."""

    def setUp(self):
        self.module_patch = patch.object(base, "summary", summary)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)


class NearIntervalReplayTests(unittest.TestCase):
    def test_positive_plausible_but_corrupted_interval_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_near_interval_unit_") as temp:
            ev = summary.plot.Evidence(*base.sealed_fixture(Path(temp)))
            candidate = ev.confirmation["by_shape"]["unit_m256_n512_k128"]["cubic"]["candidates"][0]
            # All prior v001 checks accept positive ordered endpoints and the
            # matching <=1.05 flag. Exact replay must nevertheless reject it.
            candidate["scopes"]["call"]["latency_ratio_ci95"] = [.95, 1.01]
            candidate["scopes"]["call"]["certified_within_5_percent"] = True
            with self.assertRaisesRegex(ValueError, "archived interval differs from exact-seed"):
                summary.near_tile_sets(ev)

    def test_replayed_interval_uses_near_seed_and_preserves_values(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_near_interval_unit_") as temp:
            ev = summary.plot.Evidence(*base.sealed_fixture(Path(temp)))
            _, candidates = summary.near_tile_sets(ev)
            self.assertEqual(len(candidates), 6)
            for candidate in candidates:
                self.assertEqual(candidate["interval_bootstrap_seed"], ev.campaign["seed"] + 711)
                self.assertEqual(candidate["paired_round_count"], 3)
                self.assertEqual(candidate["latency_ratio_ci95"], [1., 1.])
                self.assertEqual(candidate["latency_ratio_to_frozen_winner"], 1.)
                self.assertIn("independently replayed", candidate["interval_audit"])

    def test_output_tile_count_is_logical_not_physical(self):
        result = summary.source_covariates({"m": 256, "n": 512, "k": 1024},
            {"algorithm": "strassen", "tile": [128, 256, 256],
             "kernel_metadata": {"padded_shape_mkn": [256, 1024, 512]}})
        self.assertEqual(result["parallel_output_tiles"], 4)
        self.assertEqual(result["sequential_k_panels"], 4)
        self.assertIn("not measured physical concurrency", result["parallel_output_tiles_estimand"])

    def test_nonself_candidate_interval_is_inverted_in_correct_order(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="synthetic_near_interval_unit_") as temp:
            ev = summary.plot.Evidence(*base.sealed_fixture(Path(temp)))
            shape_id = "unit_m256_n512_k128"
            reference = ev.selected(shape_id, "call", "cubic")
            candidate_row = copy.deepcopy(reference)
            candidate_row.update(candidate_id="synthetic_cubic_alternative", tile=[128, 512, 128])
            candidate_values = [5.5, 6.3, 7.0]
            candidate_row["timing"]["mean_ms"] = statistics.mean(candidate_values)
            ev.results.append(candidate_row)
            ev.samples[("confirmation", summary.plot.row_key(candidate_row))] = [
                {**candidate_row, "round": i, "elapsed_ms": value} for i, value in enumerate(candidate_values)]
            reference_values = np.asarray([r["elapsed_ms"] for r in ev.raw(reference)])
            rng = np.random.Generator(np.random.PCG64(ev.campaign["seed"] + 711))
            draws = rng.integers(0, 3, size=(2000, 3))
            ratios = reference_values[draws].mean(axis=1) / np.asarray(candidate_values)[draws].mean(axis=1)
            lo, hi = np.quantile(ratios, [.025, .975])
            expected = [float(1 / hi), float(1 / lo)]
            report = ev.confirmation["by_shape"][shape_id]["cubic"]
            item = {"candidate_id": "synthetic_cubic_alternative", "tile": [128, 512, 128],
                "group_id": reference["group_id"], "confirmed_eligible": True, "reference_available": True,
                "scopes": {"call": {"mean_ms": statistics.mean(candidate_values),
                    "latency_ratio_to_frozen_winner": statistics.mean(candidate_values) / summary.plot.mean_ms(reference),
                    "latency_ratio_ci95": expected, "certified_within_5_percent": expected[1] <= 1.05}}}
            report["candidates"].append(item)
            for percent in (1, 3, 5):
                if item["scopes"]["call"]["latency_ratio_to_frozen_winner"] <= 1 + percent / 100:
                    report["near_sets"]["call"][f"descriptive_{percent}_percent"]["tuples_bm_bn_bk"].append(item["tile"])
            if expected[1] <= 1.05:
                report["near_sets"]["call"]["pointwise_ci95_certified_5_percent"]["tuples_bm_bn_bk"].append(item["tile"])
            _, candidates = summary.near_tile_sets(ev)
            result = next(c for c in candidates if c["candidate_id"] == "synthetic_cubic_alternative")
            self.assertLess(expected[0], expected[1])
            self.assertEqual(result["latency_ratio_ci95"], expected)
            self.assertGreater(result["latency_ratio_to_frozen_winner"], 1.)


if __name__ == "__main__":
    unittest.main()
