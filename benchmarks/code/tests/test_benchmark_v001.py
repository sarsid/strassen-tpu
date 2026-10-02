"""Independent CPU utility tests; no TPU runtime or performance execution."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ml_dtypes
import numpy as np

from strassen_mm import benchmark_v001 as benchmark


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkUtilities(unittest.TestCase):
    def setUp(self):
        self.numpy_patch = mock.patch.object(benchmark, "np", np, create=True)
        self.dtype_patch = mock.patch.object(benchmark, "ml_dtypes", ml_dtypes, create=True)
        self.numpy_patch.start()
        self.dtype_patch.start()
        self.addCleanup(self.numpy_patch.stop)
        self.addCleanup(self.dtype_patch.stop)
        self.campaign = json.loads((ROOT / "configs" / "campaign_v1.json").read_text())
        self.shapes = json.loads((ROOT / "configs" / "shapes_v1.json").read_text())

    def test_preregistered_case_counts_and_n2_equal_candidate_budgets(self):
        expected = {"smoke": (3, 24), "N1": (44, 352), "N2": (72, 432),
                    "N3": (6, 96), "N4": (8, 480)}
        for phase, (group_count, record_count) in expected.items():
            with self.subTest(phase=phase):
                groups = benchmark.planned_groups(self.campaign, self.shapes, phase)
                self.assertEqual(len(groups), group_count)
                actual = sum(len(g["arms"]) * len(g["inputs"]) * len(g["scopes"]) for g in groups)
                self.assertEqual(actual, record_count)
                self.assertEqual(groups, benchmark.planned_groups(self.campaign, self.shapes, phase))
        n2 = benchmark.planned_groups(self.campaign, self.shapes, "N2")
        wanted = {tuple(t) for t in self.campaign["experiments"]["N2"]["tiles"]}
        for shape_id in self.campaign["experiments"]["N2"]["shape_ids"]:
            selected = [g for g in n2 if g["shape"]["id"] == shape_id]
            self.assertEqual({tuple(g["tile"]) for g in selected}, wanted)
            for g in selected:
                self.assertEqual({a["algorithm"] for a in g["arms"]},
                                 {"cubic_full", "cubic_quadrant", "strassen"})

    def test_five_distributions_reproduce_identical_quantized_inputs(self):
        signatures = set()
        for profile in self.campaign["experiments"]["N4"]["distributions"]:
            with self.subTest(profile=profile):
                first = benchmark.generate_inputs((13, 17, 19), profile, 531)
                second = benchmark.generate_inputs((13, 17, 19), profile, 531)
                for left, right in zip(first, second):
                    self.assertEqual(left.dtype, np.dtype(ml_dtypes.bfloat16))
                    self.assertTrue(np.isfinite(left).all())
                    np.testing.assert_array_equal(left.view(np.uint16), right.view(np.uint16))
                fingerprint = benchmark.input_fingerprint(*first)
                self.assertEqual(fingerprint, benchmark.input_fingerprint(*second))
                signatures.add(fingerprint["a_sha256"])
                if profile == "cancellation":
                    self.assertTrue(np.all(first[0][:, -1] == 0))
                    self.assertTrue(np.all(first[1][-1, :] == 0))
        # A small outlier draw can legitimately contain no selected entries.
        self.assertGreaterEqual(len(signatures), 4)

    def test_reference_uses_quantized_values_and_full_shared_dimension(self):
        source_a = np.arange(21, dtype=np.float32).reshape(3, 7) / np.float32(7.3)
        source_b = np.arange(28, dtype=np.float32).reshape(7, 4) / np.float32(11.7)
        a, b = source_a.astype(ml_dtypes.bfloat16), source_b.astype(ml_dtypes.bfloat16)
        reference, rows, cols, info = benchmark.make_reference(a, b, self.campaign, 98)
        expected = a.astype(np.float32) @ b.astype(np.float32)
        np.testing.assert_array_equal(reference, expected)
        self.assertFalse(np.array_equal(reference, source_a @ source_b))
        self.assertEqual(info["reference_scope"], "full")
        self.assertEqual(info["all_k_used"], 7)
        sampled_campaign = copy.deepcopy(self.campaign)
        sampled_campaign["memory"]["max_reference_output_elements"] = 1
        sampled_campaign["correctness"]["sample_rows"] = 2
        sampled_campaign["correctness"]["sample_columns"] = 2
        sampled, rows, cols, info = benchmark.make_reference(a, b, sampled_campaign, 98)
        self.assertEqual(rows.tolist(), [0, 2])
        self.assertEqual(cols.tolist(), [0, 3])
        np.testing.assert_allclose(sampled, expected[np.ix_(rows, cols)], rtol=1e-6, atol=1e-6)
        self.assertEqual(info["reference_scope"], "sampled_cross_product")
        self.assertEqual(info["all_k_used"], 7)

    def test_nonfinite_results_remain_serializable_failures(self):
        ref = np.ones((2, 2), np.float32)
        info = {"normwise_denominator": 4.0, "reference_scope": "full", "sample_count": 4}
        result = benchmark.numeric_metrics(np.full((2, 2), np.nan), False,
                                           ref, info, self.campaign)
        self.assertFalse(result["pass"])
        self.assertFalse(result["finite"])
        self.assertIsNone(result["relative_l2"])
        json.dumps(benchmark.json_safe(result), allow_nan=False)

    def test_phase_field_in_completion_event_does_not_break_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            journal = benchmark.Journal(path, "smoke")
            journal.emit("case_result", status="numerical_failure", arm_id="strassen_basic")
            journal.emit("run_complete", phase="smoke", completed=True)
            journal.close()
            records = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
            self.assertEqual([r["sequence"] for r in records], [1, 2])
            self.assertEqual(records[-1]["phase"], "smoke")
            self.assertTrue(records[-1]["completed"])
            self.assertEqual(journal.status_counts["numerical_failure"], 1)
            with self.assertRaises(FileExistsError):
                benchmark.Journal(path, "smoke")

    def test_setup_failure_still_seals_summary_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            out = parent / "new_run"
            # A missing campaign is an intentional pre-JAX failure, making
            # this an offline test of the real main/finalization path.
            argv = ["--campaign", str(parent / "missing.json"), "--phase", "smoke",
                    "--output-dir", str(out), "--allocation-id", "test-only"]
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main(argv)
            self.assertEqual(code, 1)
            summary = json.loads((out / "summary.json").read_text())
            manifest = json.loads((out / "artifact_manifest.json").read_text())
            self.assertFalse(summary["completed"])
            self.assertEqual(summary["error"]["type"], "FileNotFoundError")
            for name, digest in manifest["sha256"].items():
                self.assertEqual(benchmark.digest_file(out / name), digest)
            events = [json.loads(line)["event"] for line in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual(events[-2:], ["run_error", "run_complete"])

    def test_identity_detects_quantization_package_or_device_changes(self):
        identity = {"allocation_id": "allocation-A", "colab_endpoint": "allocation-A",
                    "hostname": "host-A", "boot_id": "boot-A", "runtime_flags": {},
                    "versions": {"jax": "0.7.2", "ml_dtypes": "0.6.0"},
                    "devices": [{"kind": "TPU v5 lite", "id": 0, "coords": [0, 0, 0]}]}
        current = {"identity": identity, "qualified_single_v5e": True,
                   "identity_unknown_fields": []}
        with tempfile.TemporaryDirectory() as folder:
            prior_path = Path(folder) / "environment.json"
            prior_path.write_text(json.dumps(current))
            self.assertEqual(benchmark.verify_identity(current, prior_path, self.campaign)["status"], "matched")
            for mutation in ("version", "device"):
                changed = copy.deepcopy(current)
                if mutation == "version":
                    changed["identity"]["versions"]["ml_dtypes"] = "different"
                else:
                    changed["identity"]["devices"][0]["coords"] = [1, 0, 0]
                with self.subTest(mutation=mutation):
                    with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                        benchmark.verify_identity(changed, prior_path, self.campaign)
            wrong_device = {**current, "qualified_single_v5e": False}
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                benchmark.verify_identity(wrong_device, prior_path, self.campaign)

    def test_statistics_pair_rounds_and_use_ratio_of_means(self):
        reference = [{"round": 0, "elapsed_ms": 2.0}, {"round": 1, "elapsed_ms": 100.0},
                     {"round": 2, "elapsed_ms": 6.0}]
        candidate = [{"round": 0, "elapsed_ms": 1.0}, {"round": 2, "elapsed_ms": 3.0}]
        pair = benchmark.paired_comparison(reference, candidate, 123)
        self.assertEqual(pair["paired_rounds"], [0, 2])
        self.assertEqual(pair["speedup_ratio_of_means"], 2.0)
        self.assertEqual(pair["speedup_ci95"], [2.0, 2.0])
        self.assertEqual(pair["time_reduction_fraction"], .5)
        self.assertEqual(pair, benchmark.paired_comparison(reference, candidate, 123))
        stats = benchmark.basic_statistics(candidate)
        self.assertEqual(stats["mean_ms"], 2.0)
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertAlmostEqual(stats["std_ms"], np.sqrt(2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
