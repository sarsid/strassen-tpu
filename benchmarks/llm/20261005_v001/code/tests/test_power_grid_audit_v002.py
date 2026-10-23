"""Version002: retain auditor checks; normalize macOS temporary-directory symlinks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "tools/audit_power_grid_v001.py"
SPEC = importlib.util.spec_from_file_location("power_grid_audit", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def numerical_row(passed=True):
    return {"status": "ok" if passed else "numerical_failure", "correctness": {
        "finite": True, "pass": passed, "relative_l2": .001 if passed else .5,
        "max_abs_error": .001, "max_abs_reference": 1,
        "reference_scope": "sampled_cross_product", "all_k_used": 7,
        "reference_input_dtype": "quantized_bf16", "reference_accumulation_dtype": "float32",
        "sample_rows": [0, 2], "sample_columns": [0, 4], "sample_count": 4}}


class PowerGridAuditTests(unittest.TestCase):
    def test_archive_root_prefers_canonical_even_with_streamed_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "artifacts").mkdir()
            (root / "source").mkdir()
            (root / "results.jsonl").write_text("streamed copy\n")
            (root / "artifacts/results.jsonl").write_text("canonical copy\n")
            (root / "source/example.py").write_text("# frozen\n")
            (root / "source.tar").write_bytes(b"archive fixture")
            audit.save(root / "artifact-manifest.json", {
                "artifacts/results.jsonl": {"sha256": audit.sha(root / "artifacts/results.jsonl"),
                                             "bytes": (root / "artifacts/results.jsonl").stat().st_size}})
            audit.save(root / "source-manifest.json", {
                "source_sha256": {"example.py": audit.sha(root / "source/example.py")},
                "archive_sha256": audit.sha(root / "source.tar")})
            checks = audit.Checks()
            self.assertEqual(audit.resolve_benchmark_directory(root, checks), (root / "artifacts").resolve())
            self.assertEqual(checks.issues, [])

    def test_hash_checks_detect_changes_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inside = root / "sealed"
            inside.mkdir()
            file = inside / "data.json"
            file.write_text("original")
            expected = audit.sha(file)
            file.write_text("changed")
            checks = audit.Checks()
            checks.verify(inside, {"data.json": expected}, "test")
            self.assertIn("sealed_sha256", {issue["code"] for issue in checks.issues})
            outside = root / "outside.json"
            outside.write_text("external")
            checks.verify(inside, {"../outside.json": audit.sha(outside)}, "test")
            self.assertIn("sealed_file_exists", {issue["code"] for issue in checks.issues})
            self.assertNotIn(str(outside), checks.verified_files)

    def test_scientific_numerical_failure_is_valid_evidence(self):
        campaign = {"correctness": {"gate": {"relative_l2_max": .02,
                     "max_abs_atol": .001, "max_abs_reference_rtol": .05}}}
        shape = {"m": 3, "n": 5, "k": 7}
        checks = audit.Checks()
        audit.numerical_contract(numerical_row(False), shape, campaign, checks, "failure")
        self.assertEqual(checks.issues, [])
        invalid = numerical_row(False)
        invalid["correctness"]["pass"] = True
        audit.numerical_contract(invalid, shape, campaign, checks, "bad-pass")
        self.assertIn("numerical_gate_recomputed", {item["code"] for item in checks.issues})

    def test_reference_axis_swap_is_caught(self):
        campaign = {"correctness": {"gate": {"relative_l2_max": .02,
                     "max_abs_atol": .001, "max_abs_reference_rtol": .05}}}
        checks = audit.Checks()
        audit.numerical_contract(numerical_row(), {"m": 3, "n": 7, "k": 5}, campaign, checks, "swapped")
        codes = {item["code"] for item in checks.issues}
        self.assertIn("reference_uses_all_k", codes)
        self.assertIn("reference_includes_boundaries", codes)

    def test_timing_preserves_failed_attempt_and_catches_missing_or_duplicate_round(self):
        checks = audit.Checks()
        audit.timing_contract({"status": "compile_error"}, [], 7, checks, "failed compile")
        self.assertEqual(checks.issues, [])
        samples = [{"round": i, "elapsed_ms": float(i + 1)} for i in range(7)]
        row = {"status": "ok", "timing": {"sample_count": 7, "mean_ms": 4,
               "median_ms": 4, "min_ms": 1, "max_ms": 7,
               "std_ms": audit.statistics.stdev(range(1, 8))}}
        audit.timing_contract(row, samples, 7, checks, "complete")
        self.assertEqual(checks.issues, [])
        audit.timing_contract(row, samples[:-1], 7, checks, "missing")
        self.assertIn("eligible_sample_count", {item["code"] for item in checks.issues})
        checks = audit.Checks()
        samples[-1]["round"] = 5
        audit.timing_contract(row, samples, 7, checks, "duplicate")
        self.assertIn("raw_round_ids", {item["code"] for item in checks.issues})

    def test_same_candidate_samples_are_not_pooled_across_groups(self):
        run = {"groups": {name: {"arms": [{"candidate_id": "winner", "arm_id": "arm"}]}
                          for name in ("headline", "finalist")},
               "samples": {("headline", "arm", "call"): [{"round": 0, "elapsed_ms": 1}],
                           ("finalist", "arm", "call"): [{"round": 0, "elapsed_ms": 10}]}}
        self.assertEqual(audit.candidate_samples(run, "finalist", "winner", "call"),
                         [{"round": 0, "elapsed_ms": 10}])
        checks = audit.Checks()
        pair = {"latency_ratio_to_frozen_winner": 1.03, "latency_ratio_ci95": [1.02, 1.04],
                "certified_within_5_percent": True}
        audit.pair_contract(pair, [{"round": 0, "elapsed_ms": 10}],
                            [{"round": 0, "elapsed_ms": 10.3}], checks, "paired", inverse=True)
        self.assertEqual(checks.issues, [])
        pair["certified_within_5_percent"] = False
        audit.pair_contract(pair, [{"round": 0, "elapsed_ms": 10}],
                            [{"round": 0, "elapsed_ms": 10.3}], checks, "badthreshold", inverse=True)
        self.assertIn("near_interval_threshold", {issue["code"] for issue in checks.issues})

    def test_frozen_selection_replay_excludes_failure_and_preserves_capped_near_set(self):
        families = {family: [{"candidate_id": f"{family}{i}", "algorithm": family,
                             "variant": "plain", "tile": None if family == "native" else [256 * (i + 1)] * 3}
                            for i in range(4)] for family in audit.FAMILIES}
        experiment = {"shape_ids": ["a"], "candidate_families": families,
                      "shortlist_top_k": 1, "shortlist_max_per_family": 2, "near_optimal_fraction": .05}
        run = {"campaign": {"experiments": {"GRID-screen": experiment}},
               "report": {"phase": "GRID-screen"}, "case_rows": []}
        for family in audit.FAMILIES:
            for i, candidate in enumerate(families[family]):
                for scope in audit.SCOPES:
                    run["case_rows"].append({**candidate, "shape_id": "a", "family": family, "scope": scope,
                        "status": "ok", "correctness": {"pass": not (family == "cubic" and i == 0)},
                        "timing": {"sample_count": 7, "mean_ms": 1 + i * .01}})
        result, _ = audit.replay_selection(run, {"shapes": [{"id": "a", "m": 3, "n": 5, "k": 7}]})
        self.assertEqual(result["a"]["shape_mkn"], [3, 7, 5])
        cubic = result["a"]["cubic"]
        self.assertEqual(cubic["winner"]["candidate_id"], "cubic1")
        self.assertEqual(len(cubic["confirmation_candidates"]), 2)
        self.assertEqual(len(cubic["screen_near_candidates"]), 3)
        self.assertFalse(cubic["screen_near_candidates"][-1]["scheduled_for_confirmation"])

    def test_tuned_native_headline_and_default_are_both_expected(self):
        default = {"candidate_id": "native_default", "compiler_options": {}}
        tuned = {"candidate_id": "native_tuned", "compiler_options": {audit.NATIVE_OPTION: 32768}}
        cubic = {"candidate_id": "cubic1"}
        strassen = {"candidate_id": "strassen1"}
        selection = {"by_shape": {"a": {
            "native": {"winner": tuned, "confirmation_candidates": [tuned, default]},
            "cubic": {"winner": cubic, "confirmation_candidates": [cubic]},
            "strassen": {"winner": strassen, "confirmation_candidates": [strassen]}}}}
        campaign = {"experiments": {"GRID-screen": {"candidate_families": {"native": [default, tuned]}}}}
        groups = audit.expected_confirmation_groups(selection, campaign, ["a"])
        self.assertEqual(set(groups), {"a__headline_confirmation"})
        self.assertEqual([candidate["candidate_id"] for _, candidate in groups["a__headline_confirmation"]],
                         ["native_tuned", "cubic1", "strassen1", "native_default"])

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "keep.txt"
            target.write_text("preserve")
            argv = [str(SOURCE), "--screen", "s", "--confirmation", "c", "--manifest", "m", "--output-dir", folder]
            with mock.patch("sys.argv", argv), mock.patch.object(audit, "audit") as invoke:
                with self.assertRaises(FileExistsError):
                    audit.main()
                invoke.assert_not_called()
            self.assertEqual(target.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
