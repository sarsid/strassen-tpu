"""Offline selection/provenance-isolation tests; no TPU or JAX initialization."""
import copy
import unittest
from unittest import mock

from strassen_mm import benchmark_n6_supplement_v001 as supplement


def fixture():
    shapes = {"small": [512, 512, 512], "middle": [2048, 2048, 2048],
              "large_b": [4096, 4096, 4096], "large_a": [4096, 4096, 4096],
              "largest_ineligible": [8192, 8192, 8192]}
    choices = {"phase": "N5-screen", "by_shape": {name: {"shape_mkn": shape} for name, shape in shapes.items()}}
    rows = []
    for name in shapes:
        for arm in ("native_xla", "cubic_selected", "strassen_selected"):
            row = {"phase": "N5-confirm", "scope": "call", "shape_id": name, "arm_id": arm,
                   "status": "ok", "correctness": {"pass": True, "finite": True}, "timing": {"sample_count": 30}}
            if arm == "strassen_selected":
                row["comparisons"] = [{"reference_arm": ref, "valid_numerical_comparison": True,
                                       "speedup_ci95": [.99, 1.1] if name == "largest_ineligible" and ref == "native_xla" else [1.01, 1.1]}
                                      for ref in ("native_xla", "cubic_selected")]
            rows.append(row)
    return rows, choices


class SupplementTests(unittest.TestCase):
    def test_joint_win_gate_volume_order_tie_break_and_no_input_mutation(self):
        rows, choices = fixture()
        before = copy.deepcopy((rows, choices))
        selected = supplement.large_joint_wins(rows, choices)
        self.assertEqual([r["shape_id"] for r in selected["representatives"]], ["large_a", "large_b"])
        self.assertEqual(selected["available_class_counts"], {"joint_win": 4})
        self.assertTrue(selected["post_hoc"])
        self.assertFalse(selected["independent_confirmation"])
        self.assertEqual((rows, choices), before)

    def test_numeric_failure_missing_samples_and_invalid_ci_are_excluded(self):
        rows, choices = fixture()
        for row in rows:
            if row["shape_id"] == "large_a" and row["arm_id"] == "native_xla":
                row["correctness"]["pass"] = False
            if row["shape_id"] == "large_b" and row["arm_id"] == "cubic_selected":
                row["timing"]["sample_count"] = 29
        selected = supplement.large_joint_wins(rows, choices)
        self.assertEqual([r["shape_id"] for r in selected["representatives"]], ["middle", "small"])
        for row in rows:
            if row["shape_id"] == "middle" and row["arm_id"] == "strassen_selected":
                row["comparisons"][0]["speedup_ci95"] = [float("nan"), 1.1]
        with self.assertRaisesRegex(ValueError, "Fewer than two"):
            supplement.large_joint_wins(rows, choices)

    def test_n7_and_duplicate_results_are_rejected(self):
        rows, choices = fixture()
        changed = copy.deepcopy(choices); changed["phase"] = "N7-screen"
        with self.assertRaises(ValueError): supplement.large_joint_wins(rows, changed)
        changed_rows = copy.deepcopy(rows); changed_rows[0]["phase"] = "N7-confirm"
        with self.assertRaises(ValueError): supplement.large_joint_wins(changed_rows, choices)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            supplement.large_joint_wins(rows + [rows[0]], choices)

    def test_wrapper_restores_all_original_hooks_on_failure(self):
        previous = (supplement.original.ProfileRunner, supplement.original.representative_shapes,
                    supplement.original.EvidenceJournal)
        args = ["--phase", "N6", "--campaign", "unused", "--output-dir", "unused",
                "--expected-identity", "unused", "--selection", "unused", "--confirmation", "unused",
                "--prior-profile", "unused", "--allocation-id", "unused"]
        with mock.patch.object(supplement, "validate_inputs", return_value={}):
            with mock.patch.object(supplement.original, "main", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    supplement.main(args)
        self.assertEqual((supplement.original.ProfileRunner, supplement.original.representative_shapes,
                          supplement.original.EvidenceJournal), previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
