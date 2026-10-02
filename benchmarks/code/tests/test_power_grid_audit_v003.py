"""Offline v002 audit regression: terminal per-scope status must be replayed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools/audit_power_grid_v002.py"
SPEC = importlib.util.spec_from_file_location("power_grid_audit_v002", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FrozenScopeStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grid-audit-v002-test-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.manifest = {"exploratory_shape_ids": [f"shape_{index}" for index in range(60)],
                         "shapes": [{"id": f"shape_{index}", "m": index + 1, "n": 128, "k": 256}
                                    for index in range(60)]}
        families = {family: [{"candidate_id": f"{family}_{index}",
                             "algorithm": "native" if family == "native" else "cubic_full" if family == "cubic" else "strassen",
                             "variant": "plain", "tile": None if family == "native" else [256 * (index + 1), 256, 256],
                             "compiler_options": {} if family != "native" or index == 0 else {audit.NATIVE_OPTION: (index + 1) * 16384}}
                            for index in range(audit.EXPECTED_BUDGETS[family])]
                    for family in audit.FAMILIES}
        campaign = {"experiments": {"GRID-screen": {
            "shape_ids": self.manifest["exploratory_shape_ids"], "candidate_families": families}}}
        rows = []
        for shape_id in self.manifest["exploratory_shape_ids"]:
            for family, candidates in families.items():
                for index, candidate in enumerate(candidates):
                    for scope in audit.SCOPES:
                        failed = shape_id == "shape_0" and family == "cubic" and index == 0 and scope == "prepared_kernel"
                        rows.append({**candidate, "shape_id": shape_id, "family": family,
                                     "scope": scope, "status": "numerical_failure" if failed else "ok",
                                     "correctness": {"pass": not failed},
                                     "timing": {"sample_count": 7, "mean_ms": 1 + index * .01}})
        self.run = {"directory": directory, "campaign_path": directory / "campaign.json",
                    "campaign": campaign, "environment": {"identity": {"allocation_id": "fixture"}},
                    "case_rows": rows, "planned": [], "report": {"phase": "GRID-screen"}}
        audit.save(self.run["campaign_path"], campaign)
        audit.save(directory / "source_manifest.json", {"sha256": {}})
        with (directory / "results.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        by_shape, policy = audit.replay_selection(self.run, self.manifest)
        self.selection = {"phase": "GRID-screen", "stage": "screen", "by_shape": by_shape,
                          "selection_policy": policy, "candidate_attempt_budgets": audit.EXPECTED_BUDGETS,
                          "environment_identity": self.run["environment"]["identity"],
                          "campaign_sha256": audit.sha(self.run["campaign_path"]),
                          "screen_results_sha256": audit.sha(directory / "results.jsonl"),
                          "source_manifest_sha256": audit.sha(directory / "source_manifest.json")}

    def run_selection_audit(self, selection):
        audit.save(self.run["directory"] / "selections.json", selection)
        checks = audit.Checks()
        audit.audit_screen(self.run, self.manifest, checks)
        return checks

    def test_full_status_metadata_replays_and_scientific_failure_is_retained(self):
        attempt = self.selection["by_shape"]["shape_0"]["cubic"]["candidate_attempts"][0]
        self.assertEqual(attempt["scope_status"], {"call": "ok", "prepared_kernel": "numerical_failure"})
        self.assertFalse(attempt["eligible"])
        self.assertEqual(self.selection["by_shape"]["shape_0"]["cubic"]["winner"]["candidate_id"], "cubic_1")
        checks = self.run_selection_audit(self.selection)
        self.assertEqual(checks.issues, [])

    def test_corrupted_scope_status_fails_even_when_winner_and_eligibility_are_unchanged(self):
        corrupted = copy.deepcopy(self.selection)
        corrupted["by_shape"]["shape_0"]["cubic"]["candidate_attempts"][0]["scope_status"]["prepared_kernel"] = "ok"
        checks = self.run_selection_audit(corrupted)
        self.assertEqual([issue["code"] for issue in checks.issues], ["frozen_selection_replayed"])

    def test_omitted_scope_status_is_not_silently_ignored(self):
        corrupted = copy.deepcopy(self.selection)
        del corrupted["by_shape"]["shape_0"]["cubic"]["candidate_attempts"][0]["scope_status"]
        checks = self.run_selection_audit(corrupted)
        self.assertEqual([issue["code"] for issue in checks.issues], ["frozen_selection_replayed"])


if __name__ == "__main__":
    unittest.main()
