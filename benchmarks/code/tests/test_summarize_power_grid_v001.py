"""Descriptive-analysis unit tests; fake data never enter scientific reports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("grid_descriptive", ROOT / "tools/summarize_power_grid_v001.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
FIXTURE_SPEC = importlib.util.spec_from_file_location("grid_synthetic_fixture", ROOT / "tests/test_plot_power_grid_v001.py")
fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture)


def sealed_fixture(parent):
    screen, confirm, manifest = fixture.fixture(parent)
    for directory in (screen, confirm):
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        groups = {}
        for row in rows:
            if row["event"] != "case_result":
                continue
            group = groups.setdefault(row["group_id"], {"group_id": row["group_id"], "arms": [],
                "scopes": ["call", "prepared_kernel"], "inputs": [{"distribution": row["distribution"], "seed": row["seed"]}]})
            if row["candidate_id"] not in {a["candidate_id"] for a in group["arms"]}:
                group["arms"].append({"candidate_id": row["candidate_id"]})
        fixture.dump(directory / "planned_cases.json", list(groups.values()))
        data = json.loads((directory / "summary.json").read_text())
        data.update(completed_groups=list(groups), planned_group_count=len(groups))
        fixture.dump(directory / "summary.json", data)
    report = json.loads((confirm / "confirmation.json").read_text())
    shape_id = "unit_m256_n512_k128"
    for family in summary.plot.FAMILIES:
        result = report["by_shape"][shape_id][family]
        winner = result["frozen_winner"]
        winning_mean = {"native": 7., "cubic": 6., "strassen": 4.}[family]
        candidate = {**winner, "group_id": shape_id + "__headline_confirmation", "confirmed_eligible": True,
                     "reference_available": True, "scopes": {scope: {"mean_ms": winning_mean,
                        "latency_ratio_to_frozen_winner": 1., "latency_ratio_ci95": [1., 1.],
                        "certified_within_5_percent": True} for scope in summary.plot.SCOPES}}
        result["candidates"] = [candidate]
        projection = {"tuples_bm_bn_bk": [winner["tile"]] if winner["tile"] else [], "axis_values": {}}
        result["near_sets"] = {scope: {name: projection for name in
            ("descriptive_1_percent", "descriptive_3_percent", "descriptive_5_percent", "pointwise_ci95_certified_5_percent")}
            for scope in summary.plot.SCOPES}
    fixture.dump(confirm / "confirmation.json", report)
    for directory in (screen, confirm):
        fixture.dump(directory / "artifact_manifest.json", {"sha256": {
            str(p.relative_to(directory)): summary.plot.sha256(p) for p in directory.rglob("*") if p.is_file()}})
    return screen, confirm, manifest


class DescriptiveGridTests(unittest.TestCase):
    def test_fixed_tile_source_count_ratio_and_padding_counterexample(self):
        shape = {"m": 256, "n": 512, "k": 1024}
        common = {"tile": [256, 512, 256], "kernel_metadata": {"padded_shape_mkn": [256, 1024, 512]}}
        cubic = summary.source_covariates(shape, {**common, "algorithm": "cubic_full"})
        strassen = summary.source_covariates(shape, {**common, "algorithm": "strassen"})
        self.assertEqual(strassen["estimated_padded_dot_flops"] / cubic["estimated_padded_dot_flops"], 7 / 8)
        vector_increment = strassen["estimated_vector_element_operations"] - cubic["estimated_vector_element_operations"]
        saved_dot_work = cubic["estimated_padded_dot_flops"] - strassen["estimated_padded_dot_flops"]
        self.assertEqual(vector_increment / saved_dot_work, 5 / 256 + 5 / 512 + 8 / 256)
        padded = summary.source_covariates(shape, {"algorithm": "strassen", "tile": [512, 512, 256],
            "kernel_metadata": {"padded_shape_mkn": [512, 1024, 512]}})
        self.assertEqual(padded["padding_m_ratio"], 2)
        self.assertEqual(padded["estimated_padded_dot_flops"] / cubic["estimated_padded_dot_flops"], 1.75)

    def test_native_hidden_work_is_unknown_not_zero(self):
        value = summary.source_covariates({"m": 2, "n": 3, "k": 5}, {"algorithm": "native", "tile": None})
        self.assertIsNone(value["estimated_padded_dot_flops"])
        self.assertIsNone(value["estimated_vector_element_operations"])
        self.assertEqual(value["original_input_and_one_output_bytes"], 74)
        with self.assertRaisesRegex(ValueError, "Padded metadata"):
            summary.source_covariates({"m": 256, "n": 512, "k": 1024}, {"algorithm": "strassen", "tile": [256, 512, 256],
                "kernel_metadata": {"padded_shape_mkn": [256, 512, 1024]}})

    def test_sealed_fixture_completeness_discrete_sets_and_native_presets(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as temp:
            paths = sealed_fixture(Path(temp))
            ev = summary.plot.Evidence(*paths)
            self.assertIn("canonical seals", summary.audited_sources(ev))
            counts = summary.planned_completeness(ev)
            self.assertEqual(counts["confirmation"]["scoped_outcomes"], 10)
            findings = summary.describe(ev)
            native = next(r for r in findings["near_tile_sets"] if r["family"] == "native")
            self.assertEqual(native["candidate_count"], 1)
            self.assertEqual(native["measured_tuple_count"], 0)
            cubic = next(r for r in findings["near_tile_sets"] if r["family"] == "cubic")
            self.assertEqual(cubic["tuples_bm_bn_bk"], [[256, 512, 128]])
            native_strassen = next(r for r in findings["pointwise_counts"] if r["scope"] == "call" and r["reference"] == "native" and r["candidate"] == "strassen")
            self.assertEqual(native_strassen["win"], 1)
            self.assertEqual(native_strassen["median_observed_speedup"], 7 / 4)
            self.assertIn("not population", native_strassen["estimand"])

    def test_missing_seal_and_missing_planned_outcome_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as temp:
            screen, confirm, manifest = sealed_fixture(Path(temp))
            ev = summary.plot.Evidence(screen, confirm, manifest)
            (screen / "artifact_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "sealed evidence required"):
                summary.audited_sources(ev)
            ev.results = ev.results[:-1]
            with self.assertRaisesRegex(ValueError, "scoped outcomes differ"):
                summary.planned_completeness(ev)

    def test_near_projection_cannot_invent_cartesian_members(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as temp:
            ev = summary.plot.Evidence(*sealed_fixture(Path(temp)))
            near = ev.confirmation["by_shape"]["unit_m256_n512_k128"]["cubic"]["near_sets"]["call"]["descriptive_5_percent"]
            near["tuples_bm_bn_bk"].append([128, 1024, 256])
            with self.assertRaisesRegex(ValueError, "projection does not match"):
                summary.near_tile_sets(ev)

    def test_existing_output_is_preserved_without_reading_inputs(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as temp:
            path = Path(temp)
            (path / "sentinel").write_text("unchanged")
            with self.assertRaises(FileExistsError):
                summary.generate(path / "missing", path / "missing", path / "missing", path)
            self.assertEqual((path / "sentinel").read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
