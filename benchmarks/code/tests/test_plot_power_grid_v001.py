"""Synthetic unit fixtures only: no TPU measurements and no final plot outputs.

The fixture deliberately repeats a much faster winner in another group to catch
accidental pooling, and uses asymmetric dimensions to catch M/K/N transposition.
"""
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools/plot_power_grid_v001.py"
SPEC = importlib.util.spec_from_file_location("plot_power_grid", SOURCE)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def journal(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(parent):
    """Write fake data only under a caller-owned temporary directory."""
    screen, confirm = parent / "synthetic_screen", parent / "synthetic_confirmation"
    screen.mkdir()
    confirm.mkdir()
    shape_id, holdout_id = "unit_m256_n512_k128", "unit_holdout"
    manifest = {"exploratory_shape_ids": [shape_id], "holdout_shape_ids": [holdout_id],
                "shapes": [{"id": shape_id, "m": 256, "n": 512, "k": 128,
                            "stage": "exploratory", "family": "unit_fixture", "on_lattice": True},
                           {"id": holdout_id, "m": 3, "n": 6, "k": 12,
                            "stage": "holdout", "family": "unit_fixture", "on_lattice": True}]}
    manifest_path = parent / "synthetic_manifest.json"
    dump(manifest_path, manifest)
    campaign = {"seed": 123, "timing": {"screen": {"repeats": 3}, "confirm": {"repeats": 3}},
                "experiments": {"GRID-screen": {"shape_ids": [shape_id], "timing": "screen"},
                                "GRID-confirm": {"shape_ids": [shape_id], "timing": "confirm"}}}
    identity = {"allocation_id": "SYNTHETIC_UNIT_FIXTURE_NOT_A_DEVICE"}
    specs = {
        "native_default": {"candidate_id": "native_default", "family": "native", "algorithm": "native", "variant": "plain", "tile": None, "compiler_options": {}},
        "native_tuned": {"candidate_id": "native_tuned", "family": "native", "algorithm": "native", "variant": "plain", "tile": None, "compiler_options": {"xla_tpu_scoped_vmem_limit_kib": 32768}},
        "cubic_256": {"candidate_id": "cubic_256", "family": "cubic", "algorithm": "cubic_full", "variant": "output_accumulator", "tile": [256, 512, 128], "compiler_options": {}},
        "strassen_256": {"candidate_id": "strassen_256", "family": "strassen", "algorithm": "strassen", "variant": "interleaved_output_accumulator", "tile": [256, 512, 256], "compiler_options": {}},
    }
    winner_ids = {"native": "native_tuned", "cubic": "cubic_256", "strassen": "strassen_256"}
    means = {"native_default": 8., "native_tuned": 7., "cubic_256": 6., "strassen_256": 4.}

    def add(rows, candidate_id, scope, group, mean):
        seed = 314
        context = {"shape_id": shape_id, "shape_mkn": [256, 128, 512],
                   "group_id": group, "case_id": group + "__gaussian__314", "scope": scope,
                   "seed": seed, "distribution": "gaussian", **specs[candidate_id]}
        values = [mean * .9, mean, mean * 1.1]
        for round_id, value in enumerate(values):
            rows.append({"event": "sample", **context, "round": round_id, "position": 0, "elapsed_ms": value})
        row = {"event": "case_result", **context, "status": "ok",
               "correctness": {"pass": True, "relative_l2": 0.001, "max_abs_error": 0.005},
               "timing": {"sample_count": len(values), "mean_ms": statistics.mean(values)}}
        rows.append(row)
        return row

    screen_rows, confirm_rows = [], []
    for scope in plot.SCOPES:
        for candidate_id, value in means.items():
            add(screen_rows, candidate_id, scope, shape_id + "__screen", value)
            add(confirm_rows, candidate_id, scope, shape_id + "__headline_confirmation", value)
        # Much faster repeated winner controls must NEVER replace headline7ms.
        add(confirm_rows, "native_tuned", scope, shape_id + "__finalist_native_other", .1)
    for directory, rows, phase in ((screen, screen_rows, "GRID-screen"), (confirm, confirm_rows, "GRID-confirm")):
        journal(directory / "results.jsonl", rows)
        dump(directory / "summary.json", {"phase": phase, "completed": True, "status": "completed", "not_completed_group_ids": []})
        dump(directory / "environment.json", {"identity": identity})
        config = directory / "config_snapshot"
        config.mkdir()
        dump(config / "campaign.json", campaign)
        dump(directory / "source_manifest.json", {"sha256": {"source_snapshot/unit.py": "fake-test-source-only",
            "config_snapshot/campaign.json": plot.sha256(config / "campaign.json")}})
    selection = {"stage": "screen", "phase": "GRID-screen", "environment_identity": identity,
                 "shape_manifest_sha256": plot.sha256(manifest_path), "screen_results_sha256": plot.sha256(screen / "results.jsonl"),
                 "source_manifest_sha256": plot.sha256(screen / "source_manifest.json"),
                 "campaign_sha256": plot.sha256(screen / "config_snapshot/campaign.json"), "by_shape": {shape_id: {}}}
    for family, candidate_id in winner_ids.items():
        selection["by_shape"][shape_id][family] = {"winner": specs[candidate_id],
            "confirmation_candidates": [specs[candidate_id]], "screen_eligible_candidates": [specs[candidate_id]]}
    dump(screen / "selections.json", selection)
    dump(confirm / "selection_input.json", selection)
    dump(confirm / "selection_input_provenance.json", {"sha256": plot.sha256(screen / "selections.json")})
    report = {"by_shape": {shape_id: {}}}
    for family, candidate_id in winner_ids.items():
        report["by_shape"][shape_id][family] = {"frozen_winner": specs[candidate_id], "candidates": [], "near_sets": {}}
    contrasts = []
    headlines = [r for r in confirm_rows if r["event"] == "case_result" and r["group_id"].endswith("__headline_confirmation")]
    for candidate in headlines:
        for reference in headlines:
            if candidate["scope"] != reference["scope"] or candidate["candidate_id"] == reference["candidate_id"]:
                continue
            get_raw = lambda row: [r for r in confirm_rows if r["event"] == "sample" and plot.row_key(r) == plot.row_key(row)]
            pair = plot.paired_ratio(reference, candidate, get_raw(reference), get_raw(candidate), seed=campaign["seed"] + 811)
            contrasts.append({"candidate_id": candidate["candidate_id"], "reference_candidate_id": reference["candidate_id"],
                "scope": candidate["scope"], "group_id": candidate["group_id"],
                "speedup_ratio_of_means": pair["speedup"], "speedup_ci95": [pair["ci95_low"], pair["ci95_high"]], "method": pair["method"]})
    report["by_shape"][shape_id]["headline_comparisons"] = contrasts
    dump(confirm / "confirmation.json", report)
    return screen, confirm, manifest_path


class PowerGridPlotTests(unittest.TestCase):
    def test_dimensions_and_asymmetric_public_order(self):
        self.assertEqual(len(plot.dimensions()), 34)
        self.assertEqual(plot.dimensions()[-1], 131072)
        self.assertEqual(plot.shape_label({"m": 2, "n": 3, "k": 5}), "2 × 3 × 5")

    def test_headline_denominators_and_repeated_controls_are_not_pooled(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
            screen, confirm, manifest = fixture(Path(temp))
            ev = plot.Evidence(screen, confirm, manifest)
            pair = ev.pairs("call", "native", "strassen")[0]
            self.assertEqual(pair["speedup"], 7 / 4)
            self.assertEqual(ev.pairs("call", "native_default", "strassen")[0]["speedup"], 2)
            self.assertEqual(plot.mean_ms(ev.selected("unit_m256_n512_k128", "call", "native")), 7)
            self.assertEqual(pair["paired_round_count"], 3)
            self.assertEqual(pair["bootstrap_seed"], 934)
            self.assertIn("interval replay verified", pair["uncertainty_source"])
            self.assertEqual(ev.audit["headline_rows"], 8)
            self.assertEqual(ev.audit["sample_checks"], 18)

    def test_cross_group_and_duplicate_rounds_rejected(self):
        row = {"status": "ok", "correctness": {"pass": True}, "timing": {"mean_ms": 1.},
               "group_id": "g", "case_id": "case", "scope": "call", "shape_id": "s", "seed": 1}
        sample = [{"round": 0, "elapsed_ms": 1.}]
        with self.assertRaisesRegex(ValueError, "differs in group_id"):
            plot.paired_ratio(row, {**row, "group_id": "other"}, sample, sample)
        with self.assertRaisesRegex(ValueError, "Duplicate timing rounds"):
            plot.paired_ratio(row, row, sample + sample, sample)
        with self.assertRaisesRegex(ValueError, "Unequal raw round"):
            plot.paired_ratio(row, row, sample, [{"round": 1, "elapsed_ms": 1.}])

    def test_identity_mismatch_rejected_before_plotting(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
            screen, confirm, manifest = fixture(Path(temp))
            dump(confirm / "environment.json", {"identity": {"allocation_id": "other-fake-unit-cohort"}})
            with self.assertRaisesRegex(ValueError, "environment identities differ"):
                plot.Evidence(screen, confirm, manifest)

    def test_wrong_selection_or_manifest_or_unfinished_run_rejected(self):
        for mismatch in ("selection", "manifest", "unfinished"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
                screen, confirm, manifest = fixture(Path(temp))
                if mismatch == "selection":
                    dump(confirm / "selection_input.json", {"wrong": "synthetic"})
                elif mismatch == "manifest":
                    changed = json.loads(manifest.read_text())
                    changed["unregistered_change"] = True
                    dump(manifest, changed)
                else:
                    dump(confirm / "summary.json", {"phase": "GRID-confirm", "completed": False, "status": "failed_or_interrupted"})
                with self.assertRaises(ValueError):
                    plot.Evidence(screen, confirm, manifest)

    def test_holdout_and_axis_transposition_rejected(self):
        for alteration in ("holdout", "transpose"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
                screen, confirm, manifest = fixture(Path(temp))
                rows = [json.loads(line) for line in (confirm / "results.jsonl").read_text().splitlines()]
                if alteration == "holdout":
                    rows[0]["shape_id"] = "unit_holdout"
                else:
                    rows[0]["shape_mkn"] = [256, 512, 128]
                journal(confirm / "results.jsonl", rows)
                with self.assertRaisesRegex(ValueError, "holdout|Shape order"):
                    plot.Evidence(screen, confirm, manifest)

    def test_archived_ci_tampering_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
            screen, confirm, manifest = fixture(Path(temp))
            report = json.loads((confirm / "confirmation.json").read_text())
            for pair in report["by_shape"]["unit_m256_n512_k128"]["headline_comparisons"]:
                pair["speedup_ci95"] = [0.1, 99.]
            dump(confirm / "confirmation.json", report)
            with self.assertRaisesRegex(ValueError, "interval differs"):
                plot.Evidence(screen, confirm, manifest)

    def test_output_directory_refuses_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_plot_unit_") as temp:
            parent = Path(temp)
            sentinel = parent / "sentinel.txt"
            sentinel.write_text("preserve existing output")
            with self.assertRaises(FileExistsError):
                plot.generate(parent / "missing", parent / "missing", parent / "missing", parent)
            self.assertEqual(sentinel.read_text(), "preserve existing output")


if __name__ == "__main__":
    unittest.main()
