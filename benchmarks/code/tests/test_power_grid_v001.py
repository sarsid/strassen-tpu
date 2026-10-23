"""Offline protocol tests; no JAX import, hardware work, or timing experiments."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from strassen_mm import benchmark_power_grid_v001 as grid
from strassen_mm import benchmark_v001 as base


def fixtures(count=8):
    families = {
        "native": [{"candidate_id": "native_default", "algorithm": "native", "variant": "plain", "tile": None},
                   {"candidate_id": "native_vmem32", "algorithm": "native", "variant": "plain", "tile": None,
                    "compiler_options": {grid.NATIVE_OPTION: 32768}}],
        "cubic": [{"candidate_id": f"c{i}", "algorithm": "cubic_full", "variant": "output_accumulator",
                   "tile": [8 * (i + 1), 384, 128]} for i in range(count)],
        "strassen": [{"candidate_id": f"s{i}", "algorithm": "strassen", "variant": "interleaved_output_accumulator",
                      "tile": [16 * (i + 1), 256, 768]} for i in range(count)],
    }
    campaign = {"seed": 331, "timing": {"screen": {"repeats": 7}, "confirm": {"repeats": 30}},
                "memory": {"kernel_vmem_limit_mib": 48}, "experiments": {
                    "GRID-screen": {"shape_ids": ["a"], "timing": "screen", "candidate_families": families},
                    "GRID-confirm": {"shape_ids": ["a"], "timing": "confirm"}}}
    manifest = {"shapes": [{"id": "a", "m": 48, "k": 768, "n": 384}]}
    rows = []
    for family, candidates in families.items():
        for index, candidate in enumerate(candidates):
            for scope in ("call", "prepared_kernel"):
                rows.append({**candidate, "family": family, "shape_id": "a", "scope": scope,
                             "status": "ok", "correctness_pass": True, "group_id": f"screen{index}",
                             "timing": {"sample_count": 7, "mean_ms": 1 + index * .006}})
    selection = grid.select_winners(rows, campaign, manifest)
    selection["environment_identity"] = {"allocation_id": "test"}
    return campaign, manifest, rows, selection


class GridProtocol(unittest.TestCase):
    def test_generated_full_campaign_matches_registered_attempt_counts(self):
        path = Path(__file__).resolve().parents[1] / "configs/generated_power_grid_v001/campaign_power_grid_v001.json"
        if not path.is_file():
            self.skipTest("generated campaign bundle not installed")
        campaign = json.loads(path.read_text())
        manifest = json.loads((path.parent / campaign["shape_manifest"]).read_text())
        self.assertEqual(grid.validate_candidate_space(campaign["experiments"]["GRID-screen"]),
                         {"native": 4, "cubic": 24, "strassen": 24})
        for phase in ("GRID-screen", "GRID-smoke"):
            groups = grid.build_groups(campaign, manifest, phase)
            shape_ids = campaign["experiments"][phase]["shape_ids"]
            self.assertEqual(len(groups), len(shape_ids) * 24)
            self.assertEqual(sum(len(g["arms"]) for g in groups), len(shape_ids) * 52)

    def test_actual_kernel_alignments_and_native_options(self):
        campaign, _, _, _ = fixtures()
        experiment = campaign["experiments"]["GRID-screen"]
        self.assertEqual(grid.validate_candidate_space(experiment), {"native": 2, "cubic": 8, "strassen": 8})
        bad = copy.deepcopy(experiment)
        bad["candidate_families"]["strassen"][0]["tile"][1] = 384
        with self.assertRaisesRegex(ValueError, "alignments"):
            grid.validate_candidate_space(bad)
        bad = copy.deepcopy(experiment)
        bad["candidate_families"]["native"][1]["compiler_options"] = {"arbitrary_flag": 5}
        with self.assertRaisesRegex(ValueError, "unregistered"):
            grid.validate_candidate_space(bad)
        bad = copy.deepcopy(experiment)
        bad["candidate_families"]["cubic"][0]["variant"] = "plain"
        with self.assertRaisesRegex(ValueError, "fixed variant"):
            grid.validate_candidate_space(bad)

    def test_native_default_unique_and_custom_equal_budget(self):
        campaign, _, _, _ = fixtures()
        experiment = campaign["experiments"]["GRID-screen"]
        experiment["candidate_families"]["native"][1]["compiler_options"] = {}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            grid.validate_candidate_space(experiment)
        campaign, _, _, _ = fixtures()
        experiment = campaign["experiments"]["GRID-screen"]
        experiment["candidate_families"]["cubic"].pop()
        with self.assertRaisesRegex(ValueError, "equal candidate"):
            grid.validate_candidate_space(experiment)

    def test_flat_policy_aliases_are_honored(self):
        campaign, manifest, rows, _ = fixtures()
        campaign["experiments"]["GRID-screen"].update(shortlist_top_k=1, shortlist_max_per_family=2,
                                                       near_optimal_fraction=.01)
        selected = grid.select_winners(rows, campaign, manifest)
        self.assertEqual(len(selected["by_shape"]["a"]["cubic"]["confirmation_candidates"]), 2)
        self.assertEqual(selected["selection_policy"]["near_fraction"], .01)

    def test_screen_is_deterministic_and_every_candidate_attempted_once(self):
        campaign, manifest, _, _ = fixtures()
        groups = grid.build_groups(campaign, manifest, "GRID-screen")
        self.assertEqual(groups, grid.build_groups(campaign, manifest, "GRID-screen"))
        arms = [a for g in groups for a in g["arms"]]
        self.assertEqual(len(arms), 18)
        self.assertEqual(len({a["candidate_id"] for a in arms}), 18)
        self.assertTrue(all(len({a["arm_id"] for a in g["arms"]}) == len(g["arms"]) for g in groups))

    def test_shortlist_cap_preserves_all_screen_near_and_unconfirmed_flags(self):
        _, _, _, selection = fixtures()
        row = selection["by_shape"]["a"]["cubic"]
        self.assertEqual(row["winner"]["candidate_id"], "c0")
        self.assertEqual(len(row["screen_near_candidates"]), 8)
        self.assertEqual(len(row["confirmation_candidates"]), 6)
        self.assertEqual(sum(not c["scheduled_for_confirmation"] for c in row["screen_near_candidates"]), 2)
        self.assertIn("Cartesian", row["screen_near_tile_projection"]["scope"])

    def test_selection_requires_both_scopes_complete_positive_finite_correct(self):
        campaign, manifest, rows, _ = fixtures()
        for row in rows:
            if row["candidate_id"] == "c0" and row["scope"] == "prepared_kernel":
                row["correctness_pass"] = False
            if row["candidate_id"] == "c1":
                row["timing"]["sample_count"] = 6
            if row["candidate_id"] == "c2":
                row["timing"]["mean_ms"] = float("nan")
        selected = grid.select_winners(rows, campaign, manifest)
        self.assertEqual(selected["by_shape"]["a"]["cubic"]["winner"]["candidate_id"], "c3")
        duplicate = rows + [copy.deepcopy(next(r for r in rows if r["candidate_id"] == "c3"))]
        selected = grid.select_winners(duplicate, campaign, manifest)
        self.assertEqual(selected["by_shape"]["a"]["cubic"]["winner"]["candidate_id"], "c4")

    def test_confirm_headline_plus_bounded_pairs_and_no_duplicate_default(self):
        campaign, manifest, _, selection = fixtures()
        groups = grid.build_groups(campaign, manifest, "GRID-confirm", selection)
        self.assertEqual(groups[0]["group_id"], "a__headline_confirmation")
        self.assertEqual(len(groups[0]["arms"]), 3)
        self.assertTrue(all(len(g["arms"]) == 2 for g in groups[1:]))
        self.assertTrue(all(len({a["arm_id"] for a in g["arms"]}) == len(g["arms"]) for g in groups))
        native = selection["by_shape"]["a"]["native"]
        native["winner"] = native["confirmation_candidates"][1]
        groups = grid.build_groups(campaign, manifest, "GRID-confirm", selection)
        self.assertEqual(len(groups[0]["arms"]), 4)
        self.assertEqual(sum(g["group_id"].startswith("a__finalist_native") for g in groups), 0)

    def test_selection_rejects_identity_tile_and_option_mutation(self):
        campaign, manifest, _, selection = fixtures()
        identity = selection["environment_identity"]
        self.assertTrue(grid.verify_selection(selection, campaign, manifest, identity))
        with self.assertRaisesRegex(ValueError, "identities differ"):
            grid.verify_selection(selection, campaign, manifest, {})
        selection["by_shape"]["a"]["native"]["confirmation_candidates"][1]["compiler_options"][grid.NATIVE_OPTION] = 1
        with self.assertRaisesRegex(ValueError, "differs from preregistered"):
            grid.verify_selection(selection, campaign, manifest, identity)

    def test_journal_groups_repeated_reference_samples_without_pooling(self):
        campaign, manifest, _, selection = fixtures()
        groups = grid.build_groups(campaign, manifest, "GRID-confirm", selection)
        with tempfile.TemporaryDirectory() as folder:
            journal = grid.GridJournal(Path(folder), "GRID-confirm")
            journal.groups = {g["group_id"]: g for g in groups}
            references = [g for g in groups if any(a["candidate_id"] == "c0" for a in g["arms"])]
            for i, group in enumerate(references):
                arm = next(a for a in group["arms"] if a["candidate_id"] == "c0")
                journal.emit("sample", group_id=group["group_id"], arm_id=arm["arm_id"], scope="call", round=0, elapsed_ms=i+1)
            journal.close()
            self.assertEqual(len(journal.samples), len(references))
            self.assertTrue(all(len(samples) == 1 for samples in journal.samples.values()))

    def test_confirmation_bootstrap_pairs_use_same_group_and_frozen_reference(self):
        campaign, manifest, _, selection = fixtures(2)
        groups = grid.build_groups(campaign, manifest, "GRID-confirm", selection)
        journal = SimpleNamespace(selection_rows=[], samples={})
        for i, group in enumerate(groups):
            for arm in group["arms"]:
                for scope in group["scopes"]:
                    # Each later pair is deliberately ten times slower than headline.
                    time_ms = 10 ** i * (1.03 if arm["candidate_id"] in ("c1", "s1", "native_vmem32") else 1)
                    journal.selection_rows.append({**arm, "shape_id": "a", "group_id": group["group_id"],
                        "scope": scope, "status": "ok", "correctness_pass": True,
                        "timing": {"sample_count": 30, "mean_ms": time_ms}})
                    journal.samples[(group["group_id"], "a", arm["candidate_id"], scope)] = [
                        {"round": r, "elapsed_ms": time_ms} for r in range(30)]
        def pair(left, right, seed):
            self.assertEqual(len(left), 30)
            self.assertEqual(len(right), 30)
            ratio = left[0]["elapsed_ms"] / right[0]["elapsed_ms"]
            return {"speedup_ci95": [ratio, ratio], "speedup_ratio_of_means": ratio, "method": "test"}
        with mock.patch.object(base, "paired_comparison", side_effect=pair):
            report = grid.confirmation_report(journal, selection, campaign)
        for family in grid.FAMILIES:
            candidate = report["by_shape"]["a"][family]["candidates"][1]
            self.assertAlmostEqual(candidate["scopes"]["call"]["latency_ratio_to_frozen_winner"], 1.03)
            self.assertTrue(candidate["scopes"]["call"]["certified_within_5_percent"])
        self.assertEqual(report["by_shape"]["a"]["cubic"]["frozen_winner"]["candidate_id"], "c0")

    def test_compile_options_are_local_and_rejections_recorded(self):
        campaign, manifest, _, _ = fixtures(1)
        arms = [grid.make_arm(c, f) for f, candidates in campaign["experiments"]["GRID-screen"]["candidate_families"].items()
                for c in candidates]
        group = {"group_id": "g", "shape": manifest["shapes"][0], "arms": arms,
                 "scopes": ["call", "prepared_kernel"]}
        calls = []
        def compile_fn(**kwargs):
            calls.append(kwargs)
            if kwargs:
                raise ValueError("unknown compiler option xla_tpu_scoped_vmem_limit_kib")
            return "executable"
        lowered = SimpleNamespace(compile=compile_fn)
        fake_jax = SimpleNamespace(ShapeDtypeStruct=lambda shape, dtype: (shape, dtype),
                                  jit=lambda fn: SimpleNamespace(lower=lambda *args: lowered))
        fake_fn = SimpleNamespace(metadata={"padded_shape_mkn": [48, 768, 384]}, kernel=lambda: None)
        journal = SimpleNamespace(emit=mock.Mock())
        runner = grid.GridRunner(SimpleNamespace(max_wall_seconds=100), campaign, journal)
        operand = SimpleNamespace(shape=(48, 768), dtype="bf16")
        with mock.patch.object(base, "jax", fake_jax, create=True), mock.patch.object(base, "make_matmul", return_value=fake_fn, create=True):
            entries = runner.compile_entries(group, operand, operand)
        self.assertEqual(len(entries), 8)
        self.assertEqual(sum("error" in e for e in entries), 2)
        self.assertEqual(sum(bool(c) for c in calls), 2)
        self.assertTrue(all(c == {"compiler_options": {grid.NATIVE_OPTION: 32768}} for c in calls if c))
        self.assertTrue(all("executable" in e for e in entries if e["candidate_id"] != "native_vmem32"))


if __name__ == "__main__":
    unittest.main()
