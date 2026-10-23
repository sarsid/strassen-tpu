"""Offline isolation/leakage checks; no JAX import or accelerator operations."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from strassen_mm import benchmark_region_grid_v001 as region


def fixture():
    rows = [{"id": role, "m": 256 + i * 256, "n": 512, "k": 512,
             "sampling_role": role, "sampling_stratum": "stratum_" + role,
             "stage": "holdout" if role == "holdout" else "exploratory"}
            for i, role in enumerate(region.ROLES)]
    old = [{"id": f"old_{i}", "m": 4096 + i, "n": 8192, "k": 1024} for i in range(12)]
    manifest = {
        "manifest_id": "region_grid_v001", "cohort_id": "region_grid_v001_allocation_001",
        "shape_order": ["m", "n", "k"], "legacy_benchmark_tuple_order": ["m", "k", "n"],
        "tile_order": ["bm", "bn", "bk"], "shapes": rows,
        "exploratory_shape_ids": list(region.ROLES[:-1]), "new_holdout_shape_ids": ["holdout"],
        **{f"{role}_shape_ids": [role] for role in region.ROLES},
        "old_holdout_shapes": copy.deepcopy(old), "old_holdout_source_file": "legacy.json",
        "protected_legacy_holdout_shape_ids": [row["id"] for row in old],
        "protected_legacy_holdout_geometries": [{axis: row[axis] for axis in region.AXES} for row in old],
    }
    families = {
        "native": [{"candidate_id": "native_default", "algorithm": "native", "variant": "plain", "tile": None}],
        "cubic": [{"candidate_id": "cubic", "algorithm": "cubic_full", "variant": "output_accumulator", "tile": [256, 256, 256]}],
        "strassen": [{"candidate_id": "strassen", "algorithm": "strassen", "variant": "interleaved_output_accumulator", "tile": [256, 256, 256]}],
    }
    campaign = {
        "campaign_id": "region_grid_v001", "cohort_id": manifest["cohort_id"],
        "shape_manifest": "shapes.json", "distribution_manifest": "distributions.json",
        "region_study": {"new_cohort": True, "pooled_with_prior": False},
        "grid_study": {"cohort_manifest_id": "region_grid_v001", "reserved_holdout_shape_ids": ["holdout"],
                       "protected_legacy_holdout_shape_ids": manifest["protected_legacy_holdout_shape_ids"],
                       "sampling_roles": {row["id"]: row["sampling_role"] for row in rows}},
        "timing": {phase: {"warmups": 2, "repeats": 7} for phase in region.grid.PHASES},
        "experiments": {phase: {"shape_ids": ["anchor"] if phase == "GRID-smoke" else list(region.ROLES[:-1]),
                                "candidate_families": copy.deepcopy(families), "timing": phase,
                                "distribution": "gaussian", "seed": 100 + i}
                        for i, phase in enumerate(region.grid.PHASES)},
    }
    legacy = {"holdout_shape_ids": [row["id"] for row in old], "shapes": old}
    return campaign, manifest, legacy


def materialize(folder, campaign, manifest, legacy, phase="GRID-screen"):
    root = Path(folder)
    for name, value in (("campaign.json", campaign), ("legacy.json", legacy),
                        ("distributions.json", {}), ("identity.json", {"identity": {
                            "colab_endpoint": "new-allocation", "allocation_id": "new-allocation"}})):
        (root / name).write_text(json.dumps(value))
    manifest["old_holdout_source_sha256"] = region.grid.base.digest_file(root / "legacy.json")
    (root / "shapes.json").write_text(json.dumps(manifest))
    return SimpleNamespace(campaign=root / "campaign.json", phase=phase, output_dir=root / "output",
                           expected_identity=root / "identity.json", allocation_id="new-allocation",
                           max_wall_seconds=1000, selection=None)


def arguments(args):
    return ["--campaign", str(args.campaign), "--phase", args.phase,
            "--output-dir", str(args.output_dir), "--expected-identity", str(args.expected_identity),
            "--allocation-id", args.allocation_id, "--max-wall-seconds", str(args.max_wall_seconds)]


class RegionPreflightTests(unittest.TestCase):
    def check_mutation(self, mutate, message):
        campaign, manifest, legacy = fixture()
        mutate(campaign, manifest, legacy)
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, campaign, manifest, legacy)
            with self.assertRaisesRegex((ValueError, FileExistsError), message):
                region.preflight(args)
            self.assertFalse(args.output_dir.exists())

    def test_valid_cohort_dispatches_original_arguments_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, *fixture())
            report = region.preflight(args)
            self.assertEqual(report["exploratory_shapes"], 3)
            self.assertEqual(report["old_holdouts_protected"], 12)
            self.assertFalse(report["prior_results_pooled"])
            argv = arguments(args)
            with mock.patch.object(region.grid, "main", return_value=17) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(region.main(argv), 17)
            run.assert_called_once_with(argv)
            self.assertFalse(args.output_dir.exists())

    def test_import_does_not_initialize_jax(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from strassen_mm import benchmark_region_grid_v001; assert 'jax' not in sys.modules"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_holdouts_cannot_be_scheduled_in_any_phase(self):
        for phase in region.grid.PHASES:
            with self.subTest(phase=phase):
                self.check_mutation(lambda c, m, l: c["experiments"][phase]["shape_ids"].append("holdout"),
                                    "reserved holdout")

    def test_actual_old_coordinate_alias_is_rejected(self):
        def mutate(campaign, manifest, legacy):
            manifest["shapes"][0].update({axis: legacy["shapes"][0][axis] for axis in region.AXES})
        self.check_mutation(mutate, "coordinate alias")

    def test_duplicate_new_coordinate_alias_is_rejected(self):
        def mutate(campaign, manifest, legacy):
            manifest["shapes"][0].update({axis: manifest["shapes"][3][axis] for axis in region.AXES})
        self.check_mutation(mutate, "actual coordinates must both be unique")

    def test_duplicate_shape_ids_and_noninteger_coordinates_are_rejected(self):
        self.check_mutation(lambda c, m, l: m["shapes"].append(copy.deepcopy(m["shapes"][0])), "must both be unique")
        self.check_mutation(lambda c, m, l: m["shapes"][0].update(m=True), "positive integer")

    def test_cohort_identity_and_no_pooling_are_explicit(self):
        self.check_mutation(lambda c, m, l: c.update(cohort_id="other_allocation"), "same explicit cohort_id")
        self.check_mutation(lambda c, m, l: c["region_study"].update(pooled_with_prior=True), "without pooling")

    def test_roles_cannot_be_reassigned_or_phase_inventory_truncated(self):
        self.check_mutation(lambda c, m, l: m["shapes"][3].update(stage="exploratory"), "sampling_role and stage")
        self.check_mutation(lambda c, m, l: c["experiments"]["GRID-confirm"]["shape_ids"].pop(), "full frozen")
        self.check_mutation(lambda c, m, l: m["broad_shape_ids"].append("focused"), "sampling_role values")
        self.check_mutation(lambda c, m, l: c["experiments"]["GRID-smoke"]["shape_ids"].append("anchor"), "unique nonempty")

    def test_phase_manifest_cannot_point_outside_its_bundle(self):
        self.check_mutation(lambda c, m, l: c["experiments"]["GRID-screen"].update(shape_manifest="../other/shapes.json"), "inside the campaign bundle")

    def test_candidate_changes_and_seed_reuse_are_rejected(self):
        def candidate_change(campaign, manifest, legacy):
            campaign["experiments"]["GRID-smoke"]["candidate_families"]["strassen"][0]["tile"] = [512, 512, 512]
        self.check_mutation(candidate_change, "identical across phases")
        self.check_mutation(lambda c, m, l: c["experiments"]["GRID-confirm"].update(seed=101), "independent input seeds")

    def test_legacy_protection_is_reconciled_with_sealed_source(self):
        self.check_mutation(lambda c, m, l: m["old_holdout_shapes"][0].update(m=333), "sealed legacy source")
        self.check_mutation(lambda c, m, l: m["protected_legacy_holdout_shape_ids"].pop(), "differ from their sealed source")
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, *fixture())
            (Path(folder) / "legacy.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "source hash"):
                region.preflight(args)

    def test_wrong_allocation_is_rejected_before_delegation(self):
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, *fixture())
            args.allocation_id = "old-allocation"
            stderr = io.StringIO()
            with mock.patch.object(region.grid, "main") as run, contextlib.redirect_stderr(stderr):
                self.assertEqual(region.main(arguments(args)), 2)
            run.assert_not_called()
            self.assertEqual(json.loads(stderr.getvalue())["kind"], "region_preflight_failed")
            self.assertFalse(args.output_dir.exists())

    def test_existing_output_remains_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, *fixture())
            args.output_dir.mkdir()
            evidence = args.output_dir / "results.jsonl"
            evidence.write_text("keep exactly\n")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                region.preflight(args)
            self.assertEqual(evidence.read_text(), "keep exactly\n")

    def test_confirmation_requires_this_cohorts_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            args = materialize(folder, *fixture(), phase="GRID-confirm")
            with self.assertRaisesRegex(ValueError, "frozen screen selection"):
                region.preflight(args)
            args.selection = Path(folder) / "old-selection.json"
            args.selection.write_text(json.dumps({"campaign_sha256": "old"}))
            with self.assertRaisesRegex(ValueError, "another campaign"):
                region.preflight(args)

    def test_confirmation_accepts_setup_subset_and_checks_full_identity_when_given(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, manifest, legacy = fixture()
            args = materialize(folder, campaign, manifest, legacy, phase="GRID-confirm")
            setup = {"colab_endpoint": "new-allocation", "hostname": "host-1", "boot_id": "boot-1",
                     "versions": {"jax": "0.7.2", "libtpu": "0.0.21.1"},
                     "devices": [{"kind": "TPU v5 lite", "id": 0, "process_index": 0}]}
            full = {**setup, "allocation_id": "new-allocation", "runtime_flags": {"JAX_PLATFORMS": "tpu"},
                    "mosaic_compatibility": {"target_ir_version": 7}, "device_kind": "TPU v5 lite"}
            selected = {"phase": "GRID-screen", "stage": "screen", "environment_identity": full,
                        "campaign_sha256": region.grid.base.digest_file(args.campaign),
                        "shape_manifest_sha256": region.grid.base.digest_file(Path(folder) / "shapes.json"),
                        "by_shape": {}}
            for row in manifest["shapes"]:
                if row["sampling_role"] == "holdout":
                    continue
                selected["by_shape"][row["id"]] = {
                    "shape_mkn": [row[axis] for axis in ("m", "k", "n")],
                    **{family: {"winner": catalog[0], "confirmation_candidates": catalog[:1]}
                       for family, catalog in campaign["experiments"]["GRID-screen"]["candidate_families"].items()}}
            args.selection = Path(folder) / "selections.json"
            args.selection.write_text(json.dumps(selected))
            for expected in (setup, {"identity": full}):
                args.expected_identity.write_text(json.dumps(expected))
                self.assertEqual(region.preflight(args)["phase"], "GRID-confirm")
            for expected in ({**setup, "boot_id": "different-boot"},
                             {**full, "runtime_flags": {"JAX_PLATFORMS": "cpu"}}):
                args.expected_identity.write_text(json.dumps(expected))
                with self.assertRaisesRegex(ValueError, "selection environment differs"):
                    region.preflight(args)


if __name__ == "__main__":
    unittest.main()
