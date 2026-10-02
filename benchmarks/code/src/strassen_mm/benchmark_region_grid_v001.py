"""Preflight one isolated region-grid cohort, then run the frozen grid engine.

The wrapper performs only local JSON and hash checks before delegation. It does
not alter timing groups, kernels, selection rules, output files, or raw samples.
Failed preflight leaves the requested output directory uncreated; the immutable
outer launcher preserves the error in its log and completion evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from strassen_mm import benchmark_power_grid_v001 as grid


ROLES = ("anchor", "broad", "focused", "holdout")
AXES = ("m", "n", "k")


def ids(value, context, *, nonempty=True):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{context} must contain unique nonempty shape IDs")
    return set(value)


def coordinates(row, context):
    if not isinstance(row, dict) or any(type(row.get(axis)) is not int or row[axis] <= 0 for axis in AXES):
        raise ValueError(f"{context} requires positive integer M,N,K coordinates")
    return tuple(row[axis] for axis in AXES)


def local_input(campaign_path, value, context):
    """Bundle inputs may not silently resolve to another cohort's files."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{context} must be a relative bundle path")
    result = (campaign_path.parent / value).resolve()
    if not result.is_relative_to(campaign_path.parent) or not result.is_file():
        raise ValueError(f"{context} must exist inside the campaign bundle")
    return result


def validate_manifest(campaign, manifest, campaign_path):
    cohort = campaign.get("cohort_id")
    if (not isinstance(cohort, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", cohort)
            or cohort != manifest.get("cohort_id")):
        raise ValueError("campaign and shape manifest require the same explicit cohort_id")
    region = campaign.get("region_study", {})
    if region.get("new_cohort") is not True or region.get("pooled_with_prior") is not False:
        raise ValueError("region_study must explicitly declare a new cohort without pooling prior data")
    study = campaign.get("grid_study", {})
    if (manifest.get("manifest_id") != "region_grid_v001"
            or study.get("cohort_manifest_id") != manifest["manifest_id"]
            or campaign.get("campaign_id") != manifest["manifest_id"]):
        raise ValueError("campaign and region manifest identifiers differ")
    if manifest.get("shape_order") != list(AXES):
        raise ValueError("region manifest must explicitly use public M,N,K coordinate order")
    if (manifest.get("legacy_benchmark_tuple_order") != ["m", "k", "n"]
            or manifest.get("tile_order") != ["bm", "bn", "bk"]):
        raise ValueError("region manifest has an inconsistent kernel or tile coordinate order")
    records = manifest.get("shapes")
    if not isinstance(records, list) or not records:
        raise ValueError("region shape manifest must be nonempty")
    shapes, seen_coordinates = {}, set()
    for row in records:
        coord = coordinates(row, "manifest shape")
        name = row.get("id")
        if not isinstance(name, str) or not name or name in shapes or coord in seen_coordinates:
            raise ValueError("manifest shape IDs and actual coordinates must both be unique")
        role = row.get("sampling_role")
        if role not in ROLES or row.get("stage") != ("holdout" if role == "holdout" else "exploratory"):
            raise ValueError("shape sampling_role and stage must agree")
        if not isinstance(row.get("sampling_stratum"), str) or not row["sampling_stratum"]:
            raise ValueError("every shape requires a nonempty sampling_stratum")
        shapes[name] = row
        seen_coordinates.add(coord)
    role_ids = {}
    for role in ROLES:
        role_ids[role] = ids(manifest.get(f"{role}_shape_ids"), f"{role}_shape_ids")
        actual = {name for name, row in shapes.items() if row["sampling_role"] == role}
        if role_ids[role] != actual:
            raise ValueError(f"{role}_shape_ids differs from shape sampling_role values")
    exploratory = ids(manifest.get("exploratory_shape_ids"), "exploratory_shape_ids")
    if exploratory != set.union(*(role_ids[role] for role in ROLES if role != "holdout")):
        raise ValueError("exploratory IDs must be exactly anchor, broad and focused shapes")
    held = role_ids["holdout"]
    if (exploratory & held or exploratory | held != set(shapes)
            or ids(manifest.get("new_holdout_shape_ids"), "new_holdout_shape_ids") != held):
        raise ValueError("new holdout and exploratory inventories are inconsistent")
    if ids(study.get("reserved_holdout_shape_ids"), "campaign reserved holdouts") != held:
        raise ValueError("campaign holdout reservation differs from manifest")
    if study.get("sampling_roles") != {name: row["sampling_role"] for name, row in shapes.items()}:
        raise ValueError("campaign sampling_roles differ from manifest")

    legacy_path = local_input(campaign_path, manifest.get("old_holdout_source_file"), "old holdout source")
    legacy_hash = manifest.get("old_holdout_source_sha256")
    if grid.base.digest_file(legacy_path) != legacy_hash:
        raise ValueError("old holdout source hash differs from frozen protection inventory")
    legacy = json.loads(legacy_path.read_text())
    legacy_ids = ids(legacy.get("holdout_shape_ids"), "legacy holdout IDs")
    if len(legacy_ids) != 12:
        raise ValueError("the original twelve holdout shapes must stay protected")
    legacy_rows = [row for row in legacy.get("shapes", []) if row.get("id") in legacy_ids]
    if len(legacy_rows) != len(legacy_ids) or {row["id"] for row in legacy_rows} != legacy_ids:
        raise ValueError("legacy source does not resolve its exact holdout inventory")
    legacy_by_id = {row["id"]: coordinates(row, "legacy holdout") for row in legacy_rows}
    legacy_coords = set(legacy_by_id.values())
    if len(legacy_coords) != 12:
        raise ValueError("original holdout coordinates must be unique")
    protected = manifest.get("old_holdout_shapes")
    if (not isinstance(protected, list) or len(protected) != 12
            or {row.get("id"): coordinates(row, "protected old holdout") for row in protected} != legacy_by_id):
        raise ValueError("old_holdout_shapes differs from the sealed legacy source")
    for values in (manifest.get("protected_legacy_holdout_shape_ids"), study.get("protected_legacy_holdout_shape_ids")):
        if ids(values, "protected legacy holdout IDs") != legacy_ids:
            raise ValueError("protected legacy holdout IDs differ from their sealed source")
    protected_coords = manifest.get("protected_legacy_holdout_geometries")
    if (not isinstance(protected_coords, list) or len(protected_coords) != 12
            or {coordinates(row, "protected legacy coordinates") for row in protected_coords} != legacy_coords):
        raise ValueError("protected legacy coordinates differ from their sealed source")
    if seen_coordinates & legacy_coords or set(shapes) & legacy_ids:
        raise ValueError("region manifest contains a protected old holdout or coordinate alias")
    return {"cohort_id": cohort, "exploratory": exploratory, "holdout": held,
            "shapes": shapes, "legacy_holdout_count": len(legacy_ids)}


def preflight(args):
    campaign_path = args.campaign.resolve()
    campaign = json.loads(campaign_path.read_text())
    manifest_path = local_input(campaign_path, campaign.get("shape_manifest"), "shape manifest")
    local_input(campaign_path, campaign.get("distribution_manifest"), "distribution manifest")
    manifest = json.loads(manifest_path.read_text())
    checked = validate_manifest(campaign, manifest, campaign_path)
    experiments = campaign.get("experiments", {})
    if set(experiments) != set(grid.PHASES):
        raise ValueError("region campaign requires exactly GRID-smoke, GRID-screen and GRID-confirm")
    seeds = []
    screen = experiments["GRID-screen"]
    grid.validate_candidate_space(screen)
    for phase, experiment in experiments.items():
        planned = ids(experiment.get("shape_ids"), f"{phase} shape IDs")
        if not planned <= checked["exploratory"]:
            raise ValueError(f"{phase} includes unknown or reserved holdout shapes")
        if phase != "GRID-smoke" and planned != checked["exploratory"]:
            raise ValueError("screen and confirmation must cover the full frozen exploratory inventory")
        path = local_input(campaign_path, experiment.get("shape_manifest", campaign["shape_manifest"]), f"{phase} manifest")
        if path != manifest_path:
            raise ValueError("phase-specific shape manifest overrides cannot change the frozen cohort")
        if experiment.get("candidate_families", screen["candidate_families"]) != screen["candidate_families"]:
            raise ValueError("candidate families must remain identical across phases")
        if experiment.get("distribution", "gaussian") != "gaussian":
            raise ValueError("region timing phases require Gaussian inputs")
        seed = experiment.get("seed", campaign.get("seed"))
        if type(seed) is not int or seed < 0:
            raise ValueError("every phase requires a nonnegative integer input seed")
        seeds.append(seed)
        timing = campaign.get("timing", {}).get(experiment.get("timing"), {})
        if any(type(timing.get(key)) is not int or timing[key] < (1 if key == "repeats" else 0)
               for key in ("warmups", "repeats")):
            raise ValueError("every phase requires valid warmup and repeat counts")
    if len(set(seeds)) != len(seeds):
        raise ValueError("smoke, screen and confirmation must use independent input seeds")
    expected = json.loads(args.expected_identity.read_text())
    identity = expected.get("identity", expected)
    if (identity.get("colab_endpoint") != args.allocation_id
            or identity.get("allocation_id", args.allocation_id) != args.allocation_id):
        raise ValueError("expected identity does not match the requested allocation")
    if args.phase == "GRID-confirm":
        selection_path = grid.n5.resolve_input_path(campaign_path, args.selection, experiments[args.phase].get("selection_file"))
        if selection_path is None:
            raise ValueError("confirmation requires this cohort's frozen screen selection")
        selection = json.loads(selection_path.read_text())
        if (selection.get("campaign_sha256") != grid.base.digest_file(campaign_path)
                or selection.get("shape_manifest_sha256") != grid.base.digest_file(manifest_path)):
            raise ValueError("confirmation selection belongs to another campaign or shape manifest")
        if set(selection.get("by_shape", {})) != checked["exploratory"]:
            raise ValueError("confirmation selection shape inventory differs from this cohort")
        selected_identity = selection.get("environment_identity", {})
        # A provisioning identity predates JAX initialization and intentionally
        # lacks the complete fixed runtime flags/compatibility record. Mirror
        # the original runner's provisioning subset here; actual full runtime
        # identity is still captured and checked by the timing engine itself.
        if "versions" in identity and "runtime_flags" not in identity:
            fields = {"colab_endpoint", "hostname", "boot_id", "versions", "devices"}
        else:
            fields = set(identity)
        if any(selected_identity.get(key) != identity.get(key) for key in fields):
            raise ValueError("confirmation selection environment differs from the expected allocation identity")
        grid.verify_selection(selection, campaign, manifest, selected_identity)
    elif args.selection is not None:
        raise ValueError("a frozen selection is only accepted for confirmation")
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; use a new immutable run ID")
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("max-wall-seconds must be finite and positive")
    return {"kind": "region_preflight_ok", "cohort_id": checked["cohort_id"],
            "phase": args.phase, "allocation_id": args.allocation_id,
            "campaign_sha256": grid.base.digest_file(campaign_path),
            "shape_manifest_sha256": grid.base.digest_file(manifest_path),
            "exploratory_shapes": len(checked["exploratory"]), "new_holdouts_unobserved": len(checked["holdout"]),
            "old_holdouts_protected": checked["legacy_holdout_count"],
            "timing_engine": "benchmark_power_grid_v001", "prior_results_pooled": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=grid.PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    try:
        record = preflight(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"kind": "region_preflight_failed", "error_type": type(error).__name__,
                          "message": str(error), "output_directory_created": False}), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(record, sort_keys=True), flush=True)
    return grid.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
