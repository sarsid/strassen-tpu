#!/usr/bin/env python3
"""Freeze a separate, unmeasured Strassen-region campaign; never overwrite data.

Public coordinates are M,N,K. Broad sampling is a seeded simple random sample
from a fully enumerated feasible power/midpoint domain. Focused geometries use
previous exploratory findings as hypotheses, never new timing observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_power_midpoint_grid_v001 as geometry
import configure_power_grid_v001 as configuration

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ID = "region_grid_v001"
COHORT_ID = "region_grid_v001_20260920"
SEED = 2026092001
BROAD_COUNT = 60
FOCUSED_COUNT = 48
HOLDOUT_COUNT = 20
MIN_DIMENSION = 256
MAX_BROAD_VOLUME = 16384 ** 3
TILES = tuple(dict.fromkeys(configuration.CUBIC_TILES + configuration.STRASSEN_TILES))
ANCHORS = (
    (3, 3, 3), (512, 512, 512), (768, 768, 768),
    (2048, 2048, 2048), (4096, 4096, 4096),
    (8192, 8192, 8192), (16384, 16384, 16384),
    (2048, 2048, 16384), (2048, 2048, 32768), (2048, 2048, 131072),
    (16384, 2048, 2048), (2048, 16384, 2048),
)


def point(row):
    return tuple(row[axis] for axis in ("m", "n", "k"))


def shape_id(shape):
    m, n, k = shape
    return f"region_m{m}_n{n}_k{k}"


def _visit_geometry(value, found):
    if isinstance(value, dict):
        if all(type(value.get(axis)) is int for axis in ("m", "n", "k")):
            found.add(point(value))
        mkn = value.get("shape_mkn")
        if isinstance(mkn, list) and len(mkn) == 3 and all(type(v) is int for v in mkn):
            found.add((mkn[0], mkn[2], mkn[1]))
        for child in value.values():
            _visit_geometry(child, found)
    elif isinstance(value, list):
        for child in value:
            _visit_geometry(child, found)


def historical_inventory(root=ROOT):
    """Read a bounded, explicit configuration inventory, never latency results."""
    paths = [root / "configs" / name for name in (
        "shapes_v1.json", "shapes_heldout_v1.json", "shapes_real_weights_v1.json",
        "generated_power_grid_v001/sampled_shapes.json",
    )]
    for folder in ("generated_n8_v001", "generated_n9_v001"):
        files = sorted((root / "configs" / folder).glob("*.json"))
        if not files:
            raise ValueError(f"Missing historical configuration inventory: {folder}")
        paths.extend(files)
    found, hashes = set(), {}
    for path in paths:
        content = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        _visit_geometry(json.loads(content), found)
    legacy = json.loads((root / "configs/generated_power_grid_v001/sampled_shapes.json").read_text())
    old_holdouts = [row for row in legacy["shapes"] if row["id"] in legacy["holdout_shape_ids"]]
    old_exploratory = {point(row) for row in legacy["shapes"] if row["id"] in legacy["exploratory_shape_ids"]}
    if len(old_holdouts) != 12 or not set(ANCHORS) <= old_exploratory:
        raise ValueError("Unexpected legacy holdout inventory or missing historical anchor")
    return found, old_holdouts, {
        "files_sha256": hashes,
        "unique_configured_geometries": len(found),
        "parsed_geometry_fields": ["named integer m,n,k", "shape_mkn lists"],
        "limitation": "Bounded active configuration inventory only; archived runs and other projects are not exhaustively searched. New means absent from these files, not globally never measured.",
    }


def _focus_pools():
    """Priority-ordered geometry hypotheses, with replacements fixed before timing."""
    squares = [(s, s, s) for s in (
        5120, 6144, 7168, 9216, 10240, 11264, 14336, 15360,
        4608, 5632, 6656, 7680, 8704, 9728, 10752, 11776,
    )]
    aspects = []
    for seed in ((4096, 8192, 8192), (6144, 8192, 8192),
                 (8192, 12288, 12288), (8192, 16384, 16384),
                 (7168, 8192, 8192), (8192, 10240, 10240)):
        aspects.extend(sorted(set(itertools.permutations(seed))))
    long_k = [(side, side, k) for k in (8192, 16384, 32768, 65536, 131072)
              for side in (1536, 3072)]
    long_k.extend(((2048, 3072, 32768), (3072, 2048, 32768),
                   (1536, 3072, 65536), (3072, 1536, 65536),
                   (2560, 2560, 32768), (2560, 2560, 65536)))
    boundaries = [(s, s, s) for s in (8191, 8193, 16383, 16385)]
    for center in (8192, 16384):
        for axis in range(3):
            for delta in (-1, 1):
                p = [8192, 8192, 8192]
                p[axis] = center + delta
                boundaries.append(tuple(p))
    for center in (8192, 16384):
        for axis in range(3):
            for delta in (-128, 128):
                p = [8192, 8192, 8192]
                p[axis] = center + delta
                boundaries.append(tuple(p))
    return (("focused_square_scale", 8, squares),
            ("focused_axis_permutations", 12, aspects),
            ("focused_long_k", 12, long_k),
            ("focused_tile_boundaries", 16, boundaries))


def build_manifest(root=ROOT, seed=SEED, cohort_id=COHORT_ID):
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("A nonempty, allocation-specific cohort ID is required")
    historical, old_holdouts, history = historical_inventory(root)
    values = tuple(x for x in geometry.dimensions() if x >= MIN_DIMENSION)
    lattice = set(geometry.dimensions())
    estimates = {}

    def estimate(p):
        if p not in estimates:
            estimates[p] = geometry.pair_buffer_estimate(*p, tiles=TILES)
        return estimates[p]

    inventory = []
    for p in itertools.product(values, repeat=3):
        fits = estimate(p) <= geometry.BUDGET_BYTES
        within_volume = p[0] * p[1] * p[2] <= MAX_BROAD_VOLUME
        inventory.append({"shape_id": shape_id(p), "m": p[0], "n": p[1], "k": p[2],
                          "conservative_pair_buffer_bytes": estimate(p),
                          "within_planning_budget": fits,
                          "within_broad_volume_limit": within_volume,
                          "historically_configured": p in historical,
                          "eligible_for_broad": fits and within_volume and p not in historical})
    population = [point(row) for row in inventory if row["eligible_for_broad"]]
    broad = random.Random(seed).sample(population, BROAD_COUNT)
    used = set(ANCHORS) | set(broad)
    selected = []

    def add(p, role, family):
        if estimate(p) > geometry.BUDGET_BYTES:
            raise ValueError(f"Selected shape exceeds conservative planning budget: {p}")
        stage = "holdout" if role == "holdout" else "exploratory"
        selected.append({"id": shape_id(p), "m": p[0], "n": p[1], "k": p[2],
                         "family": family, "stage": stage, "sampling_role": role,
                         "sampling_stratum": family,
                         "on_lattice": all(v in lattice for v in p),
                         "lower_bound_resident_bytes": geometry.resident_lower_bound_bytes(*p),
                         "conservative_pair_buffer_bytes": estimate(p),
                         "useful_cubic_flops": 2 * p[0] * p[1] * p[2],
                         "broad_inclusion_probability": BROAD_COUNT / len(population) if role == "broad" else None,
                         "legacy_shape_id": geometry.shape_id(*p) if role == "anchor" else None})

    for p in ANCHORS:
        add(p, "anchor", "cross_cohort_anchor")
    for p in broad:
        add(p, "broad", "uniform_feasible_grid")
    rejected_focus = []
    for family, count, pool in _focus_pools():
        kept = 0
        for p in pool:
            if p in used or p in historical or estimate(p) > geometry.BUDGET_BYTES:
                rejected_focus.append({"m": p[0], "n": p[1], "k": p[2], "family": family,
                                       "reason": "already_selected" if p in used else
                                       "historical_geometry" if p in historical else "planning_memory"})
                continue
            add(p, "focused", family)
            used.add(p)
            kept += 1
            if kept == count:
                break
        if kept != count:
            raise ValueError(f"Focused pool exhausted for {family}: {kept}/{count}")
    holdout_population = [p for p in population if p not in used]
    holdouts = random.Random(seed + 1).sample(holdout_population, HOLDOUT_COUNT)
    for p in holdouts:
        add(p, "holdout", "new_uniform_holdout")
    all_points = [point(row) for row in selected]
    if len(all_points) != len(set(all_points)) or set(all_points) & {point(row) for row in old_holdouts}:
        raise AssertionError("Geometry duplicated or protected legacy holdout leaked")
    role_ids = {role: [row["id"] for row in selected if row["sampling_role"] == role]
                for role in ("anchor", "broad", "focused", "holdout")}
    if tuple(map(len, role_ids.values())) != (12, BROAD_COUNT, FOCUSED_COUNT, HOLDOUT_COUNT):
        raise AssertionError("Unexpected cohort size")
    manifest = {
        "schema_version": 1, "manifest_id": MANIFEST_ID, "cohort_id": cohort_id,
        "shape_order": ["m", "n", "k"], "legacy_benchmark_tuple_order": ["m", "k", "n"],
        "tile_order": ["bm", "bn", "bk"], "selection_seed": seed,
        "selection_basis": "Broad sample is geometry-only SRS without replacement. Focused hypotheses and repeat anchors are informed by old exploratory results; no new outcomes used.",
        "execution_status": "planned_unmeasured",
        "cohort_policy": "Every new allocation is a distinct cohort. Never pool raw timings, fit rows, or calibration coefficients across allocations; anchors support labeled side-by-side comparisons only.",
        "holdout_policy": "Do not compile, time, accuracy-test, tune or fit new holdouts until the model, feature definitions, coefficients and dispatch threshold are frozen in a hashed artifact. Existing twelve holdouts stay excluded entirely.",
        "historical_inventory": history,
        "protected_legacy_holdout_shape_ids": [row["id"] for row in old_holdouts],
        "protected_legacy_holdout_geometries": [{"m": row["m"], "n": row["n"], "k": row["k"]} for row in old_holdouts],
        "protected_legacy_holdout_shapes_mnk": [list(point(row)) for row in old_holdouts],
        "old_holdout_shapes": [{"id": row["id"], "m": row["m"], "n": row["n"], "k": row["k"]} for row in old_holdouts],
        "old_holdout_source_sha256": history["files_sha256"]["configs/generated_power_grid_v001/sampled_shapes.json"],
        "old_holdout_source_file": "legacy_sampled_shapes_v001.json",
        "broad_sampling_design": {
            "method": "simple_random_sample_without_replacement",
            "algorithm": "Python random.Random(seed).sample(ascending lexicographic eligible geometries, 60)",
            "dimension_values": values, "full_domain_count": len(inventory),
            "planning_budget_bytes": geometry.BUDGET_BYTES,
            "maximum_mnk_product": MAX_BROAD_VOLUME,
            "planning_tiles_bm_bn_bk": TILES,
            "eligible_population_size": len(population),
            "sample_size": BROAD_COUNT, "inclusion_probability": BROAD_COUNT / len(population),
            "inverse_probability_weight": len(population) / BROAD_COUNT,
            "exclusions": ["outside inclusive per-axis range 256..131072", "over conservative 8 GiB candidate-pair planning budget", "M*N*K exceeds 16384^3 to bound exploratory compute", "any geometry in the declared historical configuration inventory, including all legacy holdouts"],
            "estimand": "Win fraction among the eligible, previously unconfigured, finite power/midpoint grid on this allocation; not uniform over raw integer dimensions or representative of application traffic.",
            "analysis": "Use broad rows only for the unweighted SRS win fraction (equivalently equal inverse-probability weights); report uncertainty and missing outcomes. Focused/anchor rows cannot enter that numerator or denominator.",
        },
        "new_holdout_sampling_design": {
            "method": "independent seeded SRS from remaining eligible grid after exploratory geometry selection",
            "selection_seed": seed + 1, "conditional_population_size": len(holdout_population),
            "sample_size": HOLDOUT_COUNT,
            "conditional_inclusion_probability": HOLDOUT_COUNT / len(holdout_population),
            "limitation": "Holdouts evaluate the remaining grid, not local off-lattice boundaries; do not treat this conditional distribution as an application workload.",
        },
        "focused_rejected_geometry_candidates": rejected_focus,
        "exploratory_shape_ids": role_ids["anchor"] + role_ids["broad"] + role_ids["focused"],
        "broad_shape_ids": role_ids["broad"], "focused_shape_ids": role_ids["focused"],
        "anchor_shape_ids": role_ids["anchor"], "holdout_shape_ids": role_ids["holdout"],
        "new_holdout_shape_ids": role_ids["holdout"], "shapes": selected,
    }
    return manifest, inventory


def build_campaign(manifest, root=ROOT):
    base = json.loads((root / "configs/campaign_v1.json").read_text())
    campaign = configuration.build_config(base, manifest, "sampled_shapes.json")
    campaign.update(campaign_id=MANIFEST_ID, cohort_id=manifest["cohort_id"], seed=SEED,
                    status="preregistered_separate_cohort_pending_hardware_smoke")
    campaign["region_study"] = {"new_cohort": True, "pooled_with_prior": False,
                                "allocation_binding": "Bind this cohort ID to exactly one captured machine/allocation identity before execution; reuse only for the same allocation."}
    for offset, phase in enumerate(("GRID-smoke", "GRID-screen", "GRID-confirm")):
        campaign["experiments"][phase]["seed"] = SEED + 100 + offset
    campaign["grid_study"].update(
        scope="One-level tile-local Strassen; one independently identified allocation; resident Gaussian BF16 operands and FP32 output.",
        holdout_policy=manifest["holdout_policy"],
        cohort_policy=manifest["cohort_policy"],
        protected_legacy_holdout_shape_ids=manifest["protected_legacy_holdout_shape_ids"],
        cohort_manifest_id=MANIFEST_ID,
        sampling_roles={row["id"]: row["sampling_role"] for row in manifest["shapes"]},
        primary_scope="complete_call",
        comparison="Frozen Strassen screen winner versus frozen native screen winner, paired on this allocation; report native default separately.",
    )
    return campaign


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generate(output_dir, root=ROOT, cohort_id=COHORT_ID):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    manifest, inventory = build_manifest(root, cohort_id=cohort_id)
    campaign = build_campaign(manifest, root)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "sampled_shapes.json", manifest)
    _write_json(output / "campaign_region_grid_v001.json", campaign)
    with (output / "distributions_v1.json").open("xb") as stream:
        stream.write((root / "configs/distributions_v1.json").read_bytes())
    with (output / "legacy_sampled_shapes_v001.json").open("xb") as stream:
        stream.write((root / "configs/generated_power_grid_v001/sampled_shapes.json").read_bytes())
    with (output / "broad_population_preflight.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    summary = {
        "manifest_id": MANIFEST_ID, "cohort_id": cohort_id,
        "status": "prepared_unmeasured", "seed": SEED,
        "exploratory_count": len(manifest["exploratory_shape_ids"]),
        "new_holdout_count": len(manifest["holdout_shape_ids"]),
        "protected_legacy_holdout_count": len(manifest["protected_legacy_holdout_shape_ids"]),
        "role_counts": dict(Counter(row["sampling_role"] for row in manifest["shapes"])),
        "focused_family_counts": dict(Counter(row["family"] for row in manifest["shapes"] if row["sampling_role"] == "focused")),
        "new_geometry_count": sum(row["sampling_role"] != "anchor" for row in manifest["shapes"]),
        "off_lattice_exploratory_count": sum(not row["on_lattice"] and row["stage"] == "exploratory" for row in manifest["shapes"]),
        "candidate_counts": {name: len(rows) for name, rows in configuration.candidates().items()},
        "planned_screen_shape_candidate_attempts": 52 * len(manifest["exploratory_shape_ids"]),
        "broad_domain_count": len(inventory),
        "broad_feasible_count_before_history_exclusion": sum(row["within_planning_budget"] for row in inventory),
        "broad_eligible_population_size": manifest["broad_sampling_design"]["eligible_population_size"],
        "maximum_selected_pair_buffer_bytes": max(row["conservative_pair_buffer_bytes"] for row in manifest["shapes"]),
        "memory_estimate_limitations": ["not a measured memory peak or TPU feasibility guarantee", "excludes executable/runtime peaks, on-chip VMEM and host memory", "actual concurrently resident candidate arms require runtime preflight"],
        "cohort_policy": manifest["cohort_policy"],
    }
    _write_json(output / "analytical_summary.json", summary)
    _write_json(output / "bundle_manifest.json", {
        "schema_version": 1, "purpose": "Frozen geometry and candidate configuration only; no latency observations",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(geometry.__file__), Path(configuration.__file__), root / "configs/campaign_v1.json")},
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())},
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cohort-id", default=COHORT_ID,
                        help="Identifier bound to exactly one allocation; new allocations need distinct IDs and output directories")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, cohort_id=args.cohort_id), sort_keys=True))


if __name__ == "__main__":
    main()
