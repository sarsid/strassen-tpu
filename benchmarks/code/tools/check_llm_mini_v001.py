#!/usr/bin/env python3
"""Validate the frozen 18-shape mini-MM design without importing JAX or timing.

This is a configuration/provenance-consistency check, not model qualification,
hardware memory measurement, or a substitute for the final evidence audit.
The verified model-configuration plan supplies dimensions; no network is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from strassen_mm import benchmark_power_grid_v001 as grid
from strassen_mm import benchmark_n5_v001 as tuning

TILES = ((512, 1024, 512), (1024, 1024, 512), (1024, 1024, 1024),
         (2048, 1024, 512), (1024, 2048, 512), (2048, 2048, 512))
MODELS = {
    "Qwen/Qwen3-8B": (4096, 12288),
    "mistralai/Mistral-7B-v0.3": (4096, 14336),
    "google/gemma-3-12b-pt": (3840, 15360),
}
M_VALUES = (2048, 8192, 16384)
GATE = {"max_abs_atol": 0.001, "max_abs_reference_rtol": 0.05,
        "relative_l2_max": 0.02, "require_finite": True}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def bundle_file(campaign_path, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Bundle filenames must be relative")
    path = (campaign_path.parent / relative).resolve()
    require(path.is_relative_to(campaign_path.parent) and path.is_file(), "Bundle input missing or escapes its directory")
    return path


def indexed(rows, key, count, context):
    require(isinstance(rows, list) and len(rows) == count, f"{context}: expected {count} records")
    result = {row[key]: row for row in rows}
    require(len(result) == count, context + ": duplicate keys")
    return result


def validate(campaign_path, plan_path):
    campaign_path, plan_path = campaign_path.resolve(), plan_path.resolve()
    campaign, plan = read(campaign_path), read(plan_path)
    shapes_path = bundle_file(campaign_path, campaign["shape_manifest"])
    distributions_path = bundle_file(campaign_path, campaign["distribution_manifest"])
    manifest = read(shapes_path)
    base_path, original_distributions = ROOT / "configs/campaign_v1.json", ROOT / "configs/distributions_v1.json"
    base = read(base_path)
    require(plan.get("plan_id") == "llm_large_shapes_v001" and plan.get("shape_count") == 18,
            "Unexpected verified geometry plan")
    require(plan.get("m_values") == list(M_VALUES), "Plan M dimensions changed")
    models = indexed(plan["models"], "model_id", 3, "Plan models")
    require(set(models) == set(MODELS), "Wrong model cohort")
    for name, (hidden, inner) in MODELS.items():
        model = models[name]
        require((model["hidden_size"], model["intermediate_size"]) == (hidden, inner), "Model dimensions changed: " + name)
        require(re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is not None, "Model revision is not frozen")
        require(re.fullmatch(r"[0-9a-f]{64}", model["config_sha256"]) is not None, "Model config SHA256 missing")
        source = model["source_config"]
        source = source["text_config"] if model["config_path"] == "$.text_config" else source
        require((source["hidden_size"], source["intermediate_size"]) == (hidden, inner),
                "Embedded official config contradicts dimensions: " + name)
        require(model["config_url"] == "https://huggingface.co/" + name + "/resolve/" + model["revision"] + "/config.json",
                "Official config URL is not bound to its frozen model revision")

    source_shapes = indexed(plan["shapes"], "id", 18, "Plan shapes")
    shapes = indexed(manifest["shapes"], "id", 18, "Campaign shapes")
    provenance_ids, geometry, by_model = set(), set(), Counter()
    for name, shape in shapes.items():
        source_id = shape.get("source_plan_shape_id", name)
        require(source_id in source_shapes and source_id not in provenance_ids, "Missing or duplicate source-plan shape binding")
        provenance_ids.add(source_id)
        original = source_shapes[source_id]
        require(all(shape.get(field) == original.get(field) for field in
                    ("m", "k", "n", "model_id", "projection", "shape_mkn")), "Campaign changed an official-plan geometry")
        model_id, projection = shape["model_id"], shape["projection"]
        require(model_id in MODELS and shape["m"] in M_VALUES, "Unexpected model or batch/token dimension")
        hidden, inner = MODELS[model_id]
        expected = (shape["m"], hidden, 2 * inner) if projection == "gate_up_concat" else (shape["m"], inner, hidden)
        require(projection in ("gate_up_concat", "down"), "Unexpected projection")
        actual = tuple(shape[axis] for axis in ("m", "k", "n"))
        require(actual == expected and shape["shape_mkn"] == list(expected), "M/K/N coordinate order or values changed")
        require(actual not in geometry, "Duplicate actual matrix geometry")
        geometry.add(actual)
        require(shape.get("geometry_source_revision") == models[model_id]["revision"] and
                shape.get("geometry_source_config_sha256") == models[model_id]["config_sha256"],
                "Shape provenance differs from its model config")
        require(shape["weight_elements"] == shape["k"] * shape["n"], "Wrong logical weight element count")
        require(shape["nominal_operand_output_bytes"] == 2 * (shape["m"] * shape["k"] + shape["k"] * shape["n"]) +
                4 * shape["m"] * shape["n"], "Wrong logical BF16-input/FP32-output byte count")
        by_model[model_id] += 1
    require(provenance_ids == set(source_shapes) and by_model == Counter({name: 6 for name in MODELS}),
            "Each of the three models must contribute six exact plan shapes")
    for model_id in MODELS:
        require({(s["m"], s["projection"]) for s in shapes.values() if s["model_id"] == model_id} ==
                set(itertools.product(M_VALUES, ("gate_up_concat", "down"))), "Incomplete model/projection/M cross product")

    for key in ("precision", "correctness", "memory", "timing"):
        require(campaign[key] == base[key], "Frozen base " + key + " contract changed")
    require(campaign["correctness"]["gate"] == GATE, "Accuracy thresholds changed")
    require(campaign["correctness"]["sample_rows"] == campaign["correctness"]["sample_columns"] == 128,
            "Numerical sample dimensions changed")
    require(read(distributions_path) == read(original_distributions), "Input distribution definition changed")
    require(campaign["device"]["target"] == "v5e" and campaign["device"]["single_device"] is True and
            campaign["device"]["require_identity_match"] is True and campaign["device"]["reuse_same_allocation"] is True and
            campaign["device"]["allow_concurrent_timing"] is False and campaign["device"]["allow_v6e"] is False,
            "Single identified v5e execution contract changed")
    require(campaign["memory"]["estimated_live_device_budget_gib"] == 8 and
            campaign["memory"]["kernel_vmem_limit_mib"] == 48, "Memory budgets changed")

    experiments = campaign["experiments"]
    require(set(experiments) == set(grid.PHASES), "Unexpected campaign phases")
    seeds = []
    for phase in grid.PHASES:
        exp = experiments[phase]
        require(exp["distribution"] == "gaussian", "Timing input distribution must be Gaussian")
        seed = exp.get("seed", campaign.get("seed"))
        require(type(seed) is int and seed >= 0, "Invalid phase input seed")
        seeds.append(seed)
        require(len(exp["shape_ids"]) == len(set(exp["shape_ids"])), "Duplicate phase shape ID")
        require(set(exp["shape_ids"]) <= set(shapes), "Unknown phase shape ID")
        require(exp.get("shape_manifest", campaign["shape_manifest"]) == campaign["shape_manifest"],
                "Phase-specific shape manifest changed")
        expected_timing = {"GRID-smoke": "smoke", "GRID-screen": "screen", "GRID-confirm": "confirm"}[phase]
        require(exp["timing"] == expected_timing, "Wrong timing policy for phase")
    require(len(set(seeds)) == 3, "Smoke, screen and confirmation must use distinct seeds")
    for phase in ("GRID-screen", "GRID-confirm"):
        require(set(experiments[phase]["shape_ids"]) == set(shapes), "Screen/confirmation must each cover all eighteen shapes")
    require(len(experiments["GRID-smoke"]["shape_ids"]) == 2, "Smoke must contain two shapes")
    require(any(shapes[name]["model_id"].startswith("google/") for name in experiments["GRID-smoke"]["shape_ids"]),
            "Smoke must exercise Gemma's nondivisible hidden dimension")

    screen = experiments["GRID-screen"]
    require(grid.validate_candidate_space(screen) == {"native": 1, "cubic": 6, "strassen": 6}, "Wrong candidate attempt budget")
    families = screen["candidate_families"]
    for family in ("cubic", "strassen"):
        require([tuple(c["tile"]) for c in families[family]] == list(TILES), "Custom tile list/order differs from preregistration")
    require(grid.selection_policy(screen) == {"top_k": 1, "near_fraction": 0, "max_confirm_per_family": 1},
            "Confirmation must use only each frozen screen winner")
    require(experiments["GRID-confirm"].get("candidate_families", families) == families,
            "Confirmation candidate specification differs from screening")
    smoke_space = experiments["GRID-smoke"]["candidate_families"]
    require(grid.validate_candidate_space({**screen, "candidate_families": smoke_space}) ==
            {"native": 1, "cubic": 1, "strassen": 1}, "Smoke must use one candidate per family")
    for family in grid.FAMILIES:
        allowed = [grid.candidate_spec(c) for c in families[family]]
        require(all(grid.candidate_spec(c) in allowed for c in smoke_space[family]), "Smoke introduces an unregistered candidate")

    # This is a mocked eligible selection for plan expansion, never experimental
    # evidence or a suggested tuning winner. Every possible pair is checked below.
    selection = {"by_shape": {name: {
        family: {"winner": grid.candidate_spec(families[family][0]),
                 "confirmation_candidates": [grid.candidate_spec(families[family][0])]}
        for family in grid.FAMILIES} for name in shapes}}
    groups = {phase: grid.build_groups(campaign, manifest, phase, selection if phase == "GRID-confirm" else None)
              for phase in grid.PHASES}
    group_counts = {phase: len(values) for phase, values in groups.items()}
    require(group_counts == {"GRID-smoke": 2, "GRID-screen": 108, "GRID-confirm": 18}, "Expanded group count mismatch")
    scope_counts = {phase: sum(len(g["arms"]) * len(g["inputs"]) * len(g["scopes"]) for g in values)
                    for phase, values in groups.items()}
    require(scope_counts == {"GRID-smoke": 12, "GRID-screen": 468, "GRID-confirm": 108}, "Scoped outcome count mismatch")
    for phase, values in groups.items():
        require(len({g["group_id"] for g in values}) == len(values), "Duplicate expanded group ID")
        for group in values:
            require(group["scopes"] == ["call", "prepared_kernel"] and len(group["inputs"]) == 1,
                    "Timing/correctness scope expansion changed")

    budget = 8 * 1024**3
    estimates = []
    for phase, values in groups.items():
        for group in values:
            estimate = tuning.group_memory_estimate(group)
            require(estimate["estimated_live_device_bytes"] <= budget, "A planned group exceeds the unchanged 8GiB preflight")
            estimates.append({"phase": phase, "shape_id": group["shape"]["id"], **estimate})
    worst = {}
    for name, shape in shapes.items():
        maximum = None
        for cubic, strassen in itertools.product(families["cubic"], families["strassen"]):
            group = {"shape": shape, "arms": [copy.deepcopy(families["native"][0]), cubic, strassen]}
            estimate = tuning.group_memory_estimate(group)
            require(estimate["estimated_live_device_bytes"] <= budget, "A possible independently selected confirmation pair exceeds 8GiB")
            if maximum is None or estimate["estimated_live_device_bytes"] > maximum["estimated_live_device_bytes"]:
                maximum = {**estimate, "cubic_tile": cubic["tile"], "strassen_tile": strassen["tile"]}
        worst[name] = maximum
    require("jax" not in sys.modules and "numpy" not in sys.modules,
            "Configuration validation unexpectedly imported an execution dependency")
    return {"passed": True, "scope": "configuration only; no device execution, model weights or timing",
            "group_counts": group_counts, "scoped_outcome_counts_if_families_eligible": scope_counts,
            "counts_note": "Mock selection expands all three eligible families; actual failures remain recorded and may remove a family from confirmation.",
            "models": dict(by_model), "phase_seeds": dict(zip(grid.PHASES, seeds)),
            "candidate_attempt_budget_per_shape": {"native": 1, "cubic": 6, "strassen": 6},
            "possible_confirmation_pairs_checked": len(shapes) * 6 * 6,
            "worst_estimated_group_gib": max(e["estimated_live_device_bytes"] for e in worst.values()) / 1024**3,
            "memory_estimates_are_not_measured_peaks": True, "worst_confirmation_memory_by_shape": worst,
            "qualification_claim": False, "geometric_generalization_claim": False,
            "source_sha256": {str(p): digest(p) for p in (campaign_path, plan_path, shapes_path, distributions_path,
                                                          base_path, original_distributions, Path(__file__),
                                                          Path(grid.__file__), Path(tuning.__file__))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "configs/llm_large_shapes_v001/campaign.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "plans/llm_large_shapes_v001/model_shapes.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    if output is None and os.environ.get("STRASSEN_EXECUTION_DIR"):
        output = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    try:
        result = validate(args.campaign, args.plan)
    except Exception as error:
        result = {"passed": False, "scope": "configuration only", "error": str(error), "traceback": traceback.format_exc()}
    if output is not None:
        with (output / "summary.json").open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        with (output / "artifact_manifest.json").open("x") as handle:
            json.dump({"sha256": {"summary.json": digest(output / "summary.json")}}, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
