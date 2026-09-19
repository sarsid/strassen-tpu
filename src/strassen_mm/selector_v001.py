"""Frozen interpretable performance selector; never certifies numerical safety.

Fit offline from N5 confirmation only. Unknown/cancellation/real-activation
input scopes fall back to native. No N7 performance data is accepted by fit.
This module has no JAX or third-party dependency and never starts a TPU job.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

VERSION = "selector_v001"
CONFIG_SHA256 = "535808e9832f4c800c653f5b51eb8707828b2e27c94e705e5bc26c2543da2a5a"
HELDOUT_SHA256 = "e5ed73f42fdba354ef869f60081edb2f63dc43d1ae984ba6446ad5434a6dac0f"
NATIVE = {"candidate_id": "native", "algorithm": "native", "variant": "plain", "tile": None}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def artifacts(path):
    path = Path(path).resolve()
    if (path / "artifacts" / "results.jsonl").is_file():
        return path / "artifacts"
    if (path / "results.jsonl").is_file():
        return path
    raise ValueError("Supply the explicit N5 confirmation run or canonical artifacts directory")


def check_shape(shape):
    if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("shape must be positive integer (M,K,N)")
    return tuple(shape)


def distance(left, right):
    return math.sqrt(sum((math.log2(a) - math.log2(b)) ** 2 for a, b in zip(left, right)))


def padding_ratio(shape, tile):
    if tile is None:
        return 1.0
    m, k, n = shape
    bm, bn, bk = tile
    mp, kp, nn = [((size + block - 1) // block) * block for size, block in zip((m, k, n), (bm, bk, bn))]
    return (mp * kp * nn) / (m * k * n)


def choose(rule, shape, *, input_scope="unknown", exclude_shape_id=None):
    """Predict from frozen prototypes only; no measured target-shape data."""
    shape = check_shape(shape)
    start = time.perf_counter_ns()
    result = {"shape_mkn": list(shape), "input_scope": input_scope,
              "choice": dict(NATIVE), "label": "native", "reason": "conservative_native_fallback",
              "numerical_accuracy_certified": False}
    parameters = rule["parameters"]
    if input_scope not in rule["allowed_input_scopes"]:
        result["reason"] = "input_scope_not_qualified"
    elif shape[0] < parameters["min_m"] or shape[1] < parameters["min_k"] or shape[2] < parameters["min_n"]:
        result["reason"] = "small_dimension_guard"
    else:
        prototypes = [item for item in rule["prototypes"] if item["shape_id"] != exclude_shape_id]
        if not prototypes:
            result["reason"] = "no_training_prototype"
        else:
            nearest = min(prototypes, key=lambda item: (distance(shape, item["shape_mkn"]), item["shape_id"]))
            dist = distance(shape, nearest["shape_mkn"])
            result.update(prototype_id=nearest["shape_id"], log2_distance=dist)
            if dist > parameters["max_log2_distance"]:
                result["reason"] = "outside_training_neighborhood"
            elif nearest["label"] == "native":
                result["reason"] = "prototype_has_no_confident_custom_advantage"
            else:
                ratio = padding_ratio(shape, nearest["choice"]["tile"])
                result["padding_volume_ratio"] = ratio
                if ratio > parameters["max_padding_volume_ratio"]:
                    result["reason"] = "padding_guard"
                else:
                    result.update(label=nearest["label"], choice=dict(nearest["choice"]),
                                  reason="nearest_confident_prototype")
    result["host_prediction_ns"] = time.perf_counter_ns() - start
    return result


def passing(row):
    return bool(row and row.get("status") == "ok" and (row.get("correctness") or {}).get("pass") is True
                and (row.get("correctness") or {}).get("finite") is True)


def confidence(row, baseline, margin):
    pair = next((item for item in (row or {}).get("comparisons", []) if item.get("reference_arm") == baseline), None)
    if not pair or pair.get("valid_numerical_comparison") is not True:
        return False
    interval = pair.get("speedup_ci95", [])
    return (len(interval) == 2 and all(isinstance(value, (float, int)) and math.isfinite(value) for value in interval)
            and interval[0] > margin and interval[1] >= interval[0])


def choice_from_selection(value):
    if not value:
        raise ValueError("A custom training label cannot use a null screen selection")
    required = ("candidate_id", "algorithm", "variant", "tile")
    result = {key: value[key] for key in required}
    if result["algorithm"] not in ("cubic_full", "strassen"):
        raise ValueError("Unexpected custom selected algorithm")
    return result


def verify_seal(directory):
    seal = load(directory / "artifact_manifest.json")
    for relative, expected in seal["sha256"].items():
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or digest(target) != expected:
            raise ValueError(f"N5 training artifact seal mismatch: {relative}")
    return digest(directory / "artifact_manifest.json")


def fit(confirmation, selections_path, campaign_path):
    directory = artifacts(confirmation)
    campaign_path = Path(campaign_path).resolve()
    if digest(campaign_path) != CONFIG_SHA256:
        raise ValueError("N5-N9 configuration differs from the preregistered source freeze")
    campaign = load(campaign_path)
    held_path = campaign_path.parent / campaign["heldout_shape_manifest"]
    if digest(held_path) != HELDOUT_SHA256:
        raise ValueError("Held-out reservation differs from the preregistration")
    held = [check_shape([item[key] for key in ("m", "k", "n")]) for item in load(held_path)["shapes"]]
    seal_sha = verify_seal(directory)
    summary = load(directory / "summary.json")
    if summary.get("completed") is not True or summary.get("phase") != "N5-confirm":
        raise ValueError("Selector fitting requires a completed N5-confirm run, never held-out observations")
    environment = load(directory / "environment.json")
    if environment.get("qualified_single_v5e") is not True:
        raise ValueError("N5 confirmation is not from the qualified v5e cohort")
    selections = load(selections_path)
    recorded_selection = load(directory / "selection_input_provenance.json")
    if digest(selections_path) != recorded_selection.get("sha256"):
        raise ValueError("Provided selections are not the exact selections used by N5 confirmation")
    if selections.get("phase") != "N5-screen" or selections.get("campaign_sha256") != CONFIG_SHA256:
        raise ValueError("Selector fit requires the preregistered N5-screen selections")
    if selections.get("environment_identity") != environment["identity"]:
        raise ValueError("N5 selection and confirmation machine identities differ")
    expected_ids = set(campaign["training_shape_ids"])
    if set(selections["by_shape"]) != expected_ids:
        raise ValueError("N5 screen selections do not match the frozen training IDs")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
    call_rows = [row for row in rows if row.get("event") == "case_result" and row.get("scope") == "call"]
    by_shape = {}
    for row in call_rows:
        shape_id = row.get("shape_id")
        if shape_id not in expected_ids:
            raise ValueError(f"Confirmation includes an unknown or held-out shape: {shape_id}")
        group = by_shape.setdefault(shape_id, {})
        if row["arm_id"] in group:
            raise ValueError(f"Duplicate confirmation row: {shape_id}/{row['arm_id']}")
        group[row["arm_id"]] = row
    if set(by_shape) != expected_ids:
        raise ValueError("N5 confirmation lacks a training shape")
    settings = campaign["experiments"]["N6a"]
    prototypes = []
    for shape_id in sorted(expected_ids):
        selected = selections["by_shape"][shape_id]
        shape = check_shape(selected["shape_mkn"])
        if shape in held:
            raise ValueError("Training/held-out geometry overlap")
        shape_rows = by_shape[shape_id]
        native = shape_rows.get("native_xla")
        cubic = shape_rows.get("cubic_selected")
        strassen = shape_rows.get("strassen_selected")
        label, candidate = "native", dict(NATIVE)
        margin = settings["confidence_lower_bound_min"]
        if passing(native) and passing(cubic) and passing(strassen) and confidence(strassen, "native_xla", margin) and confidence(strassen, "cubic_selected", margin):
            label, candidate = "strassen", choice_from_selection(selected["strassen"])
        elif passing(native) and passing(cubic) and confidence(cubic, "native_xla", margin):
            label, candidate = "cubic", choice_from_selection(selected["cubic"])
        prototypes.append({"shape_id": shape_id, "shape_mkn": list(shape), "label": label, "choice": candidate,
                           "qualification": {"native_pass": passing(native), "cubic_pass": passing(cubic), "strassen_pass": passing(strassen)},
                           "confirmation_call_means_ms": {name: (row.get("timing") or {}).get("mean_ms") for name, row in shape_rows.items()},
                           "paired_comparisons": {name: row.get("comparisons", []) for name, row in shape_rows.items()}})
    parameter_keys = ("confidence_lower_bound_min", "max_log2_distance", "min_m", "min_k", "min_n", "max_padding_volume_ratio")
    rule = {"schema_version": 1, "model_version": VERSION, "model_type": settings["model"],
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "parameters": {key: settings[key] for key in parameter_keys},
            "allowed_input_scopes": settings["allowed_input_scopes"], "fallback": dict(NATIVE),
            "training_shapes": [item["shape_mkn"] for item in prototypes], "reserved_heldout_shapes": [list(shape) for shape in held],
            "source_identity": environment["identity"], "prototypes": prototypes,
            "source_hashes": {"campaign": CONFIG_SHA256, "heldout_reservation": HELDOUT_SHA256,
                              "confirmation_results": digest(directory / "results.jsonl"), "confirmation_seal": seal_sha,
                              "selections": digest(selections_path), "selector_source": digest(Path(__file__))},
            "training_inputs": {"confirmation_artifacts": str(directory), "selections": str(Path(selections_path).resolve())},
            "scope": "Performance selector under explicitly declared synthetic Gaussian input scope. Shape does not certify numerical accuracy; every measured result still needs the unchanged gate.",
            "known_failure": "All 24 N4 cancellation cases failed Strassen's unchanged gate; unknown and cancellation input scopes therefore route to native.",
            "label_counts": dict(Counter(item["label"] for item in prototypes))}
    cv = []
    for item in prototypes:
        predicted = choose(rule, item["shape_mkn"], input_scope="gaussian_synthetic", exclude_shape_id=item["shape_id"])
        cv.append({"shape_id": item["shape_id"], "observed_confident_label": item["label"],
                   "predicted_label": predicted["label"], "agreement": predicted["label"] == item["label"],
                   "reason": predicted["reason"], "prototype_id": predicted.get("prototype_id")})
    rule["leave_one_shape_out"] = {"cases": cv, "agreement_fraction": sum(item["agreement"] for item in cv) / len(cv),
                                    "scope": "Descriptive label agreement only; no threshold/model retuning and no claim of measured borrowed-tile latency."}
    return rule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fitting = subs.add_parser("fit")
    fitting.add_argument("--confirmation", type=Path, required=True)
    fitting.add_argument("--selection", type=Path, required=True)
    fitting.add_argument("--campaign", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    prediction = subs.add_parser("predict")
    prediction.add_argument("--selector", type=Path, required=True)
    prediction.add_argument("--shape", type=int, nargs=3, required=True)
    prediction.add_argument("--input-scope", default="unknown")
    args = parser.parse_args(argv)
    if args.command == "fit":
        if args.output.exists():
            raise FileExistsError("Frozen selector output already exists")
        rule = fit(args.confirmation, args.selection, args.campaign)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(rule, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"selector": str(args.output), "sha256": digest(args.output), "label_counts": rule["label_counts"],
                          "leave_one_shape_out_agreement": rule["leave_one_shape_out"]["agreement_fraction"]}, sort_keys=True))
    else:
        print(json.dumps(choose(load(args.selector), args.shape, input_scope=args.input_scope), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
