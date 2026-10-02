"""Fit direct native/candidate log margins within one new allocation; CPU only.

Screen ratios are unpaired point estimates. Confirmation is evaluation only.
Outputs are exclusive-created and historical/held-out observations are rejected.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from strassen_mm.power_grid_selector_v001 import geometry

VERSION = "region_margin_v001"
FEATURE_NAMES = ["log2_volume", "log2_m_over_n", "log2_k_over_sqrt_mn",
                 "squared_log2_m_over_n", "squared_log2_k_over_sqrt_mn",
                 "log2_axis_spread", "volume_hinge_36", "volume_hinge_39",
                 "log2_output_tiles", "log2_k_panels", "log2_padding_volume_ratio",
                 "output_tile_count_under_8", "k_panel_count_under_4"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def features(m, n, k, candidate):
    """Shape and declared tile only; never uses timing, winner, or shape ID."""
    g = geometry(m, n, k, candidate)
    lm, ln, lk = map(math.log2, (m, n, k))
    volume, aspect, reduction = lm + ln + lk, lm - ln, lk - (lm + ln) / 2
    tiles, panels = g["logical_output_tiles"], g["sequential_k_panels"]
    return [volume, aspect, reduction, aspect * aspect, reduction * reduction,
            max(lm, ln, lk) - min(lm, ln, lk), max(0., volume - 36),
            max(0., volume - 39), math.log2(tiles) if tiles else 0.,
            math.log2(panels) if panels else 0., math.log2(g["padding_volume_ratio"]),
            max(0., 3 - math.log2(tiles)) if tiles else 0.,
            max(0., 2 - math.log2(panels)) if panels else 0.]


def block_key(shape):
    # Nearby +/- tile-boundary shapes and axis permutations share a block.
    return tuple(sorted(int(math.floor(math.log2(shape[d]) + .5)) for d in ("m", "n", "k")))


def blocked_folds(shapes, nfold=5, seed=20260920):
    require(type(nfold) is int and nfold >= 2, "At least two folds required")
    groups = defaultdict(list)
    for i, shape in enumerate(shapes):
        groups[block_key(shape)].append(i)
    require(len(groups) >= nfold, "Too few distinct geometry blocks for requested folds")
    # Deterministic allocation by geometry only; largest blocks balanced first.
    keys = sorted(groups, key=lambda key: (-len(groups[key]), hashlib.sha256(
        f"{seed}:{key}".encode()).hexdigest()))
    loads, assignment = [0] * nfold, np.empty(len(shapes), dtype=int)
    for key in keys:
        fold = min(range(nfold), key=lambda f: (loads[f], f))
        assignment[groups[key]] = fold
        loads[fold] += len(groups[key])
    return assignment


def fit(x, y, train, ridge=1.):
    """Independent ridge models, normalization and intercept fitted on train."""
    require(math.isfinite(ridge) and ridge > 0, "Ridge strength must be positive")
    require(len(train) >= 2, "At least two training geometries required")
    models = []
    for c in range(y.shape[1]):
        a, b = x[train, c], y[train, c]
        center, scale = a.mean(0), a.std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        a = (a - center) / scale
        offset = float(b.mean())
        coef = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ (b - offset))
        models.append({"center": center.tolist(), "scale": scale.tolist(),
                       "coefficients": coef.tolist(), "intercept": offset})
    return models


def predict(x, models, indices):
    p = np.empty((len(indices), len(models)))
    for c, model in enumerate(models):
        p[:, c] = ((x[indices, c] - model["center"]) / model["scale"]) @ np.array(
            model["coefficients"]) + model["intercept"]
    require(np.isfinite(p).all(), "Nonfinite model predictions")
    return p


def fixed_native(y, train, candidates):
    natives = [i for i, c in enumerate(candidates) if c["family"] == "native"]
    # This comparison never reads held-out labels. Ties use catalog order.
    return max(natives, key=lambda c: float(y[train, c].mean()))


def choose(p, native, candidates, min_speedup=.03, shortlist_size=3):
    require(math.isfinite(min_speedup) and min_speedup >= 0, "Invalid speedup threshold")
    require(type(shortlist_size) is int and shortlist_size > 0, "Invalid shortlist size")
    custom = [i for i, c in enumerate(candidates) if c["family"] == "strassen"]
    require(bool(custom), "Strassen catalog is empty")
    ranked = np.array([sorted(custom, key=lambda c: (-row[c], c))[:shortlist_size] for row in p])
    winner = ranked[:, 0]
    selected = np.where(p[np.arange(len(p)), winner] - p[:, native] > math.log1p(min_speedup),
                        winner, native)
    return selected, ranked


def metrics(y, selected, baseline, shortlist, candidates):
    rows = np.arange(len(y))
    observed, reference, oracle = y[rows, selected], y[rows, baseline], y.max(1)
    regret = np.exp(oracle - observed)
    shortlist_best = np.maximum(reference, np.max(np.take_along_axis(y, shortlist, axis=1), axis=1))
    strassen = np.array([c["family"] == "strassen" for c in candidates])
    selected_strassen = strassen[selected]
    actual_strassen_wins = np.max(y[:, strassen], axis=1) > reference
    return {"shape_count": len(y),
            "geomean_speedup_vs_training_selected_fixed_native": float(np.exp((observed-reference).mean())),
            "geomean_regret_vs_screen_catalog_oracle": float(np.exp((oracle-observed).mean())),
            "fixed_native_geomean_regret_vs_screen_catalog_oracle": float(np.exp((oracle-reference).mean())),
            "p95_regret_vs_screen_catalog_oracle": float(np.quantile(regret, .95)),
            "worst_slowdown_vs_fixed_native": float(np.exp(np.max(reference-observed))),
            "strassen_dispatch_count": int(selected_strassen.sum()),
            "strassen_dispatch_losing_to_fixed_native_count": int(np.sum(selected_strassen & (observed < reference))),
            "screen_strassen_win_count_vs_fixed_native": int(actual_strassen_wins.sum()),
            "shortlist_contains_screen_best_strassen_count": int(sum(
                int(np.flatnonzero(strassen)[np.argmax(row[strassen])]) in short
                for row, short in zip(y, shortlist))),
            "shortlist_oracle_geomean_speedup_vs_fixed_native": float(np.exp((shortlist_best-reference).mean())),
            "shortlist_metric_interpretation": "Retrospective best within predicted shortlist plus native; requires extra tuning and is not dispatched performance.",
            "timing_interpretation": "Unpaired ratios of screen arithmetic means; exploratory, no paired confidence interval or confirmed population win fraction."}


def evaluate(dataset, nfold=5, ridge=1., min_speedup=.03, shortlist_size=3):
    shapes, candidates, y = dataset["shapes"], dataset["candidates"], dataset["y"]
    x = np.array([[features(s["m"], s["n"], s["k"], c) for c in candidates] for s in shapes])
    folds = blocked_folds(shapes, nfold)
    p, baseline = np.empty_like(y), np.empty(len(y), dtype=int)
    selected, shortlist = np.empty(len(y), dtype=int), np.empty((len(y), min(
        shortlist_size, sum(c["family"] == "strassen" for c in candidates))), dtype=int)
    for fold in range(nfold):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        native = fixed_native(y, train, candidates)
        p[test] = predict(x, fit(x, y, train, ridge), test)
        baseline[test] = native
        selected[test], shortlist[test] = choose(p[test], native, candidates, min_speedup, shortlist_size)
    by_role = {}
    for role in sorted({s["sampling_role"] for s in shapes}):
        rows = np.array([i for i, s in enumerate(shapes) if s["sampling_role"] == role])
        by_role[role] = metrics(y[rows], selected[rows], baseline[rows], shortlist[rows], candidates)
    all_rows = np.arange(len(shapes))
    native = fixed_native(y, all_rows, candidates)
    model = {"version": VERSION, "feature_names": FEATURE_NAMES, "target": "log(native_default_mean_ms / candidate_mean_ms)",
             "candidates": candidates, "models": fit(x, y, all_rows, ridge),
             "reference_candidate_id": dataset["reference_candidate_id"],
             "fixed_native_candidate_id": candidates[native]["candidate_id"],
             "ridge_strength": ridge, "minimum_predicted_speedup_fraction": min_speedup,
             "shortlist_size": shortlist_size, "training_shape_ids": [s["id"] for s in shapes],
             "memory_budget_bytes": dataset.get("memory_budget_bytes"),
             "training_shapes_mnk": [[s[d] for d in ("m", "n", "k")] for s in shapes],
             "provenance": dataset["provenance"],
             "status": "experimental_single_allocation_screen_model_not_a_confirmed_dispatch_rule"}
    selections = [{"shape_id": s["id"], "sampling_role": s["sampling_role"],
                   "fold": int(folds[i]), "block": list(block_key(s)),
                   "selected_candidate_id": candidates[selected[i]]["candidate_id"],
                   "fixed_native_candidate_id": candidates[baseline[i]]["candidate_id"],
                   "predicted_log_margin_vs_fixed_native": float(p[i, selected[i]]-p[i, baseline[i]]),
                   "screen_speedup_vs_fixed_native": float(np.exp(y[i, selected[i]]-y[i, baseline[i]])),
                   "strassen_shortlist": [candidates[j]["candidate_id"] for j in shortlist[i]]}
                  for i, s in enumerate(shapes)]
    report = {"version": VERSION, "provenance": dataset["provenance"],
              "validation": "Whole rounded-log geometry blocks held out, axis permutations grouped; candidate labels, feature normalization, and native baseline selection confined to fold training data.",
              "fold_count": nfold, "overall_design_mix": metrics(y, selected, baseline, shortlist, candidates),
              "by_sampling_role": by_role, "out_of_fold_selections": selections,
              "population_interpretation": "Broad, focused and repeated-anchor strata remain separate. Overall metrics describe this design mix, not a uniform grid population.",
              "heldout_shapes_used": [], "confirmation_used_for_fit": False}
    return model, report


def select_shape(model, m, n, k):
    """Prospective inference from a frozen model; no benchmark-data lookup."""
    candidates = model["candidates"]
    x = np.array([[features(m, n, k, c) for c in candidates]])
    prediction = predict(x, model["models"], [0])[0]
    catalog_geometry = [geometry(m, n, k, c) for c in candidates]
    budget = model.get("memory_budget_bytes")
    feasible = [budget is None or g["conservative_single_candidate_bytes"] <= budget for g in catalog_geometry]
    native = next(i for i, c in enumerate(candidates) if c["candidate_id"] == model["fixed_native_candidate_id"])
    custom = sorted((i for i, c in enumerate(candidates) if c["family"] == "strassen" and feasible[i]),
                    key=lambda i: (-prediction[i], i))
    if not feasible[native] or not custom:
        return {"input_mnk": [m, n, k], "status": "no_comparable_catalog_within_memory_budget",
                "selected_candidate": None, "strassen_shortlist": []}
    selected = custom[0] if prediction[custom[0]] - prediction[native] > math.log1p(
        model["minimum_predicted_speedup_fraction"]) else native

    def candidate_row(index):
        return {**candidates[index], **catalog_geometry[index],
                "predicted_log_margin_vs_fixed_native": float(prediction[index] - prediction[native])}

    logs = [math.log2(v) for v in (m, n, k)]
    distance = min(math.sqrt(sum((a-math.log2(b))**2 for a, b in zip(logs, shape)))
                   for shape in model["training_shapes_mnk"])
    return {"input_mnk": [m, n, k], "status": "experimental_prediction",
            "selected_candidate": candidate_row(selected), "fixed_native_candidate": candidate_row(native),
            "strassen_shortlist": [candidate_row(i) for i in custom[:model["shortlist_size"]]],
            "nearest_training_log2_distance": distance,
            "cohort_id": model["provenance"]["cohort_id"],
            "allocation_id": model["provenance"]["allocation_id"],
            "interpretation": "Single-allocation empirical prediction; threshold is not a confidence bound, and new-allocation validation remains necessary."}


def checked_journal(path, environment, config, manifest_path, config_path, phase, cohort_id, allocation_id):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    starts = [r for r in rows if r.get("event") == "run_start"]
    require(len(starts) == 1, "Expected exactly one run_start; concatenated/mixed runs are forbidden")
    require(starts[0].get("allocation_id") == allocation_id, "Run allocation does not match requested allocation")
    require(environment.get("identity", {}).get("allocation_id") == allocation_id, "Environment allocation mismatch")
    require(environment.get("campaign_id") == config.get("campaign_id"), "Environment campaign mismatch")
    require(all(r.get("phase", phase) == phase for r in rows), "Mixed or unexpected phases")
    require(all(r.get("cohort_id", cohort_id) == cohort_id for r in rows), "Mixed cohort IDs")
    require(all(r.get("allocation_id", allocation_id) == allocation_id for r in rows), "Mixed allocation IDs")
    hashes = set(starts[0].get("source_manifest", {}).get("sha256", {}).values())
    require(sha(manifest_path) in hashes and sha(config_path) in hashes,
            "Run snapshot hashes do not match supplied shape manifest and config")
    completed = [r for r in rows if r.get("event") == "run_complete"]
    require(len(completed) == 1 and completed[0].get("completed") is True,
            "Only a completed screen/confirmation run can be analyzed")
    return rows


def load_dataset(manifest_path, config_path, screen_path, environment_path, cohort_id, allocation_id):
    manifest, config, environment = map(read_json, (manifest_path, config_path, environment_path))
    require(manifest.get("cohort_id") == cohort_id and config.get("cohort_id") == cohort_id,
            "New manifest and config must both explicitly declare the requested cohort_id; old artifacts are not accepted")
    rows = checked_journal(screen_path, environment, config, manifest_path, config_path,
                           "GRID-screen", cohort_id, allocation_id)
    experiment = config["experiments"]["GRID-screen"]
    ids = experiment["shape_ids"]
    held = set(manifest.get("holdout_shape_ids", [])) | set(manifest.get("new_holdout_shape_ids", [])) | set(
        manifest.get("protected_legacy_holdout_shape_ids", []))
    require(len(ids) == len(set(ids)) and not set(ids) & held, "Duplicate or protected holdout training IDs")
    require(set(ids) <= set(manifest["exploratory_shape_ids"]), "Screen IDs must be declared exploratory")
    shape_map = {s["id"]: s for s in manifest["shapes"]}
    require(len(shape_map) == len(manifest["shapes"]), "Duplicate shape IDs")
    shapes = [shape_map[sid] for sid in sorted(ids)]
    require(len({(s["m"], s["n"], s["k"]) for s in shapes}) == len(shapes), "Duplicate training geometries")
    protected_geometries = {tuple(v) for v in manifest.get("protected_legacy_holdout_shapes_mnk", [])}
    protected_geometries.update((s["m"], s["n"], s["k"]) for s in manifest["shapes"] if s["id"] in held)
    require(not {(s["m"], s["n"], s["k"]) for s in shapes} & protected_geometries,
            "Protected holdout geometry cannot be trained under an alias")
    for s in shapes:
        require(s.get("sampling_role") in {"broad", "focused", "anchor"}, "Shape sampling_role must be broad, focused, or anchor")
    candidates = sorted([dict(c, family=f, compiler_options=c.get("compiler_options", {}))
                         for f, cs in experiment["candidate_families"].items() for c in cs],
                        key=lambda c: c["candidate_id"])
    catalog = {c["candidate_id"]: c for c in candidates}
    require(len(catalog) == len(candidates), "Duplicate candidate IDs")
    defaults = [c for c in candidates if c["family"] == "native" and not c["compiler_options"]]
    require(len(defaults) == 1, "Exactly one compiler-default native reference required")
    reference = defaults[0]["candidate_id"]
    observed = {}
    repeats = config["timing"][experiment["timing"]]["repeats"]
    seed = experiment.get("seed", config["seed"])
    for row in rows:
        if row.get("event") != "case_result":
            continue
        sid = row.get("shape_id")
        require(sid in ids and sid not in held, "Unexpected/protected holdout result")
        if row.get("scope") != "call":
            continue
        cid = row.get("candidate_id")
        require(cid in catalog, "Unexpected candidate in screen")
        key = (sid, cid)
        require(key not in observed, f"Duplicate screen candidate result: {key}")
        require(row.get("status") == "ok" and row.get("correctness", {}).get("pass") is True
                and row.get("eligible_for_speedup_claim") is True, f"Failed/ineligible candidate: {key}; no silent row dropping")
        c, s = catalog[cid], shape_map[sid]
        require(row.get("shape_mkn") == [s[d] for d in ("m", "k", "n")], "Shape axis mismatch")
        require(all(row.get(k, {} if k == "compiler_options" else None) == c.get(k)
                    for k in ("family", "algorithm", "variant", "tile", "compiler_options")), "Candidate metadata mismatch")
        require(row.get("distribution") == "gaussian" and row.get("seed") == seed, "Mixed input distribution or seed")
        t = row.get("timing", {})
        mean = t.get("mean_ms")
        require(t.get("sample_count") == repeats and isinstance(mean, (int, float))
                and math.isfinite(mean) and mean > 0, "Invalid timing or inconsistent repeat count")
        observed[key] = mean
    require(len(observed) == len(shapes) * len(candidates),
            f"Incomplete candidate catalog: expected {len(shapes)*len(candidates)} rows, received {len(observed)}")
    times = np.array([[observed[s["id"], c["candidate_id"]] for c in candidates] for s in shapes])
    ref = next(i for i, c in enumerate(candidates) if c["candidate_id"] == reference)
    provenance = {"cohort_id": cohort_id, "allocation_id": allocation_id,
                  "environment_identity": environment["identity"],
                  "input_sha256": {"manifest": sha(manifest_path), "config": sha(config_path),
                                   "screen": sha(screen_path), "environment": sha(environment_path)},
                  "cohort_pooling": False, "screen_ratio_pairing": "unpaired_cross_candidate_group_point_estimates"}
    return {"shapes": shapes, "candidates": candidates, "y": np.log(times[:, ref, None] / times),
            "reference_candidate_id": reference, "provenance": provenance,
            "memory_budget_bytes": int(config["memory"]["estimated_live_device_budget_gib"] * 2**30)
            if "memory" in config else None}


def confirmation_check(path, environment_path, manifest_path, config_path, dataset, selections):
    environment, config = read_json(environment_path), read_json(config_path)
    provenance = dataset["provenance"]
    require(environment.get("identity") == provenance["environment_identity"],
            "Confirmation environment changed; analyze another allocation separately")
    rows = checked_journal(path, environment, config, manifest_path, config_path, "GRID-confirm",
                           provenance["cohort_id"], provenance["allocation_id"])
    shapes = {s["id"] for s in dataset["shapes"]}
    groups = defaultdict(dict)
    for row in rows:
        if row.get("event") != "case_result" or row.get("scope") != "call":
            continue
        require(row.get("shape_id") in shapes, "Confirmation contains unexpected/held-out geometry")
        key = (row["shape_id"], row["group_id"], row.get("seed"), row.get("distribution"))
        require(row["candidate_id"] not in groups[key], "Duplicate candidate in confirmation group")
        groups[key][row["candidate_id"]] = row
    results = []
    for action in selections:
        sid, cid, baseline = (action[k] for k in ("shape_id", "selected_candidate_id", "fixed_native_candidate_id"))
        item = {"shape_id": sid, "selected_candidate_id": cid, "fixed_native_candidate_id": baseline,
                "covered": False}
        if cid == baseline:
            item.update(covered=True, speedup_vs_fixed_native=1., reason="native_fallback_identity")
        else:
            matches = sorted((key, group) for key, group in groups.items()
                             if key[0] == sid and cid in group and baseline in group)
            if matches:
                key, group = matches[0]
                pair = [group[cid], group[baseline]]
                if all(r.get("status") == "ok" and r.get("correctness", {}).get("pass") is True
                       and r.get("eligible_for_speedup_claim") is True
                       and isinstance(r.get("timing", {}).get("mean_ms"), (int, float))
                       and math.isfinite(r["timing"]["mean_ms"]) and r["timing"]["mean_ms"] > 0 for r in pair):
                    require(pair[0]["timing"]["sample_count"] == pair[1]["timing"]["sample_count"], "Confirmation repeat mismatch")
                    item.update(covered=True, group_id=key[1], speedup_vs_fixed_native=
                                pair[1]["timing"]["mean_ms"] / pair[0]["timing"]["mean_ms"])
        results.append(item)
    return {"confirmation_sha256": sha(path), "environment_sha256": sha(environment_path),
            "interpretation": "Coverage of frozen OOF choices against the fold-frozen native preset, same timed group only. Missing coverage is not imputed; native fallback is the identical action. Point estimates only; confirmation never refits the model.",
            "covered_count": sum(r["covered"] for r in results), "rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "screen", "environment", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.)
    parser.add_argument("--minimum-speedup", type=float, default=.03)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--confirm", type=Path)
    parser.add_argument("--confirmation-environment", type=Path)
    args = parser.parse_args()
    require(bool(args.confirm) == bool(args.confirmation_environment), "Confirmation requires its own environment file")
    require(not args.out_dir.exists(), "Output directory already exists; choose a new immutable output path")
    dataset = load_dataset(args.manifest, args.config, args.screen, args.environment, args.cohort_id, args.allocation_id)
    model, report = evaluate(dataset, args.folds, args.ridge, args.minimum_speedup, args.shortlist_size)
    if args.confirm:
        report["confirmation"] = confirmation_check(args.confirm, args.confirmation_environment, args.manifest,
                                                    args.config, dataset, report["out_of_fold_selections"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.out_dir / "margin_model.json", model)
    write_new(args.out_dir / "margin_evaluation.json", report)
    print(json.dumps({"status": "complete", "cohort_id": args.cohort_id,
                      "shape_count": len(dataset["shapes"]), "output_directory": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
