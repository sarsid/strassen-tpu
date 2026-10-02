#!/usr/bin/env python3
"""Descriptive findings from completed, sealed power-grid experiments only.

No kernels, model fitting, interpolation, new data inference or winner reselection
occur here. The frozen plotter Evidence reader verifies same-cohort provenance,
raw means/rounds, frozen winners and canonical headline confidence intervals.
This companion adds sealed-input and planned-result completeness requirements.
Version 002 additionally replays every reported near-candidate confidence
interval with the frozen seed; this audits an existing estimand, not a new test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics


VERSION = "summarize_power_grid_v002"
PLOTTER_PATH = Path(__file__).with_name("plot_power_grid_v001.py")
SPEC = importlib.util.spec_from_file_location("power_grid_descriptive_evidence", PLOTTER_PATH)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)

CAUTIONS = [
    "Geometries were deliberately selected. Nominal pointwise win/loss counts describe this sample, not prevalence over the 39,304-point lattice.",
    "Headline choices are frozen screening winners. Confirmation minima are not reselected; shortlist controls in other groups are not pooled into headline timings.",
    "Speedup is reference mean latency / candidate mean latency. Above one favors the candidate. A confidence interval crossing one is inconclusive, not equivalent.",
    "Intervals are pointwise paired 95% bootstrap intervals without multiplicity correction and do not quantify between-allocation or between-day variability.",
    "Complete call and prepared kernel are separate scopes. Compilation and initial host transfer are excluded; complete call is not end-to-end application latency.",
    "Native denotes the frozen tuned compiler-option choice; native_default denotes library defaults. Native tiles are compiler-managed, and identical routes are not algorithmic improvements.",
    "Near-tile sets contain actual confirmed tuples relative to the frozen family winner, not a global optimum. Axis projections do not certify their Cartesian product.",
    "Pointwise CI-qualified five-percent sets do not establish simultaneous equivalence, untested-tile guarantees, or a globally best tile range.",
    "Tiny shapes are sensitive to host/synchronization latency floors. Effective classical-equivalent throughput is not hardware utilization.",
    "Source operation counts and original-buffer bytes are analytical covariates, not measured executed instructions, traffic, peak memory, bandwidth or MXU/vector overlap.",
    "One Strassen level is applied per tile-panel, so fixed-tile implementations both remain Theta(MNK). Larger global size does not by itself eliminate repeated pre-add overhead.",
    "This campaign uses the registered Gaussian BF16-input/pre-add and FP32-accumulation/output contract. Sampled numerical checks do not certify arbitrary inputs, cancellation robustness, or real-model quality.",
    "Reserved holdouts remain unmeasured. This analysis fits no predictor, observes no new holdout, and claims no new device profiling or stress-suite result.",
    "Off-lattice boundary probes are explicitly labeled and are not snapped onto lattice coordinates.",
]


def dimensions_for(ev, shape_id):
    shape = ev.shapes[shape_id]
    return {"shape_id": shape_id, "m": shape["m"], "n": shape["n"], "k": shape["k"],
            "design_family": shape.get("family"), "on_lattice": shape.get("on_lattice", True)}


def audited_sources(ev):
    """Require canonical seals to cover every consumed file in each artifact dir."""
    for directory in ev.artifacts.values():
        seal_path = directory / "artifact_manifest.json"
        if not seal_path.is_file():
            raise ValueError(f"Final sealed evidence required: {seal_path}")
        hashes = json.loads(seal_path.read_text()).get("sha256", {})
        if not hashes:
            raise ValueError(f"Missing canonical sha256 map: {seal_path}")
        for source, metadata in ev.sources.items():
            path = Path(source)
            try:
                relative = str(path.relative_to(directory))
            except ValueError:
                continue
            if relative == "artifact_manifest.json":
                continue
            if hashes.get(relative) != metadata["sha256"]:
                raise ValueError(f"Consumed input is unsealed or changed: {path}")
    return "All consumed artifact files covered by canonical seals; manifest bound by frozen selection hash"


def planned_completeness(ev):
    """Reconcile scoped outcomes with registered group/arm/input/scope products."""
    report = {}
    for tag in ("screen", "confirmation"):
        plan = ev.metadata[tag].get("planned_cases.json")
        summary = ev.metadata[tag]["summary.json"]
        if not isinstance(plan, list) or not plan:
            raise ValueError(f"Nonempty final planned_cases required: {tag}")
        group_ids = [g["group_id"] for g in plan]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"Duplicate planned groups: {tag}")
        completed = summary.get("completed_groups", [])
        if len(completed) != len(set(completed)) or set(completed) != set(group_ids):
            raise ValueError(f"Completed group inventory differs from plan: {tag}")
        if summary.get("planned_group_count") != len(plan):
            raise ValueError(f"Planned group count differs from actual plan: {tag}")
        expected = Counter((g["group_id"], a["candidate_id"], scope, item["distribution"], item["seed"])
                           for g in plan for a in g["arms"] for scope in g["scopes"] for item in g["inputs"])
        actual = Counter((r.get("group_id"), r.get("candidate_id"), r.get("scope"), r.get("distribution"), r.get("seed"))
                         for r in ev.results if r["_cohort"] == tag)
        if expected != actual or any(count != 1 for count in expected.values()):
            raise ValueError(f"Planned/recorded scoped outcomes differ: {tag}; missing={list((expected-actual).elements())[:3]}, extra={list((actual-expected).elements())[:3]}")
        report[tag] = {"planned_groups": len(plan), "scoped_outcomes": sum(actual.values()), "complete": True}
    return report


def comparison_counts(comparisons, exploratory_count):
    result = []
    for scope in plot.SCOPES:
        for reference, candidate in plot.PAIRS:
            rows = [r for r in comparisons if r["scope"] == scope and r["reference"] == reference and r["candidate"] == candidate]
            counts = Counter(r["classification"] for r in rows)
            result.append({"scope": scope, "reference": reference, "candidate": candidate,
                "win": counts["win"], "inconclusive": counts["inconclusive"], "loss": counts["loss"],
                "eligible_comparison_count": len(rows), "unavailable_comparison_count": exploratory_count - len(rows),
                "identical_spec_comparison_count": sum(bool(r["same_executable_spec"]) for r in rows),
                "median_observed_speedup": statistics.median(r["speedup"] for r in rows) if rows else None,
                "min_observed_speedup": min((r["speedup"] for r in rows), default=None),
                "max_observed_speedup": max((r["speedup"] for r in rows), default=None),
                "estimand": "Nominal pointwise classifications for deliberate exploratory sample; not population/lattice prevalence"})
    return result


def source_covariates(shape, row):
    """Analytical source counts, with explicit nulls for native hidden work."""
    m, n, k = (shape[axis] for axis in ("m", "n", "k"))
    record = {"useful_cubic_flops": 2 * m * n * k,
              "original_input_and_one_output_bytes": 2 * m * k + 2 * k * n + 4 * m * n,
              "buffer_byte_estimand": "BF16 A/B and one FP32 C allocation reference; not peak liveness or warm-call traffic",
              "operation_count_estimand": "Source-level padded tile-panel work; not measured executed instruction counts",
              "estimated_padded_dot_flops": None, "estimated_vector_element_operations": None,
              "same_tile_incremental_vector_per_saved_dot_flop": None,
              "parallel_output_tiles": None, "sequential_k_panels": None,
              "parallel_output_tiles_estimand": "Logical independent output tiles in the program grid; not measured physical concurrency or occupancy",
              "padded_m": None, "padded_n": None, "padded_k": None,
              "padding_m_ratio": None, "padding_n_ratio": None, "padding_k_ratio": None,
              "padding_volume_ratio": None, "tile_bm_bn_bk": row.get("tile") if row else None}
    if row is None:
        record["status"] = "no_selected_result"
        return record
    if plot.finite(plot.mean_ms(row), True):
        record["effective_classical_equivalent_tflops_per_second"] = 2 * m * n * k / plot.mean_ms(row) / 1e9
    if row.get("algorithm") == "native":
        record["status"] = "native_internal_tiling_padding_and_work_not_observed"
        return record
    metadata = row.get("kernel_metadata") or {}
    tile = row.get("tile") or metadata.get("tile_bm_bn_bk")
    padded = metadata.get("padded_shape_mkn")
    if not tile or not padded:
        record["status"] = "missing_compiled_kernel_metadata"
        return record
    bm, bn, bk = tile
    mp, kp, np_ = padded
    if any(type(v) is not int or v <= 0 for v in (bm, bn, bk, mp, kp, np_)):
        raise ValueError("Nonpositive or noninteger source metadata")
    expected = [math.ceil(m / bm) * bm, math.ceil(k / bk) * bk, math.ceil(n / bn) * bn]
    if list(padded) != expected:
        raise ValueError("Padded metadata does not match registered tile and original dimensions")
    panels = (mp // bm) * (np_ // bn) * (kp // bk)
    volume = mp * np_ * kp
    record.update(status="analytical_metadata_available", tile_bm_bn_bk=list(tile),
                  padded_m=mp, padded_n=np_, padded_k=kp,
                  padding_m_ratio=mp / m, padding_n_ratio=np_ / n, padding_k_ratio=kp / k,
                  padding_volume_ratio=volume / (m * n * k),
                  parallel_output_tiles=(mp // bm) * (np_ // bn), sequential_k_panels=kp // bk,
                  tile_panel_products=panels)
    if row.get("algorithm") == "strassen":
        record.update(estimated_padded_dot_flops=7 * volume / 4,
                      estimated_vector_element_operations=panels * (5 * bk * (bm + bn) / 4 + 3 * bm * bn),
                      same_tile_incremental_vector_per_saved_dot_flop=5 / bm + 5 / bn + 8 / bk,
                      conditional_dot_saving_against_same_padded_full_cubic_flops=volume / 4,
                      conditional_estimate_note="Counterfactual full cubic with exactly the same tile/padding; not comparison to independently selected cubic")
    elif row.get("algorithm") == "cubic_full":
        record.update(estimated_padded_dot_flops=2 * volume,
                      estimated_vector_element_operations=panels * bm * bn)
    else:
        raise ValueError(f"Unregistered custom headline algorithm: {row.get('algorithm')}")
    return record


def near_tile_sets(ev):
    sets, candidates = [], []
    report_path = str(ev.artifacts["confirmation"] / "confirmation.json")
    grouped_rows = defaultdict(list)
    for row in ev.results:
        if row["_cohort"] == "confirmation":
            grouped_rows[(row["group_id"], row["candidate_id"], row["scope"])].append(row)
    for shape_id in ev.exploratory:
        shape_report = ev.confirmation["by_shape"][shape_id]
        for family in plot.FAMILIES:
            report = shape_report[family]
            for candidate in report.get("candidates", []):
                for scope, observation in candidate.get("scopes", {}).items():
                    reference_id = (report.get("frozen_winner") or {}).get("candidate_id")
                    lhs = grouped_rows[(candidate.get("group_id"), candidate["candidate_id"], scope)]
                    rhs = grouped_rows[(candidate.get("group_id"), reference_id, scope)]
                    if len(lhs) != 1 or len(rhs) != 1 or not plot.eligible(lhs[0]) or not plot.eligible(rhs[0]):
                        raise ValueError("Near-set candidate lacks an eligible same-group measured reference")
                    measured_mean, reference_mean = plot.mean_ms(lhs[0]), plot.mean_ms(rhs[0])
                    if (not math.isclose(observation["mean_ms"], measured_mean, rel_tol=1e-10, abs_tol=1e-12)
                            or not math.isclose(observation["latency_ratio_to_frozen_winner"], measured_mean / reference_mean, rel_tol=1e-10, abs_tol=1e-12)):
                        raise ValueError("Near-set means/ratios disagree with same-group raw-verified results")
                    ci = observation.get("latency_ratio_ci95")
                    if (not isinstance(ci, list) or len(ci) != 2 or not all(plot.finite(v, True) for v in ci)
                            or ci[0] > ci[1] or bool(observation["certified_within_5_percent"]) != (ci[1] <= 1.05)):
                        raise ValueError("Near-set interval or tolerance flag is inconsistent")
                    # The runner bootstraps reference/candidate then inverts
                    # [low, high] to [1/high, 1/low] for its latency ratio.
                    bootstrap_seed = ev.campaign["seed"] + 711
                    replay = plot.paired_ratio(rhs[0], lhs[0], ev.raw(rhs[0]), ev.raw(lhs[0]), seed=bootstrap_seed)
                    replay_interval = [1 / replay["ci95_high"], 1 / replay["ci95_low"]]
                    if not all(math.isclose(saved, calculated, rel_tol=1e-10, abs_tol=1e-12)
                               for saved, calculated in zip(ci, replay_interval)):
                        raise ValueError("Near-set archived interval differs from exact-seed paired bootstrap replay")
                    candidates.append({**dimensions_for(ev, shape_id), "family": family, "scope": scope,
                        "candidate_id": candidate["candidate_id"], "tile_bm_bn_bk": candidate.get("tile"),
                        "compiler_options": candidate.get("compiler_options", {}),
                        "paired_group_id": candidate.get("group_id"), "confirmed_eligible": candidate.get("confirmed_eligible"),
                        "frozen_reference_candidate_id": reference_id,
                        **observation, "source_file": report_path, "source_refs": [lhs[0]["_ref"], rhs[0]["_ref"]],
                        "interval_audit": "Archived candidate/reference interval independently replayed from exact same-group rounds",
                        "interval_bootstrap_seed": bootstrap_seed, "paired_round_count": replay["paired_round_count"]})
            for scope in plot.SCOPES:
                observed = [c for c in report.get("candidates", []) if scope in c.get("scopes", {})]
                for percent in (1, 3, 5):
                    label = f"descriptive_{percent}_percent"
                    members = [c for c in observed if c["scopes"][scope]["latency_ratio_to_frozen_winner"] <= 1 + percent / 100]
                    sets.append(near_set_row(ev, shape_id, family, scope, label, members, report, report_path))
                label = "pointwise_ci95_certified_5_percent"
                members = [c for c in observed if c["scopes"][scope]["certified_within_5_percent"]]
                sets.append(near_set_row(ev, shape_id, family, scope, label, members, report, report_path))
    return sets, candidates


def near_set_row(ev, shape_id, family, scope, label, members, report, report_path):
    expected = [list(c["tile"]) for c in members if c.get("tile") is not None]
    recorded = report.get("near_sets", {}).get(scope, {}).get(label)
    if recorded is None or sorted(expected) != sorted(recorded.get("tuples_bm_bn_bk", [])):
        raise ValueError(f"Near tuple projection does not match confirmed members: {shape_id}/{family}/{scope}/{label}")
    if any(not c.get("confirmed_eligible") or not c.get("reference_available") for c in members):
        raise ValueError("Ineligible or unpaired candidate in measured near set")
    return {**dimensions_for(ev, shape_id), "family": family, "scope": scope, "definition": label,
            "candidate_count": len(members), "candidate_ids": [c["candidate_id"] for c in members],
            "measured_tuple_count": len(expected), "tuples_bm_bn_bk": expected,
            "axis_values": recorded.get("axis_values", {}),
            "native_note": "Native members are compiler presets, not tiles" if family == "native" else None,
            "estimand": "Latency tolerance relative to frozen screen family winner, using same-group confirmation controls",
            "scope_note": "Discrete tested members only; axis values do not certify the Cartesian product; pointwise intervals are not simultaneous guarantees",
            "source_file": report_path}


def describe(ev):
    comparisons = [{**dimensions_for(ev, r["shape_id"]), **r,
                    "candidate_latency_reduction_fraction": 1 - 1 / r["speedup"]}
                   for r in ev.comparisons]
    shape_rows, covariates, padded_comparisons = [], [], []
    for shape_id in ev.exploratory:
        shape = ev.shapes[shape_id]
        result = {**dimensions_for(ev, shape_id), "headline_complete_call": {}, "prepared_kernel": {},
                  "complete_call_comparisons": {}, "prepared_kernel_comparisons": {}}
        cost_by_family = {}
        for family in (*plot.FAMILIES, "native_default"):
            for scope in plot.SCOPES:
                row = ev.selected(shape_id, scope, family)
                target = "headline_complete_call" if scope == "call" else "prepared_kernel"
                result[target][family] = ({"candidate_id": row["candidate_id"], "status": row.get("status"),
                    "mean_ms": plot.mean_ms(row), "sample_count": (row.get("timing") or {}).get("sample_count"),
                    "numerical_pass": (row.get("correctness") or {}).get("pass"),
                    "relative_l2": (row.get("correctness") or {}).get("relative_l2"),
                    "max_abs_error": (row.get("correctness") or {}).get("max_abs_error"),
                    "reference_scope": (row.get("correctness") or {}).get("reference_scope"),
                    "tile_bm_bn_bk": row.get("tile"), "compiler_options": row.get("compiler_options", {}),
                    "source_ref": row["_ref"]} if row else {"status": "unavailable"})
                costs = source_covariates(shape, row)
                covariates.append({**dimensions_for(ev, shape_id), "family": family, "scope": scope,
                    "candidate_id": row.get("candidate_id") if row else None,
                    "source_ref": row.get("_ref") if row else None, **costs})
                if scope == "call":
                    cost_by_family[family] = costs
        for pair in comparisons:
            if pair["shape_id"] == shape_id:
                target = "complete_call_comparisons" if pair["scope"] == "call" else "prepared_kernel_comparisons"
                result[target][pair["reference"] + "_over_" + pair["candidate"]] = pair
        cubic, strassen = cost_by_family["cubic"], cost_by_family["strassen"]
        if all(plot.finite(c["estimated_padded_dot_flops"], True) for c in (cubic, strassen)):
            ratio = strassen["estimated_padded_dot_flops"] / cubic["estimated_padded_dot_flops"]
            padded_comparisons.append({**dimensions_for(ev, shape_id),
                "strassen_over_cubic_estimated_padded_dot_work": ratio,
                "strassen_over_cubic_padding_volume_ratio": strassen["padding_volume_ratio"] / cubic["padding_volume_ratio"],
                "strassen_has_less_estimated_dot_work": ratio < 1,
                "estimand": "Source padded dot-work ratio for independently frozen selected tiles; not a latency prediction",
                "necessary_padding_condition_for_dot_saving": "rho_strassen/rho_cubic < 8/7; insufficient to establish a speedup"})
        shape_rows.append(result)
    squares = sorted([r for r in shape_rows if r["m"] == r["n"] == r["k"]], key=lambda r: r["m"])
    near_sets, near_candidates = near_tile_sets(ev)
    failures = [{key: row.get(key) for key in ("_cohort", "_ref", "event", "group_id", "shape_id", "candidate_id", "family", "scope", "status", "during", "error_type", "error_message", "message")}
                for row in ev.statuses]
    status_counts, numeric = [], []
    for tag in ("screen", "confirmation"):
        for family in plot.FAMILIES:
            for scope in plot.SCOPES:
                rows = [r for r in ev.results if r["_cohort"] == tag and r.get("family") == family and r.get("scope") == scope]
                status_counts.append({"phase": tag, "family": family, "scope": scope,
                    "scoped_result_count": len(rows), "status_counts": dict(Counter(r.get("status") for r in rows)),
                    "counting_scope": "Scoped case_result records; confirmation includes repeated shortlist reference controls"})
                metrics = [(r, r.get("correctness") or {}) for r in rows]
                record = {"phase": tag, "family": family, "scope": scope,
                          "numerical_checks_available": sum(bool(m) for _, m in metrics),
                          "numerical_pass_count": sum(m.get("pass") is True for _, m in metrics),
                          "numerical_fail_count": sum(m.get("pass") is False for _, m in metrics)}
                for field in ("relative_l2", "max_abs_error"):
                    valid = [(r, m[field]) for r, m in metrics if plot.finite(m.get(field))]
                    maximum = max(valid, key=lambda pair: pair[1]) if valid else None
                    record["maximum_" + field] = maximum[1] if maximum else None
                    record["maximum_" + field + "_source_ref"] = maximum[0]["_ref"] if maximum else None
                numeric.append(record)
    return {"shape_rows": shape_rows, "square_rows": squares, "headline_comparisons": comparisons,
            "pointwise_counts": comparison_counts(comparisons, len(ev.exploratory)),
            "near_tile_sets": near_sets, "near_candidates": near_candidates,
            "source_covariates": covariates, "selected_padding_comparisons": padded_comparisons,
            "failure_inventory": failures, "scoped_status_counts": status_counts, "numerical_inventory": numeric}


def generate(screen, confirmation, manifest, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite descriptive output: {output}")
    ev = plot.Evidence(screen, confirmation, manifest)
    seal_check = audited_sources(ev)
    completeness = planned_completeness(ev)
    findings = describe(ev)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in findings.items():
        plot.write_csv(output / f"{name}.csv", rows)
    report = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "descriptive_analysis_of_final_sealed_exploratory_evidence",
              "shape_count": len(ev.exploratory), "reserved_unmeasured_count": len(ev.holdout),
              "off_lattice_exploratory_count": sum(not ev.shapes[s].get("on_lattice", True) for s in ev.exploratory),
              "public_shape_order": ["m", "n", "k"], "kernel_shape_order": ["m", "k", "n"],
              "confidence_scope": plot.CI_NOTE, "cautions": CAUTIONS,
              "validation": {"canonical_seals": seal_check, "planned_completeness": completeness, **ev.audit},
              "input_sources": ev.sources, "generator_sha256": plot.sha256(__file__),
              "evidence_reader_sha256": plot.sha256(PLOTTER_PATH), **findings}
    plot.write_json(output / "findings.json", report)
    plot.write_json(output / "artifact_manifest.json", {"schema_version": 1,
        "sha256": {path.name: plot.sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    return {"shape_count": len(ev.exploratory), "square_count": len(findings["square_rows"]),
            "headline_comparison_count": len(findings["headline_comparisons"]), "output_dir": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.screen, args.confirmation, args.manifest, args.output_dir)))


if __name__ == "__main__":
    main()
