#!/usr/bin/env python3
"""Audit and summarize the frozen larger-LLM-shape MM cohort, without kernels.

Reuses only generic integrity/numerical/timing helpers from audit_power_grid_v002;
its fixed-size audit entry points are deliberately not used. Selection and paired
bootstrap statistics are replayed with the cohort's frozen runner and NumPy.
This is an internal evidence-consistency audit, not independent MM validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from audit_power_grid_v002 import Checks, numerical_contract, timing_contract
from run_region_cohort_v001 import read, seal, sha, verify_frozen, write

FAMILIES = ("native", "cubic", "strassen")
SCOPES = ("call", "prepared_kernel")
PHASES = ("GRID-smoke", "GRID-screen", "GRID-confirm")


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping input: " + str(name))
    return path


def ensure_clean(checks):
    if checks.issues:
        raise ValueError(f"Evidence audit found {len(checks.issues)} inconsistencies")


def equivalent(left, right):
    """Allow only numerical roundoff, not missing fields or reordered records."""
    if type(left) in (float, int) and type(right) in (float, int):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def frozen_runner(cohort, frozen, checks):
    # Disable import caches before reading the immutable source snapshot.
    sys.dont_write_bytecode = True
    source = cohort / "source"
    for name in frozen["source_sha256"]:
        contained(source, name)
    verify_frozen(cohort)
    sys.path.insert(0, str(source / "src"))
    runner = importlib.import_module("strassen_mm.benchmark_power_grid_v001")
    for module in (runner, runner.base, runner.n5):
        expected = source / "src/strassen_mm" / (module.__name__.split(".")[-1] + ".py")
        checks.require(Path(module.__file__).resolve() == expected.resolve(), "frozen_module_origin", module.__name__)
    ensure_clean(checks)
    import numpy as np
    runner.base.np = np  # Statistical replay only; no JAX import or kernel call.
    return runner, np.__version__


def phase_artifacts(cohort, phase, frozen, configs, checks):
    marker = read(cohort / (phase + "-finished.json"))
    run = Path(marker["run"]).resolve()
    if not run.is_relative_to(cohort / "phases") or run.name != marker["run_id"]:
        raise ValueError("Phase receipt escapes cohort: " + phase)
    completion = read(run / "completion.json")
    checks.require(marker.get("completion_sha256") == sha(run / "completion.json"), "phase_receipt", phase)
    checks.require(marker.get("status") == completion.get("status") == "completed"
                   and completion.get("remote_may_still_be_running") is False,
                   "phase_stopped_and_retrieved", phase)
    checks.require(completion.get("allocation_id") == frozen["allocation_id"], "phase_allocation", phase)
    outer = read(run / "artifact-manifest.json")["sha256"]
    # Verify the consumed outer evidence, and every file in the inner benchmark
    # seal. Large duplicated remote archives need not be decoded or remeasured.
    consumed = ("completion.json", "execution.json", "source.tar", "artifacts/artifact_manifest.json")
    checks.require(set(consumed) <= outer.keys(), "outer_seal_coverage", phase)
    checks.verify(run, {k: outer[k] for k in consumed if k in outer}, phase + ":outer_consumed")
    checks.require(sha(run / "source.tar") == frozen["archive_sha256"], "same_source_archive", phase)
    execution = read(run / "execution.json")
    for key, value in {"phase": phase, "cohort_id": frozen["cohort_id"], "allocation_id": frozen["allocation_id"],
                       "source_archive_sha256": frozen["archive_sha256"], "source_commit": frozen["baseline_commit"]}.items():
        checks.require(execution.get(key) == value, "execution_provenance", phase + ":" + key)
    artifacts = run / "artifacts"
    inventory = read(artifacts / "artifact_manifest.json")["sha256"]
    required = {"environment.json", "summary.json", "results.jsonl", "source_manifest.json", "planned_cases.json"}
    if phase == "GRID-screen":
        required.add("selections.json")
    if phase == "GRID-confirm":
        required.update(("confirmation.json", "selection_input.json", "selection_input_provenance.json"))
    checks.require(required <= inventory.keys(), "benchmark_seal_coverage", phase)
    checks.verify(artifacts, inventory, phase + ":benchmark")
    expected_sources = {"source_snapshot/" + Path(name).name: digest
                        for name, digest in frozen["source_sha256"].items()
                        if Path(name).parent == Path("src/strassen_mm") and name.endswith(".py")}
    expected_sources.update({"config_snapshot/" + p.name: sha(p) for p in configs})
    actual_sources = read(artifacts / "source_manifest.json")["sha256"]
    checks.require(actual_sources == expected_sources, "frozen_source_and_config_snapshot", phase)
    checks.require(set(actual_sources) <= inventory.keys(), "snapshot_seal_coverage", phase)
    for name, digest in actual_sources.items():
        checks.require(inventory.get(name) == digest, "snapshot_digest_link", phase + ":" + name)
    ensure_clean(checks)
    return artifacts


def inspect_phase(directory, phase, campaign, manifest, runner, checks, selection=None):
    plan = runner.build_groups(campaign, manifest, phase, selection)
    checks.require(read(directory / "planned_cases.json") == plan, "plan_replayed", phase)
    experiment = campaign["experiments"][phase]
    checks.require(len(plan) == experiment["expected_group_count"], "registered_group_count", phase)
    repeats = campaign["timing"][experiment["timing"]]["repeats"]
    raw = (directory / "results.jsonl").read_text()
    checks.require(raw.endswith("\n"), "journal_complete_line", phase)
    events = [json.loads(line) for line in raw.splitlines()]
    checks.require(all(row.get("sequence") == i and row.get("phase") == phase
                       for i, row in enumerate(events, 1)), "journal_sequence", phase)
    groups = {group["group_id"]: group for group in plan}
    expected = {(g["group_id"], arm["arm_id"], scope): (g, arm)
                for g in plan for arm in g["arms"] for scope in g["scopes"]}
    if "expected_scoped_case_count" in experiment:
        checks.require(len(expected) == experiment["expected_scoped_case_count"], "registered_case_count", phase)
    cases, samples, compact, paired = defaultdict(list), defaultdict(list), [], defaultdict(list)
    for event in events:
        if event.get("event") not in ("case_result", "sample"):
            continue
        key = (event.get("group_id"), event.get("arm_id"), event.get("scope"))
        if not checks.require(key in expected, "planned_case_event", (phase, key)):
            continue
        group, arm = expected[key]
        fields = {**arm, "shape_id": group["shape"]["id"],
                  "shape_mkn": [group["shape"][d] for d in ("m", "k", "n")], **group["inputs"][0]}
        checks.require(all(event.get(k) == v for k, v in fields.items()), "event_matches_candidate_and_input", (phase, key))
        if event["event"] == "case_result":
            cases[key].append(event)
            compact.append(dict(runner.n5.compact_selection_result(event), compiler_options=event.get("compiler_options", {})))
        else:
            samples[key].append(event)
            pkey = (event["group_id"], event["shape_id"], event["candidate_id"], event["scope"])
            paired[pkey].append({k: event[k] for k in ("round", "elapsed_ms")})
    checks.require(set(cases) == set(expected), "terminal_case_coverage", phase)
    for key, records in cases.items():
        checks.require(len(records) == 1, "unique_terminal_case", (phase, key))
        for row in records:
            # The inherited runner also emits execution_error, which is not
            # enumerated in the original campaign's illustrative status list.
            allowed_statuses = set(campaign["execution_contract"]["record_statuses"]) | {"execution_error"}
            checks.require(row.get("status") in allowed_statuses, "known_case_status", (phase, key))
            checks.require(row.get("eligible_for_speedup_claim") is (row.get("status") == "ok"),
                           "recorded_eligibility", (phase, key))
            numerical_contract(row, expected[key][0]["shape"], campaign, checks, (phase, key))
            timing_contract(row, samples[key], repeats, checks, (phase, key))
    for event_name in ("group_start", "group_complete"):
        observed = [row for row in events if row.get("event") == event_name]
        checks.require([row.get("group_id") for row in observed] == list(groups), "ordered_group_coverage", phase + ":" + event_name)
        checks.require(all(row.get("index") == i and row.get("total") == len(plan)
                           for i, row in enumerate(observed, 1)), "group_progress_count", phase + ":" + event_name)
    identity_checks = [row for row in events if row.get("event") == "identity_check"]
    checks.require(len(identity_checks) == 1 and identity_checks[0].get("status") == "matched", "runtime_identity_checked", phase)
    summary = read(directory / "summary.json")
    statuses = dict(Counter(row["status"] for records in cases.values() for row in records))
    checks.require(summary.get("completed") is True and summary.get("status") == "completed"
                   and summary.get("phase") == phase and summary.get("error") is None
                   and summary.get("completed_groups") == list(groups)
                   and summary.get("planned_group_count") == len(plan)
                   and summary.get("not_completed_group_ids") == []
                   and summary.get("case_status_counts") == statuses, "completed_summary_reconciled", phase)
    final = [row for row in events if row.get("event") == "run_complete"]
    checks.require(len(final) == 1 and all(final[0].get(k) == v for k, v in summary.items())
                   and events[-1].get("event") == "run_complete", "journal_final_summary", phase)
    if phase == "GRID-smoke":
        checks.require(set(statuses) == {"ok"}, "smoke_all_pass", phase)
    failures = [{"phase": phase, "row_ref": str(directory / "results.jsonl") + ":" + str(row["sequence"]), "case": row}
                for records in cases.values() for row in records if row.get("status") != "ok"]
    return SimpleNamespace(selection_rows=compact, samples=dict(paired), cases=cases, failures=failures,
                           groups=len(plan), scoped_cases=len(expected), repeats=repeats, status_counts=statuses)


def table_rows(manifest, campaign, selection, confirmation, evidence, artifacts, runner):
    shapes = {s["id"]: s for s in manifest["shapes"]}
    models = {m["model_key"]: m for m in manifest["models"]}
    rows = []
    for sid in campaign["experiments"]["GRID-confirm"]["shape_ids"]:
        shape = shapes[sid]
        model = models[shape["model_key"]]
        group = sid + "__headline_confirmation"
        item = {"shape_id": sid, "model": model["display_name"], "model_id": model["model_id"],
                "model_key": model["model_key"], "projection": shape["projection"],
                **{d: shape[d] for d in ("m", "k", "n")}, "scope": "call", "repeats": evidence.repeats}
        for family in FAMILIES:
            winner = selection["by_shape"][sid][family]["winner"]
            cid = winner["candidate_id"] if winner else None
            eligible = runner.eligible_rows(evidence.selection_rows, sid, family, cid, evidence.repeats, group) if cid else None
            observed = [r for records in evidence.cases.values() for r in records
                        if r["group_id"] == group and r.get("candidate_id") == cid]
            item.update({family + "_candidate": cid, family + "_variant": winner["variant"] if winner else None,
                         family + "_tile": winner["tile"] if winner else None,
                         family + "_compiler_options": winner.get("compiler_options", {}) if winner else None,
                         family + "_eligible": eligible is not None,
                         family + "_status_by_scope": {s: next((r["status"] for r in observed if r["scope"] == s),
                                                                        "no_eligible_screen_winner" if not winner else "missing") for s in SCOPES},
                         family + "_observed_ms_by_scope": {r["scope"]: (r.get("timing") or {}).get("mean_ms") for r in observed},
                         family + "_row_refs": [str(artifacts / "results.jsonl") + ":" + str(r["sequence"]) for r in observed]})
            for scope in SCOPES:
                item[family + "_" + scope + "_ms"] = eligible[scope]["timing"]["mean_ms"] if eligible else None
        for baseline in ("native", "cubic"):
            contrasts = [p for p in confirmation["by_shape"][sid]["headline_comparisons"]
                         if p["scope"] == "call" and p["candidate_id"] == item["strassen_candidate"]
                         and p["reference_candidate_id"] == item[baseline + "_candidate"]]
            if len(contrasts) > 1:
                raise ValueError("Duplicate headline contrast: " + sid)
            pair = contrasts[0] if contrasts and item["strassen_eligible"] and item[baseline + "_eligible"] else None
            ci = pair["speedup_ci95"] if pair else [None, None]
            item["strassen_vs_" + baseline] = pair["speedup_ratio_of_means"] if pair else None
            item[baseline + "_ci_low"], item[baseline + "_ci_high"] = ci
            item[baseline + "_classification"] = ("unavailable" if pair is None else "win" if ci[0] > 1
                                                 else "loss" if ci[1] < 1 else "inconclusive")
        rows.append(item)
    return rows


def audit(cohort, checks):
    frozen = read(cohort / "frozen.json")
    runner, numpy_version = frozen_runner(cohort, frozen, checks)
    config_path = contained(cohort / "source", frozen["campaign_relative"])
    campaign = read(config_path)
    shape_path = contained(config_path.parent, campaign["shape_manifest"])
    distribution_path = contained(config_path.parent, campaign["distribution_manifest"])
    manifest = read(shape_path)
    checks.require(frozen.get("benchmark_module") == "strassen_mm.benchmark_power_grid_v001", "benchmark_module", frozen.get("benchmark_module"))
    checks.require(campaign.get("campaign_id") == manifest.get("plan_id") == "llm_large_shapes_v001", "campaign_identity", config_path)
    checks.require(campaign["experiments"]["GRID-screen"]["shape_ids"] == campaign["experiments"]["GRID-confirm"]["shape_ids"], "same_shape_inventory", config_path)
    checks.require(campaign["experiments"]["GRID-screen"]["seed"] != campaign["experiments"]["GRID-confirm"]["seed"], "fresh_confirmation_inputs", config_path)
    plan_provenance = campaign["plan_provenance"]
    for prefix in ("source_plan", "source_notes"):
        path = contained(cohort / "source", plan_provenance[prefix + "_path"])
        checks.require(sha(path) == plan_provenance[prefix + "_sha256"], "frozen_plan_provenance", path)
    paths, evidence, identities = {}, {}, {}
    selection = None
    for phase in PHASES:
        directory = phase_artifacts(cohort, phase, frozen, (config_path, shape_path, distribution_path), checks)
        environment = read(directory / "environment.json")
        identity = environment["identity"]
        checks.require(environment.get("qualified_single_v5e") is True
                       and identity.get("colab_endpoint") == frozen["allocation_id"]
                       and identity.get("allocation_id") == frozen["allocation_id"], "same_single_v5e_allocation", phase)
        if identities:
            checks.require(identity == identities["GRID-smoke"], "same_runtime_identity", phase)
        paths[phase], identities[phase] = directory, identity
        evidence[phase] = inspect_phase(directory, phase, campaign, manifest, runner, checks, selection)
        ensure_clean(checks)
        if phase == "GRID-screen":
            selection = read(directory / "selections.json")
            replay = runner.select_winners(evidence[phase].selection_rows, campaign, manifest)
            checks.require(all(equivalent(selection.get(k), v) for k, v in replay.items()), "screen_selection_replayed", phase)
            expected_links = {"campaign_sha256": sha(config_path), "shape_manifest_sha256": sha(shape_path),
                              "source_manifest_sha256": sha(directory / "source_manifest.json"),
                              "screen_results_sha256": sha(directory / "results.jsonl"), "environment_identity": identity}
            checks.require(all(selection.get(k) == v for k, v in expected_links.items()), "screen_selection_provenance", phase)
            ensure_clean(checks)
    confirm = paths["GRID-confirm"]
    checks.require(read(confirm / "selection_input.json") == selection, "exact_frozen_selection_input", confirm)
    provenance = read(confirm / "selection_input_provenance.json")
    checks.require(provenance.get("sha256") == sha(paths["GRID-screen"] / "selections.json"), "selection_input_hash", confirm)
    report = read(confirm / "confirmation.json")
    replay_report = runner.confirmation_report(evidence["GRID-confirm"], selection, campaign)
    checks.require(equivalent(report, replay_report), "confirmation_paired_bootstrap_replayed", confirm)
    ensure_clean(checks)
    rows = table_rows(manifest, campaign, selection, report, evidence["GRID-confirm"], confirm, runner)
    counts = {model: {baseline: dict(Counter(r[baseline + "_classification"] for r in rows if r["model"] == model))
                      for baseline in ("native", "cubic")} for model in dict.fromkeys(r["model"] for r in rows)}
    return {"schema_version": 1, "cohort_id": frozen["cohort_id"], "allocation_id": frozen["allocation_id"],
            "environment_identity": identities["GRID-smoke"], "historical_measurements_pooled": False,
            "scientific_scope": campaign["scientific_scope"], "complete_call_rows": rows,
            "classifications_by_model": counts, "failures": [f for p in PHASES for f in evidence[p].failures],
            "coverage": {p: {"groups": e.groups, "scoped_cases": e.scoped_cases, "repeats": e.repeats,
                              "case_status_counts": e.status_counts} for p, e in evidence.items()},
            "provenance": {"cohort": str(cohort), "baseline_commit": frozen["baseline_commit"],
                           "source_archive_sha256": frozen["archive_sha256"], "campaign_sha256": sha(config_path),
                           "shape_manifest_sha256": sha(shape_path), "phase_artifacts": {p: str(v) for p, v in paths.items()},
                           "selection_sha256": sha(paths["GRID-screen"] / "selections.json"),
                           "confirmation_sha256": sha(confirm / "confirmation.json"), "numpy_replay_version": numpy_version},
            "audit_scope": "Internal sealed-evidence consistency, gate/statistic recomputation, frozen plan/selection and paired-bootstrap replay. No independent matmul, new measurement, model weights, or full-model qualification."}


def write_results(output, report):
    write(output / "comparisons.json", report)
    rows = report["complete_call_rows"]
    with (output / "comparisons.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    lines = ["# Larger-LLM-shape matrix multiplication", "",
             "Gaussian BF16 operands at official model-derived dimensions; **no model weights, real activations, model quality or full-model performance are measured**.", "",
             f"Cohort `{report['cohort_id']}`; one v5e allocation `{report['allocation_id']}`. Historical cohorts are not pooled.", "",
             "Native is default XLA. Cubic and one-level Strassen independently select from six preregistered tiles per shape in screening; the selected candidates are frozen before fresh-input confirmation. No confirmation minimum is reselected.", "",
             "Complete-call means below include device padding/layout/crop and exclude compilation, host-to-device transfer and input generation. Prepared-kernel means and selected BM/BN/BK tiles are in the linked data.", "",
             "| Model | Projection | M × K × N | Native ms | Cubic ms | Strassen ms | Strassen/native speedup [95% CI] | Strassen/cubic speedup [95% CI] |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        contrasts = []
        for baseline in ("native", "cubic"):
            value = row["strassen_vs_" + baseline]
            contrasts.append("unavailable" if value is None else f"{value:.3f}× [{row[baseline + '_ci_low']:.3f}, {row[baseline + '_ci_high']:.3f}]")
        lines.append(f"| {row['model']} | {row['projection']} | {row['m']} × {row['k']} × {row['n']} | "
                     + " | ".join(fmt(row[f + "_call_ms"]) for f in FAMILIES) + " | " + " | ".join(contrasts) + " |")
    lines.extend(["", "Speedup is reference mean / Strassen mean; >1 favors Strassen. Pointwise 95% paired bootstrap intervals use the same confirmation rounds (2,000 resamples), without multiplicity correction or cross-allocation uncertainty. They do not establish generalization to other shapes or numerical distributions.", "",
                  "A value is eligible only when both timing scopes pass the frozen numerical gate and have every registered repeat. Missing/failed arms are unavailable, never zero; their recorded statuses, errors and any observed timing remain in comparisons.json. A completed phase may contain scientific failures.", "",
                  "## Coverage and failures", ""])
    for phase, item in report["coverage"].items():
        lines.append(f"- {phase}: {item['groups']} groups, {item['scoped_cases']} scoped cases; {item['repeats']} registered repeats; statuses {json.dumps(item['case_status_counts'], sort_keys=True)}.")
    for model, counts in report["classifications_by_model"].items():
        lines.append(f"- {model}, Strassen vs native: {json.dumps(counts['native'], sort_keys=True)}; vs cubic: {json.dumps(counts['cubic'], sort_keys=True)}. Win/loss requires its pointwise interval entirely above/below one.")
    lines.extend(["", f"All {len(report['failures'])} non-ok scoped outcomes are retained in comparisons.json, with journal row references.", "",
                  "The audit recomputes recorded gates/statistics and replays the frozen plan, selection and confidence intervals; it does not independently recompute any matrix multiplication. Full inner artifact seals and consumed outer provenance are verified.", "",
                  "[CSV](comparisons.csv) · [Detailed results and failures](comparisons.json) · [Evidence audit](audit.json)", ""])
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cohort, output = args.cohort.resolve(), args.output_dir.resolve()
    if output.is_relative_to(cohort):
        raise ValueError("Output must be outside the immutable input cohort")
    output.mkdir(parents=True, exist_ok=False)
    checks, report, failure = Checks(), None, None
    try:
        report = audit(cohort, checks)
        write_results(output, report)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if not (output / "RESULTS.md").exists():
            with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
                stream.write("# Evidence audit failed\n\nNo performance conclusion is qualified. See [audit.json](audit.json).\n")
    passed = failure is None and not checks.issues and report is not None
    write(output / "audit.json", {"passed": passed, "status": "passed" if passed else "failed",
                                  "cohort": str(cohort), "summary_tool_sha256": sha(Path(__file__)),
                                  "generic_audit_helper_sha256": sha(Path(__file__).with_name("audit_power_grid_v002.py")),
                                  "check_counts": dict(checks.counts), "issues": checks.issues,
                                  "verified_files": checks.verified_files, "error": failure,
                                  "scope": "Internal evidence consistency; no kernels, JAX import or new measurement."})
    seal(output)
    print(json.dumps({"status": "passed" if passed else "failed", "output_dir": str(output),
                      "shapes": len(report["complete_call_rows"]) if report else None, "issue_count": len(checks.issues)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
