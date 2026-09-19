#!/usr/bin/env python3
"""Audit explicitly selected N1--N4 evidence and create a new immutable summary.

No TPU/JAX imports, experiment launches, winner promotion, or run discovery.
Usage: python tools/summarize_campaign_v001.py --run RUN_N1 --run RUN_N2
       --run RUN_N3 --run RUN_N4 --output-dir NEW_DIRECTORY

A run argument names either a local run containing artifacts/results.jsonl or
that canonical artifacts directory itself. Source snapshots are never searched
for result files. Exactly one complete run per phase is required. An invalid
audit still writes an audit report, but emits no scientific result summaries
and exits 2. Existing output directories are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

VERSION = "summarize_campaign_v001"
EXPECTED_RESULTS = {"N1": 352, "N2": 432, "N3": 96, "N4": 480}
EXPECTED_REPEATS = {"N1": 30, "N2": 7, "N3": 30, "N4": 0}
EXPECTED_GROUPS = {"N1": 44, "N2": 72, "N3": 6, "N4": 8}
BASELINES = ("cubic_basic", "cubic_quadrant", "native_xla")
LIMITATIONS = [
    "These are current fixed-tile elementary kernels and bounded tile screens, not the strongest independently tuned cubic or Strassen baselines.",
    "N2 minima are best observed screening measurements; they are not fresh confirmations, globally optimal tiles, or a deployed selection rule.",
    "There are no held-out generalization, v6e replication, model-quality, or end-to-end LLM performance claims in N1--N4.",
    "LLM-associated shapes use synthetic BF16 matrices, not checkpoint weights or recorded activations.",
    "Complete-call timing includes device preparation/padding and output cropping; it excludes compilation and host transfers. Prepared-kernel results remain separate in the evidence.",
    "A confidence interval containing one is inconclusive at this measurement precision; it does not establish equal performance. Intervals are per-comparison, without familywise multiplicity correction.",
    "Sampled-reference errors cover a saved row-column cross-product using all K, not a certified maximum over the complete output. Whole-output finite checks are separate.",
    "Machine identity describes one logical allocation/runtime and exposed device attributes, not an independently verified permanent physical-chip serial number.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def case_key(row):
    return (row.get("group_id"), row.get("distribution"), row.get("seed"),
            row.get("arm_id"), row.get("scope"))


def status_buckets(rows):
    statuses = Counter(row.get("status", "missing_status") for row in rows)
    buckets = {"ok": 0, "numerical_failure": 0, "errors": 0, "skips": 0, "incomplete": 0}
    for status, count in statuses.items():
        category = (status if status in ("ok", "numerical_failure") else
                    "skips" if status.startswith("skipped") or status == "unsupported" else
                    "incomplete" if status in ("interrupted", "not_run") else "errors")
        buckets[category] += count
    return {"status_counts": dict(sorted(statuses.items())), "categories": buckets}


def canonical_artifacts(run_path):
    run_path = run_path.resolve()
    if (run_path / "artifacts" / "results.jsonl").is_file():
        return run_path / "artifacts"
    if (run_path / "results.jsonl").is_file():
        return run_path
    raise ValueError(f"No canonical artifacts/results.jsonl in supplied run: {run_path}")


def verify_seal(artifacts, issues):
    seal = load_json(artifacts / "artifact_manifest.json")
    hashes = seal.get("sha256")
    if not isinstance(hashes, dict):
        issues.append("artifact_manifest.json has no sha256 dictionary")
        return {}
    for required in ("results.jsonl", "summary.json", "environment.json", "planned_cases.json", "source_manifest.json"):
        if required not in hashes:
            issues.append(f"artifact seal omits required file {required}")
    for relative, expected in hashes.items():
        target = (artifacts / relative).resolve()
        if not target.is_relative_to(artifacts.resolve()):
            issues.append(f"artifact seal path escapes its directory: {relative}")
        elif not target.is_file():
            issues.append(f"sealed artifact missing: {relative}")
        elif sha256(target) != expected:
            issues.append(f"sealed artifact hash mismatch: {relative}")
    return {"artifact_manifest_sha256": sha256(artifacts / "artifact_manifest.json"),
            "sealed_file_count": len(hashes), "sealed_utc": seal.get("sealed_utc")}


def audit_run(run_path):
    issues = []
    artifacts = canonical_artifacts(run_path)
    seal = verify_seal(artifacts, issues)
    summary = load_json(artifacts / "summary.json")
    environment = load_json(artifacts / "environment.json")
    planned = load_json(artifacts / "planned_cases.json")
    source = load_json(artifacts / "source_manifest.json")
    phase = summary.get("phase")
    if phase not in EXPECTED_RESULTS:
        raise ValueError(f"Expected an N1--N4 run, got phase {phase!r} at {artifacts}")
    rows, samples, event_counts = [], defaultdict(list), Counter()
    completed_events = []
    with (artifacts / "results.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                issues.append(f"blank JSONL line {line_number}")
                continue
            row = json.loads(line)
            event_counts[row.get("event", "unknown")] += 1
            if row.get("sequence") != line_number:
                issues.append(f"nonconsecutive journal sequence at line {line_number}")
            if row.get("phase") != phase:
                issues.append(f"wrong phase in journal line {line_number}")
            if row.get("event") == "case_result":
                rows.append(row)
            elif row.get("event") == "sample":
                samples[case_key(row)].append(row)
            elif row.get("event") == "run_complete":
                completed_events.append(row)
    if summary.get("completed") is not True or summary.get("status") != "completed":
        issues.append("phase did not complete")
    if len(completed_events) != 1 or completed_events[0].get("completed") is not True:
        issues.append("journal does not have exactly one completed run_complete event")
    if len(rows) != EXPECTED_RESULTS[phase]:
        issues.append(f"case_result count {len(rows)} != expected {EXPECTED_RESULTS[phase]}")
    if len(planned) != EXPECTED_GROUPS[phase] or summary.get("planned_group_count") != len(planned):
        issues.append(f"planned group count does not match expected {EXPECTED_GROUPS[phase]}")
    expected = set()
    groups = {}
    for group in planned:
        if group["group_id"] in groups:
            issues.append(f"duplicate planned group {group['group_id']}")
        groups[group["group_id"]] = group
        for inputs in group["inputs"]:
            for arm in group["arms"]:
                for scope in group["scopes"]:
                    expected.add((group["group_id"], inputs["distribution"], inputs["seed"], arm["arm_id"], scope))
    observed = [case_key(row) for row in rows]
    if len(observed) != len(set(observed)):
        issues.append("duplicate case_result key")
    if set(observed) != expected:
        issues.append(f"planned/result mismatch: missing={len(expected-set(observed))}, unexpected={len(set(observed)-expected)}")
    if set(summary.get("completed_groups", [])) != set(groups) or summary.get("not_completed_group_ids"):
        issues.append("summary does not list every planned group as completed")
    if dict(Counter(row.get("status") for row in rows)) != summary.get("case_status_counts"):
        issues.append("summary case_status_counts differ from raw results")
    identity = environment.get("identity")
    if not isinstance(identity, dict) or not environment.get("qualified_single_v5e"):
        issues.append("missing identity or machine was not qualified as single v5e")
    if not isinstance(identity, dict):
        identity = {}
    required_identity = ("allocation_id", "hostname", "device_kind", "jax_version", "jaxlib_version", "libtpu_version")
    if any(not identity.get(field) for field in required_identity):
        issues.append("required allocation/device/runtime identity is unavailable")
    for row in rows:
        key = case_key(row)
        raw = samples.get(key, [])
        if len(raw) != (row.get("timing") or {}).get("sample_count", 0):
            issues.append(f"raw/summary sample-count mismatch for {key}")
        if row.get("status") in ("ok", "numerical_failure") and len(raw) != EXPECTED_REPEATS[phase]:
            issues.append(f"completed case lacks expected repeat count for {key}")
        rounds = [item.get("round") for item in raw]
        if len(rounds) != len(set(rounds)):
            issues.append(f"duplicate timing round for {key}")
        if any(not finite_number(item.get("elapsed_ms")) or item["elapsed_ms"] <= 0 for item in raw):
            issues.append(f"invalid raw timing for {key}")
        if raw and finite_number((row.get("timing") or {}).get("mean_ms")):
            mean = statistics.mean(item["elapsed_ms"] for item in raw)
            if not math.isclose(mean, row["timing"]["mean_ms"], rel_tol=1e-9, abs_tol=1e-12):
                issues.append(f"raw/summary timing mean mismatch for {key}")
        metrics = row.get("correctness") or {}
        if row.get("status") == "ok" and (metrics.get("pass") is not True or metrics.get("finite") is not True):
            issues.append(f"ok status without passing finite numerical gate for {key}")
        metadata = row.get("kernel_metadata")
        if metadata and (metadata.get("input_dtype") != "bfloat16" or metadata.get("output_dtype") != "float32"
                         or metadata.get("accumulation_dtype") != "float32" or metadata.get("dot_precision") != "DEFAULT"):
            issues.append(f"arithmetic contract mismatch for {key}")
    if set(samples) - set(observed):
        issues.append("raw samples have no corresponding case_result")
    config_hashes = {key: value for key, value in source.get("sha256", {}).items()
                     if key.startswith("config_snapshot/")}
    if len(config_hashes) != 3:
        issues.append("source manifest must identify campaign, shapes and distribution configurations")
    return {"phase": phase, "run_path": str(run_path.resolve()), "artifacts": str(artifacts),
            "issues": issues, "summary": summary, "environment": environment, "identity": identity,
            "source_manifest": source, "configuration_hashes": config_hashes, "seal": seal,
            "event_counts": dict(event_counts), "rows": rows, "groups": groups,
            "sample_rows": samples}


def eligible(row):
    metrics = row.get("correctness") or {}
    return (row.get("status") == "ok" and row.get("eligible_for_speedup_claim") is True
            and metrics.get("pass") is True and metrics.get("finite") is True)


def contrast(candidate, baseline, reference_arm):
    pair = next((item for item in candidate.get("comparisons", [])
                 if item.get("reference_arm") == reference_arm), None)
    result = {"candidate_status": candidate.get("status"), "reference_status": baseline.get("status") if baseline else None,
              "candidate_ms": (candidate.get("timing") or {}).get("mean_ms"),
              "reference_ms": (baseline.get("timing") or {}).get("mean_ms") if baseline else None,
              "classification": "ineligible", "paired_comparison": pair}
    if not baseline or not eligible(candidate) or not eligible(baseline):
        return result
    if not pair or pair.get("valid_numerical_comparison") is not True:
        result["classification"] = "missing_valid_comparison"
        return result
    interval = pair.get("speedup_ci95", [])
    if len(interval) != 2 or not all(finite_number(value) and value > 0 for value in interval) or interval[0] > interval[1]:
        result["classification"] = "invalid_interval"
        return result
    result["classification"] = "win" if interval[0] > 1 else "loss" if interval[1] < 1 else "tie_inconclusive"
    return result


def describe_group(run, group_id):
    group = run["groups"][group_id]
    shape = group["shape"]
    return {"shape_id": shape["id"], "shape_mkn": [shape[key] for key in ("m", "k", "n")],
            "family": shape["family"], "tile_bm_bn_bk": group["tile"]}


def summarize_n1(run):
    rows = [row for row in run["rows"] if row.get("scope") == "call"]
    index = {(row["group_id"], row["arm_id"]): row for row in rows}
    comparisons, totals, family_totals = [], defaultdict(Counter), defaultdict(Counter)
    for group_id in run["groups"]:
        candidate = index[(group_id, "strassen_basic")]
        for baseline in BASELINES:
            item = {**describe_group(run, group_id), "reference_arm": baseline,
                    **contrast(candidate, index.get((group_id, baseline)), baseline)}
            comparisons.append(item)
            totals[baseline][item["classification"]] += 1
            family_totals[(item["family"], baseline)][item["classification"]] += 1
    return {"scope": "call", "interpretation": "Untuned fixed-tile comparisons over all 44 preregistered shapes.",
            "counts_by_reference": {key: dict(value) for key, value in totals.items()},
            "counts_by_family_reference": [{"family": family, "reference_arm": baseline, "counts": dict(counts)}
                                            for (family, baseline), counts in sorted(family_totals.items())],
            "comparisons": comparisons, **status_buckets(run["rows"])}


def summarize_n2(run):
    buckets = defaultdict(list)
    for row in run["rows"]:
        if row.get("scope") == "call":
            info = describe_group(run, row["group_id"])
            buckets[(info["shape_id"], row["arm_id"])].append((row, info))
    screens = []
    for (shape_id, arm), values in sorted(buckets.items()):
        candidates = [{**info, "status": row["status"], "eligible": eligible(row),
                       "mean_ms": (row.get("timing") or {}).get("mean_ms"),
                       "numerical_metrics": row.get("correctness")} for row, info in values]
        valid = [item for item in candidates if item["eligible"] and finite_number(item["mean_ms"]) and item["mean_ms"] > 0]
        best = min(valid, key=lambda item: (item["mean_ms"], item["tile_bm_bn_bk"])) if valid else None
        screens.append({"shape_id": shape_id, "arm_id": arm, "attempted_tiles": len(candidates),
                        "eligible_tiles": len(valid), "status_counts": dict(Counter(item["status"] for item in candidates)),
                        "best_observed_screen": best, "candidates": candidates})
    return {"scope": "call", "confirmation": False,
            "interpretation": "Best observed tile per arm and shape in the bounded screen, selected by complete-call mean only. No promoted winner or new experiment is generated.",
            "screens": screens, **status_buckets(run["rows"])}


def summarize_n3(run):
    rows = [row for row in run["rows"] if row.get("scope") == "call"]
    index = {(row["group_id"], row["arm_id"]): row for row in rows}
    comparisons, counts = [], defaultdict(Counter)
    for row in rows:
        if row.get("variant") == "plain":
            continue
        group = run["groups"][row["group_id"]]
        arm = next(arm for arm in group["arms"] if arm["arm_id"] == row["arm_id"])
        reference = f"{arm['algorithm']}__plain"
        item = {**describe_group(run, row["group_id"]), "algorithm": arm["algorithm"], "variant": arm["variant"],
                **contrast(row, index.get((row["group_id"], reference)), reference)}
        comparisons.append(item)
        counts[(arm["algorithm"], arm["variant"])][item["classification"]] += 1
    return {"scope": "call", "interpretation": "Fixed-tile single-change and combined ablations versus the same algorithm's plain variant; no claim that individual gains add linearly.",
            "counts_by_algorithm_variant": [{"algorithm": algorithm, "variant": variant, "counts": dict(value)}
                                            for (algorithm, variant), value in sorted(counts.items())],
            "comparisons": comparisons, **status_buckets(run["rows"])}


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def metric_distribution(rows, key):
    values = [(row.get("correctness") or {}).get(key) for row in rows]
    values = [value for value in values if finite_number(value)]
    return ({"finite_observations": 0} if not values else
            {"finite_observations": len(values), "min": min(values), "median": statistics.median(values),
             "p95": quantile(values, .95), "max": max(values)})


def summarize_n4(run):
    buckets = defaultdict(list)
    cases = []
    for row in run["rows"]:
        buckets[(row["arm_id"], row["distribution"])].append(row)
        cases.append({**describe_group(run, row["group_id"]), "arm_id": row["arm_id"],
                      "distribution": row["distribution"], "seed": row["seed"],
                      "status": row["status"], "correctness": row.get("correctness"),
                      "error_message": row.get("error_message")})
    distributions = []
    for (arm, profile), rows in sorted(buckets.items()):
        metrics = [row.get("correctness") or {} for row in rows]
        distributions.append({"arm_id": arm, "distribution": profile, "case_count": len(rows),
                              **status_buckets(rows),
                              "reference_scope_counts": dict(Counter(item.get("reference_scope", "unavailable") for item in metrics)),
                              "nonfinite_output_or_reference_count": sum(item.get("finite") is False for item in metrics),
                              "relative_l2": metric_distribution(rows, "relative_l2"),
                              "max_abs_error": metric_distribution(rows, "max_abs_error"),
                              "normwise_error": metric_distribution(rows, "normwise_error")})
    return {"interpretation": "Synthetic numerical characterization, including retained failures and near cancellation. Reported maxima are maxima of the saved reference-check scopes, not global certification.",
            "timing_claims": False, "distributions": distributions, "cases": cases,
            **status_buckets(run["rows"])}


def count(counts, key):
    return counts.get(key, 0)


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_number(value):
    return "—" if not finite_number(value) else f"{value:.4g}"


def render_markdown(report):
    audit = report["audit"]
    lines = ["# N1–N4 v5e experiment summary", "", f"Evidence audit: **{'PASS' if audit['passed'] else 'FAIL'}**.", ""]
    if not audit["passed"]:
        lines += ["No scientific summaries were produced because the required evidence contract was not satisfied.", ""]
        lines += [f"- {escape(item)}" for item in audit["issues"]]
        return "\n".join(lines) + "\n"
    identity = audit["identity"]
    lines += [f"One logical allocation: `{escape(identity['allocation_id'])}`; device: `{escape(identity['device_kind'])}`.",
              "", "Case counts include both timing scopes in N1–N3. N4 contains correctness cases only.", "",
              "| Phase | Cases | Valid | Numerical failures | Errors | Skips | Incomplete |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for phase in EXPECTED_RESULTS:
        section = report["experiments"][phase]
        values = section["categories"]
        lines.append(f"| {phase} | {EXPECTED_RESULTS[phase]} | {values['ok']} | {values['numerical_failure']} | {values['errors']} | {values['skips']} | {values['incomplete']} |")
    lines += ["", "## N1: fixed-tile complete-call comparisons", "",
              "A win requires the paired speedup CI95 to lie above 1 and both numerical gates to pass. A loss lies below 1; a tie is inconclusive.", "",
              "| Strassen versus | Wins | Ties/inconclusive | Losses | Ineligible/other |", "|---|---:|---:|---:|---:|"]
    for baseline, counts in report["experiments"]["N1"]["counts_by_reference"].items():
        other = sum(value for key, value in counts.items() if key not in ("win", "loss", "tie_inconclusive"))
        lines.append(f"| {baseline} | {count(counts,'win')} | {count(counts,'tie_inconclusive')} | {count(counts,'loss')} | {other} |")
    lines += ["", "## N2: bounded tile screening", "",
              "Best observed complete-call mean for each arm and shape. These selected minima are **screening observations, not confirmed speedups**.", "",
              "| Shape | Algorithm arm | Eligible / attempted tiles | Best observed tile (BM,BN,BK) | Mean ms |",
              "|---|---|---:|---|---:|"]
    for screen in report["experiments"]["N2"]["screens"]:
        best = screen["best_observed_screen"] or {}
        tile = ",".join(map(str, best["tile_bm_bn_bk"])) if best else "—"
        lines.append(f"| {screen['shape_id']} | {screen['arm_id']} | {screen['eligible_tiles']} / {screen['attempted_tiles']} | {tile} | {format_number(best.get('mean_ms'))} |")
    lines += ["", "## N3: fixed-tile ablations versus plain", "",
              "| Algorithm | Variant | Wins | Ties/inconclusive | Losses | Ineligible/other |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["experiments"]["N3"]["counts_by_algorithm_variant"]:
        counts = row["counts"]
        other = sum(value for key, value in counts.items() if key not in ("win", "loss", "tie_inconclusive"))
        lines.append(f"| {row['algorithm']} | {row['variant']} | {count(counts,'win')} | {count(counts,'tie_inconclusive')} | {count(counts,'loss')} | {other} |")
    lines += ["", "## N4: numerical stress cases", "",
              "Every arm/profile contains eight shapes and three seeds. Failures remain in the table; near-cancellation cases use the same frozen gate.", "",
              "| Arm | Profile | Passed | Numerical failures | Errors/skips | Median relative L2 | Maximum checked relative L2 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["experiments"]["N4"]["distributions"]:
        categories, relative = row["categories"], row["relative_l2"]
        other = categories["errors"] + categories["skips"] + categories["incomplete"]
        lines.append(f"| {row['arm_id']} | {row['distribution']} | {categories['ok']} | {categories['numerical_failure']} | {other} | {format_number(relative.get('median'))} | {format_number(relative.get('max'))} |")
    lines += ["", "## Scope and limits", ""]
    lines += [f"- {text}" for text in LIMITATIONS]
    lines += ["", "Exact selected runs, configuration hashes, per-shape comparisons, all screen candidates, and numerical cases are in `summary.json`. Source measurements remain in the supplied immutable run artifacts.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True, help="Explicit run or canonical artifacts directory; repeat exactly once per N1--N4 phase.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected.")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs, issues = {}, []
    seen_paths = set()
    for path in args.run:
        try:
            canonical = canonical_artifacts(path)
            if canonical in seen_paths:
                issues.append(f"Duplicate supplied artifact directory: {canonical}")
                continue
            seen_paths.add(canonical)
            run = audit_run(path)
            phase = run["phase"]
            issues.extend(f"{phase}: {message}" for message in run["issues"])
            if phase in runs:
                issues.append(f"Multiple supplied runs for {phase}; select one explicitly, no automatic choice is made.")
            else:
                runs[phase] = run
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(f"Cannot audit {path}: {type(error).__name__}: {error}")
    if set(runs) != set(EXPECTED_RESULTS):
        issues.append(f"Exactly N1--N4 required; received {sorted(runs)}")
    identity = runs.get("N1", {}).get("identity", {})
    config_hashes = runs.get("N1", {}).get("configuration_hashes", {})
    for phase, run in runs.items():
        if run["identity"] != identity:
            differing = sorted(key for key in set(identity) | set(run["identity"])
                               if identity.get(key) != run["identity"].get(key))
            issues.append(f"{phase}: identity differs from N1 in {differing}")
        if run["configuration_hashes"] != config_hashes:
            issues.append(f"{phase}: frozen configuration hashes differ from N1")
    report = {"schema_version": 1, "generator": VERSION, "created_utc": now(),
              "audit": {"passed": not issues, "issues": issues, "identity": identity,
                        "configuration_hashes": config_hashes, "expected_case_results": EXPECTED_RESULTS,
                        "runs": [{"phase": phase, "run_path": run["run_path"], "artifacts": run["artifacts"],
                                  "result_count": len(run["rows"]), "event_counts": run["event_counts"],
                                  "seal": run["seal"], "source_manifest": run["source_manifest"],
                                  **status_buckets(run["rows"])} for phase, run in sorted(runs.items())]},
              "limitations": LIMITATIONS, "experiments": {}}
    if not issues:
        report["experiments"] = {"N1": summarize_n1(runs["N1"]), "N2": summarize_n2(runs["N2"]),
                                 "N3": summarize_n3(runs["N3"]), "N4": summarize_n4(runs["N4"])}
    write_json(output / "summary.json", report)
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    write_json(output / "manifest.json", {"generator": VERSION, "generator_sha256": sha256(Path(__file__).resolve()),
                                         "created_utc": now(), "sha256": {name: sha256(output / name) for name in ("summary.json", "summary.md")}})
    print(json.dumps({"audit_passed": not issues, "issues": issues, "output_dir": str(output)}, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
