"""Summarize one completed region cohort, separating sampling roles and machines."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from run_region_cohort_v001 import CAMPAIGN, read, seal, sha, verify_frozen, write


def verified_artifacts(cohort, phase):
    record = read(cohort / (phase + "-finished.json"))
    if record["status"] != "completed":
        raise ValueError("Incomplete phase: " + phase)
    run = Path(record["run"])
    if not run.resolve().is_relative_to((cohort / "phases").resolve()):
        raise ValueError("Phase points outside this cohort")
    artifacts = run / "artifacts"
    expected = read(artifacts / "artifact_manifest.json")["sha256"]
    required = {"environment.json", "results.jsonl", "summary.json", "source_manifest.json", "planned_cases.json"}
    required |= ({"selections.json"} if phase == "GRID-screen" else {"confirmation.json", "selection_input.json"})
    if not required <= set(expected):
        raise ValueError("Consumed artifacts missing from seal: " + str(sorted(required - set(expected))))
    for name, digest in expected.items():
        if not (artifacts / name).resolve().is_relative_to(artifacts.resolve()):
            raise ValueError("Invalid artifact path")
        if sha(artifacts / name) != digest:
            raise ValueError("Changed artifact: " + name)
    if not read(artifacts / "summary.json").get("completed"):
        raise ValueError("Incomplete measurement journal")
    return artifacts


def describe(cohort, screen, confirmation):
    meta = read(cohort / "cohort.json")
    envs = [read(p / "environment.json") for p in (screen, confirmation)]
    if envs[0]["identity"] != envs[1]["identity"] or envs[0]["identity"]["allocation_id"] != meta["allocation_id"]:
        raise ValueError("Mixed runtime identities; refusing to combine measurements")
    config = read(cohort / "source" / CAMPAIGN)
    manifest_path = cohort / "source" / Path(CAMPAIGN).parent / config["shape_manifest"]
    manifest = read(manifest_path)
    if config["cohort_id"] != meta["cohort_id"] or manifest["cohort_id"] != meta["cohort_id"]:
        raise ValueError("Cohort ID mismatch")
    selected = read(screen / "selections.json")
    if (selected["campaign_sha256"] != sha(cohort / "source" / CAMPAIGN)
            or selected["shape_manifest_sha256"] != sha(manifest_path)
            or selected["screen_results_sha256"] != sha(screen / "results.jsonl")
            or selected["source_manifest_sha256"] != sha(screen / "source_manifest.json")
            or selected["environment_identity"] != envs[0]["identity"]):
        raise ValueError("Screen selection differs from this cohort's frozen source or evidence")
    if read(confirmation / "selection_input.json") != selected:
        raise ValueError("Confirmation did not use this exact frozen screen selection")
    for directory in (screen, confirmation):
        snapshots = read(directory / "source_manifest.json")["sha256"]
        for path in (cohort / "source" / CAMPAIGN, manifest_path):
            if snapshots.get("config_snapshot/" + path.name) != sha(path):
                raise ValueError("Artifact configuration differs from frozen cohort")
    report = read(confirmation / "confirmation.json")
    shapes = {s["id"]: s for s in manifest["shapes"]}
    if set(report["by_shape"]) != set(manifest["exploratory_shape_ids"]):
        raise ValueError("Confirmed shape inventory differs from frozen exploratory sample")
    headline = {}
    for line in (confirmation / "results.jsonl").open():
        row = json.loads(line)
        if row.get("event") == "case_result" and row.get("scope") == "call" and row.get("group_id", "").endswith("__headline_confirmation"):
            key = row["shape_id"], row["candidate_id"]
            if key in headline:
                raise ValueError("Duplicate headline candidate")
            headline[key] = row
    rows = []
    for sid in manifest["exploratory_shape_ids"]:
        shape, selection = shapes[sid], selected["by_shape"][sid]
        item = {"cohort_id": meta["cohort_id"], "allocation_id": meta["allocation_id"],
                "shape_id": sid, "sampling_role": shape["sampling_role"],
                **{d: shape[d] for d in ("m", "n", "k")}}
        for family in ("native", "cubic", "strassen"):
            winner = selection[family]["winner"]
            cid = winner["candidate_id"] if winner else None
            observed = headline.get((sid, cid), {})
            eligible = observed.get("status") == "ok" and observed.get("correctness", {}).get("pass") is True
            item[family + "_candidate"] = cid
            item[family + "_ms"] = observed["timing"]["mean_ms"] if eligible else None
        default = headline.get((sid, "native_default"), {})
        item["native_default_candidate"] = "native_default"
        item["native_default_ms"] = (default["timing"]["mean_ms"] if default.get("status") == "ok"
            and default.get("correctness", {}).get("pass") is True else None)
        for baseline in ("native", "cubic", "native_default"):
            candidate, reference = item["strassen_candidate"], item[baseline + "_candidate"]
            matches = [p for p in report["by_shape"][sid]["headline_comparisons"]
                       if p["scope"] == "call" and p["candidate_id"] == candidate and p["reference_candidate_id"] == reference]
            if len(matches) > 1:
                raise ValueError("Duplicate canonical contrast")
            pair = matches[0] if matches else None
            item["strassen_vs_" + baseline] = pair["speedup_ratio_of_means"] if pair else None
            ci = pair["speedup_ci95"] if pair else [None, None]
            item[baseline + "_ci_low"], item[baseline + "_ci_high"] = ci
            item[baseline + "_classification"] = ("unavailable" if pair is None else
                "win" if ci[0] > 1 else "loss" if ci[1] < 1 else "inconclusive")
        rows.append(item)
    counts = {role: dict(Counter(r["native_classification"] for r in rows if r["sampling_role"] == role))
              for role in sorted({r["sampling_role"] for r in rows})}
    broad = [r for r in rows if r["sampling_role"] == "broad"]
    available = [r for r in broad if r["strassen_vs_native"] is not None]
    broad_sample = {"planned": len(broad), "available": len(available), "unavailable": len(broad)-len(available),
                    "point_estimate_wins": sum(r["strassen_vs_native"] > 1 for r in available),
                    "pointwise_ci_qualified_wins": sum(r["native_classification"] == "win" for r in available),
                    "eligible_population_size": manifest["broad_sampling_design"]["eligible_population_size"],
                    "scope": "Broad sample only; timing uncertainty and finite-sample uncertainty are distinct. Missing outcomes remain in the planned denominator; no population confidence interval fitted here."}
    return {"cohort_id": meta["cohort_id"], "allocation_id": meta["allocation_id"],
            "historical_measurements_pooled": False, "environment_identity": envs[0]["identity"],
            "complete_call_rows": rows, "classifications_by_sampling_role": counts,
            "broad_sample": broad_sample,
            "interpretation": "Pointwise paired confirmation intervals for screen-frozen family winners. Broad sample, focused probes, and anchors remain separate. These intervals do not measure allocation variability.",
            "holdouts_measured": [], "manifest_path": str(manifest_path), "config_path": str(cohort / "source" / CAMPAIGN)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    verify_frozen(cohort)
    screen = verified_artifacts(cohort, "GRID-screen")
    confirmation = verified_artifacts(cohort, "GRID-confirm")
    report = describe(cohort, screen, confirmation)
    output = cohort / "analysis"
    output.mkdir(exist_ok=False)
    write(output / "confirmed_results.json", report)
    rows = report["complete_call_rows"]
    with (output / "confirmed_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "RESULTS.md").open("x") as stream:
        stream.write("# Separate region-grid cohort\n\n")
        stream.write(f"Cohort: `{report['cohort_id']}`. Allocation: `{report['allocation_id']}`.\n\n")
        stream.write("Historical measurements were not pooled. All old and new reserved holdouts remain unmeasured.\n\n")
        stream.write("| Sampling group | Wins vs tuned native | Inconclusive | Losses | Unavailable |\n|---|---:|---:|---:|---:|\n")
        for role, counts in report["classifications_by_sampling_role"].items():
            stream.write(f"| {role} | {counts.get('win',0)} | {counts.get('inconclusive',0)} | {counts.get('loss',0)} | {counts.get('unavailable',0)} |\n")
        stream.write("\nCounts use individual paired 95% confidence intervals, without multiplicity correction. Focused and anchor observations are not a uniform-grid prevalence estimate.\n\n")
        stream.write("[Full shape table](confirmed_results.csv) · [Machine-readable results](confirmed_results.json)\n")
    model = [sys.executable, str(cohort / "source/tools/fit_region_margin_v001.py"),
             "--manifest", report["manifest_path"], "--config", report["config_path"],
             "--screen", str(screen / "results.jsonl"), "--environment", str(screen / "environment.json"),
             "--cohort-id", report["cohort_id"], "--allocation-id", report["allocation_id"],
             "--confirm", str(confirmation / "results.jsonl"),
             "--confirmation-environment", str(confirmation / "environment.json"), "--out-dir", str(output / "margin_model")]
    write(output / "model-command.json", {"argv": model})
    with (output / "model.log").open("x") as log:
        result = subprocess.run(model, env=dict(os.environ, PYTHONPATH=str(cohort / "source/src"), PYTHONDONTWRITEBYTECODE="1"),
                                stdout=log, stderr=subprocess.STDOUT)
    write(output / "model-status.json", {"exit_code": result.returncode, "status": "completed" if result.returncode == 0 else "needs_attention"})
    seal(output)
    print(json.dumps({"output": str(output), "shape_count": len(rows), "model_exit_code": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
