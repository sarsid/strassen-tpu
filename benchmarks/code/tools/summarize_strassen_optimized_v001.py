#!/usr/bin/env python3
"""Summarize sealed TPU ablations; no CPU timings, pooling, or winner claims.

Only the standard library is required. Canonical runner bootstrap intervals are
transformed, not re-estimated. Paired raw round membership and point estimates
are independently checked before those recorded intervals are presented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


VERSION = "summarize_strassen_optimized_v001"
REFERENCES = ("old_strassen", "native_default", "native_vmem_64m")
SCOPES = ("call", "prepared_kernel")
LABELS = {
    "old_strassen": "Frozen Strassen", "native_default": "Native default",
    "native_vmem_64m": "Native 64 MiB", "new_baseline": "New baseline",
    "new_masked_edges": "Masked edges", "new_local_accumulators": "Local accumulators",
    "new_optimized": "Optimized", "new_optimized_nmk": "Optimized N,M,K",
    "new_peeled_edges": "Peeled edges (hybrid)",
}
TABLE_ORDER = tuple(LABELS)
INTERPRETATION = (
    "Fixed matched-tile Gaussian-input ablation. Positive time reduction means "
    "less elapsed time. Intervals are pointwise 95% paired-bootstrap intervals "
    "recorded by the runner, not multiplicity-adjusted or population estimates. "
    "Screen and confirmation samples, timing scopes and allocations are never "
    "pooled. Minimum observed means are descriptive, including in confirmation; "
    "all confirmation arms were frozen before that run. No CPU timing evidence."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def time_reduction_percent(speedup, interval):
    """Convert reference/candidate speedup and endpoints without reversing them.

    f(s)=100*(1-1/s) is increasing on s>0. This is relative time reduction,
    not 100*(speedup-1), and negative values indicate extra elapsed time.
    """
    require(positive(speedup), "Speedup must be finite and positive")
    require(isinstance(interval, (list, tuple)) and len(interval) == 2,
            "Speedup interval must contain two endpoints")
    lo, hi = interval
    require(positive(lo) and positive(hi) and lo <= hi,
            "Speedup interval must be finite, positive and ordered")
    return 100 * (1 - 1 / speedup), [100 * (1 - 1 / lo), 100 * (1 - 1 / hi)]


def eligible(row):
    """Never turn a partial/failed/CPU result into a speed comparison."""
    return (row.get("status") == "ok" and row.get("interpretation_only") is not True
            and row.get("numerically_eligible") is True
            and row.get("eligible_for_speedup_claim") is True
            and (row.get("correctness") or {}).get("pass") is True)


class Evidence:
    def __init__(self, directory, expected_phase=None):
        directory = Path(directory).resolve()
        if (directory / "artifacts/plan.json").is_file():
            directory /= "artifacts"
        self.directory = directory
        seal_path = directory / "artifact_manifest.json"
        seal = read_json(seal_path)
        self.hashes = seal["sha256"]
        required = {"plan.json", "summary.json", "results.jsonl", "environment.json",
                    "source_manifest.json", "planned_cases.json"}
        require(required <= self.hashes.keys(), "Seal omits required evidence")
        for name, digest in self.hashes.items():
            path = (directory / name).resolve()
            require(path.is_relative_to(directory) and path.is_file(),
                    f"Invalid sealed artifact path: {name}")
            require(sha256(path) == digest, f"Artifact hash mismatch: {name}")
        self.plan = read_json(directory / "plan.json")
        self.summary = read_json(directory / "summary.json")
        self.environment = read_json(directory / "environment.json")
        self.source = read_json(directory / "source_manifest.json")
        require(bool(self.source.get("sha256")) and all(self.hashes.get(name) == digest
                for name, digest in self.source["sha256"].items()), "Source manifest is not bound to sealed files")
        self.phase = self.summary["phase"]
        require(self.phase in ("screen", "confirm"), "Unknown experiment phase")
        require(expected_phase is None or self.phase == expected_phase, "Wrong phase supplied")
        require(self.plan.get("interpret_correctness") is False
                and self.summary.get("interpretation_only") is False
                and self.environment.get("interpretation_only") is False
                and self.summary.get("plan_only") is False,
                "CPU correctness or plan-only runs cannot provide timing evidence")
        self.identity = self.environment["identity"]
        devices = self.environment.get("devices", [])
        require(self.environment.get("backend") == "tpu" and len(devices) == 1
                and devices[0].get("platform") == "tpu"
                and all(self.environment.get(k) == 1 for k in
                        ("device_count", "local_device_count", "process_count"))
                and bool(self.identity.get("allocation_id")),
                "Timing evidence must identify one TPU and one allocation")
        require(self.identity.get("devices") == devices, "Environment device identities disagree")
        require(self.plan.get("shape_order") == "M,N,K"
                and self.plan.get("tile_order") == "BM,BN,BK"
                and self.plan.get("scopes") == list(SCOPES), "Unknown geometry/scope convention")
        self.repeats = self.summary["repeats"]
        self.seed = self.summary["seed"]
        require(type(self.repeats) is int and self.repeats in (7, 30), "Unexpected timing count")
        require(type(self.seed) is int and self.seed >= 0, "Invalid input seed")
        require(self.phase != "confirm" or self.repeats == 30, "Confirmation needs 30 rounds")
        self.shapes = {s["id"]: s for s in self.plan["shapes"]}
        self.candidates = {c["candidate_id"]: c for c in self.plan["candidates"]}
        require(len(self.shapes) == len(self.plan["shapes"])
                and len(self.candidates) == len(self.plan["candidates"]), "Duplicate planned identifiers")
        require(set(REFERENCES) <= self.candidates.keys(), "Missing frozen reference candidates")
        require(self.summary["planned_shapes"] == len(self.shapes)
                and self.summary["planned_candidates"] == len(self.candidates), "Plan count mismatch")
        self.events = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
        require([e["sequence"] for e in self.events] == list(range(1, len(self.events) + 1)),
                "Journal sequence is incomplete or duplicated")
        require(all(e["phase"] == self.phase for e in self.events), "Journal mixes experiment phases")
        starts = [e for e in self.events if e["event"] == "run_start"]
        ends = [e for e in self.events if e["event"] == "run_complete"]
        require(len(starts) == len(ends) == 1, "Journal must contain one start and one sealed end")
        require(all(ends[0].get(k) == v for k, v in self.summary.items()), "Summary/journal mismatch")
        require(starts[0]["seed"] == self.seed and starts[0]["repeats"] == self.repeats
                and starts[0]["interpretation_only"] is False
                and starts[0]["plan_sha256"] == self.hashes["plan.json"], "Run-start binding mismatch")
        planned = read_json(directory / "planned_cases.json")
        require(starts[0]["execution_plan"] == planned, "Execution plan/journal mismatch")
        require(len(planned) == len(self.shapes), "Execution plan shape count mismatch")
        for group in planned:
            sid = group["shape"]["id"]
            require(sid in self.shapes and group["shape"] == self.shapes[sid]
                    and group["group_id"] == sid + "__fixed_ablation"
                    and group["timing"]["repeats"] == self.repeats
                    and group["input"] == {"distribution": "gaussian", "seed": self.seed},
                    "Execution group disagrees with frozen plan")
        self.rows, self.raw = {}, defaultdict(dict)
        for event in self.events:
            if event["event"] not in ("case_result", "sample"):
                continue
            key = self.validate_context(event)
            if event["event"] == "sample":
                r = event["round"]
                require(type(r) is int and 0 <= r < self.repeats
                        and r not in self.raw[key] and positive(event["elapsed_ms"]),
                        "Invalid/duplicate raw timing round")
                self.raw[key][r] = event["elapsed_ms"]
            else:
                require(key not in self.rows, "Duplicate terminal case")
                require(event.get("interpretation_only") is not True, "CPU result in TPU evidence")
                self.rows[key] = event
        expected = {(s, c, scope) for s in self.shapes for c in self.candidates for scope in SCOPES}
        require(set(self.rows) == expected, "Terminal case coverage differs from frozen plan")
        require(dict(Counter(r["status"] for r in self.rows.values())) == self.summary["case_status_counts"],
                "Terminal status inventory disagrees with summary")
        for key, row in self.rows.items():
            count = row["timing"]["sample_count"]
            require(type(count) is int and 0 <= count <= self.repeats, "Invalid terminal sample count")
            samples = self.raw[key]
            if row["status"] != "not_run":
                require(count == len(samples), "Terminal/raw sample count mismatch")
            if count:
                require(positive(row["timing"].get("mean_ms"))
                        and close(row["timing"]["mean_ms"], statistics.mean(samples.values())),
                        "Terminal mean disagrees with raw samples")
            if row.get("eligible_for_speedup_claim"):
                require(eligible(row) and set(samples) == set(range(self.repeats)),
                        "Ineligible or incomplete case claims speedup eligibility")
        self.provenance = {
            "directory": str(directory), "artifact_manifest_sha256": sha256(seal_path),
            "phase": self.phase, "identity": self.identity, "seed": self.seed,
            "input_artifact_sha256": {name: self.hashes[name] for name in sorted(required)},
        }

    def validate_context(self, row):
        sid, cid, scope = row["shape_id"], row["candidate_id"], row["scope"]
        require(sid in self.shapes and cid in self.candidates and scope in SCOPES,
                "Unplanned result/sample")
        shape, candidate = self.shapes[sid], self.candidates[cid]
        require(row["group_id"] == sid + "__fixed_ablation" and row["seed"] == self.seed
                and row["distribution"] == "gaussian"
                and row["shape_mnk"] == [shape[d] for d in ("m", "n", "k")]
                and row["shape_mkn"] == [shape[d] for d in ("m", "k", "n")],
                "Result/sample group, input, or axis convention mismatch")
        bound = dict(candidate, tile=None if candidate["implementation"] == "native" else shape["tile_bm_bn_bk"])
        require(all(row.get(k) == v for k, v in bound.items()), "Candidate binding differs from frozen plan")
        return sid, cid, scope

    def comparison(self, key, reference_id):
        row = self.rows[key]
        reference = self.rows[key[0], reference_id, key[2]]
        if not eligible(row) or not eligible(reference):
            return {"available": False, "reason": "Candidate or reference is not timing/numerically eligible"}
        if key[1] == reference_id:
            return {"available": True, "time_reduction_percent": 0., "time_reduction_ci95_percent": [0., 0.],
                    "paired_sample_count": self.repeats, "pointwise_result": "self_reference",
                    "method": "Same recorded arm; identity comparison, not an independent confidence interval"}
        comparisons = [c for c in row["comparisons"] if c["reference_candidate_id"] == reference_id]
        require(len(comparisons) == 1, "Missing/duplicate canonical paired comparison")
        pair = comparisons[0]
        require(pair["group_id"] == row["group_id"] and pair["scope"] == key[2]
                and pair["valid_numerical_comparison"] is True
                and pair["screen_results_are_not_confirmation"] == (self.phase != "confirm")
                and pair["paired_rounds"] == list(range(self.repeats)), "Comparison pairing/provenance mismatch")
        expected = reference["timing"]["mean_ms"] / row["timing"]["mean_ms"]
        require(positive(pair["speedup_ratio_of_means"]) and close(pair["speedup_ratio_of_means"], expected)
                and close(pair["time_reduction_fraction"], 1 - 1 / expected), "Comparison point estimate mismatch")
        reduction, interval = time_reduction_percent(pair["speedup_ratio_of_means"], pair["speedup_ci95"])
        return {"available": True, "time_reduction_percent": reduction,
                "time_reduction_ci95_percent": interval, "paired_sample_count": self.repeats,
                "speedup_ratio_of_means": pair["speedup_ratio_of_means"],
                "speedup_ci95": pair["speedup_ci95"], "method": pair["method"],
                "interval_audit": "Recorded canonical interval transformed monotonically; bootstrap not independently replayed",
                "pointwise_result": "faster" if interval[0] > 0 else "slower" if interval[1] < 0 else "inconclusive"}


def bind_phases(screen, confirmation):
    if not screen or not confirmation:
        return
    require(screen.summary["completed"] is True, "Confirmation parent screen is incomplete")
    require(screen.plan == confirmation.plan and screen.identity == confirmation.identity
            and screen.source["sha256"] == confirmation.source["sha256"]
            and screen.seed != confirmation.seed, "Screen/confirmation frozen cohort mismatch")
    require("screen_provenance.json" in confirmation.hashes, "Confirmation omits sealed parent provenance")
    parent = read_json(confirmation.directory / "screen_provenance.json")
    for field, filename in (("plan_sha256", "plan.json"), ("results_sha256", "results.jsonl"),
                            ("source_manifest_sha256", "source_manifest.json")):
        require(parent[field] == screen.hashes[filename], "Confirmation belongs to a different screen run")


def summarize_run(evidence):
    shape_tables = []
    for sid, shape in evidence.shapes.items():
        candidates = sorted(evidence.candidates, key=lambda c: (TABLE_ORDER.index(c) if c in TABLE_ORDER else len(TABLE_ORDER), c))
        table = []
        for cid in candidates:
            row = evidence.rows[sid, cid, "call"]
            item = {"candidate_id": cid, "label": LABELS.get(cid, cid), "status": row["status"],
                    "eligible": eligible(row), "sample_count": row["timing"]["sample_count"],
                    "mean_ms": row["timing"].get("mean_ms"),
                    "hybrid": row.get("variant") == "peeled_edges",
                    "native_only_fallback": (row.get("kernel_metadata") or {}).get("native_only_fallback"),
                    "tile_bm_bn_bk": row["tile"], "variant": row["variant"],
                    "product_order": row.get("product_order"), "traversal": row.get("traversal"),
                    "correctness": row.get("correctness"), "error_message": row.get("error_message"),
                    "comparisons": {ref: evidence.comparison((sid, cid, "call"), ref) for ref in REFERENCES}}
            table.append(item)
        ranked = sorted((r for r in table if r["eligible"]), key=lambda r: (r["mean_ms"], r["candidate_id"]))
        new = [r for r in ranked if r["candidate_id"].startswith("new_")]
        shape_tables.append({"shape_id": sid, "shape_mnk": [shape[d] for d in ("m", "n", "k")],
                             "tile_bm_bn_bk": shape["tile_bm_bn_bk"], "scope": "call", "rows": table,
                             "descriptive_lowest_mean_candidate": ranked[0]["candidate_id"] if ranked else None,
                             "descriptive_lowest_mean_new_candidate": new[0]["candidate_id"] if new else None})
    # Validate other-scope comparisons too, but never pool them into headline rows.
    for key in evidence.rows:
        if key[2] == "prepared_kernel":
            for ref in REFERENCES:
                evidence.comparison(key, ref)
    failures = [{"shape_id": sid, "candidate_id": cid, "scope": scope, "status": row["status"],
                 "terminal_sample_count": row["timing"]["sample_count"],
                 "raw_sample_count": len(evidence.raw[key]), "error_message": row.get("error_message"),
                 "correctness": row.get("correctness")}
                for key, row in evidence.rows.items() for sid, cid, scope in [key] if not eligible(row)]
    return {"phase": evidence.phase, "completed": evidence.summary["completed"],
            "provenance": evidence.provenance, "rounds": evidence.repeats,
            "case_status_counts_all_scopes": evidence.summary["case_status_counts"],
            "shape_tables": shape_tables, "failures_and_ineligible_cases": failures,
            "error_events": [e for e in evidence.events if e["event"] in ("error", "run_error")],
            "summary_error": evidence.summary.get("error"),
            "correctness_limit": "Recorded Gaussian-input finite/error gate; sampled outputs may not cover the entire matrix. Not accuracy equivalence.",
            "comparison_interval_audit": "Raw means, pairing and canonical point estimates checked; recorded 2000-resample PCG64 bootstrap intervals transformed, not independently replayed."}


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    lines = ["# Strassen optimization TPU ablation", "", INTERPRETATION, "",
             "All tables use complete-device-call arithmetic mean latency, including preparation and finish; compilation and host transfers are excluded.",
             "Percentages are time reduction relative to the named reference: positive is faster; negative is slower. Brackets contain pointwise 95% intervals. Self-reference intervals are identities.", ""]
    for run in report["runs"]:
        identity = run["provenance"]["identity"]
        lines += [f"## {run['phase'].title()} — allocation {escaped(identity['allocation_id'])}", "",
                  f"Completed: **{run['completed']}**. Input seed: {run['provenance']['seed']}. Paired rounds: {run['rounds']}. Device: {escaped(identity.get('device_kind'))}.", "",
                  f"Source evidence: `{escaped(run['provenance']['directory'])}`.", ""]
        for shape in run["shape_tables"]:
            lines += [f"### M,N,K = {','.join(map(str, shape['shape_mnk']))}; custom tile = {','.join(map(str, shape['tile_bm_bn_bk']))}", "",
                      "| Arm | Status | n | Mean ms | Reduction vs frozen Strassen | Reduction vs native default | Reduction vs native 64 MiB |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for row in shape["rows"]:
                cells = []
                for ref in REFERENCES:
                    c = row["comparisons"][ref]
                    cells.append(f"{c['time_reduction_percent']:+.2f}% [{c['time_reduction_ci95_percent'][0]:+.2f}, {c['time_reduction_ci95_percent'][1]:+.2f}]" if c["available"] else "unavailable")
                status = row["status"] + ("; ineligible" if row["status"] == "ok" and not row["eligible"] else "")
                label = row["label"] + ("; native-only fallback" if row["native_only_fallback"] else "")
                mean = f"{row['mean_ms']:.6f}" if row["mean_ms"] is not None else "—"
                lines.append("| " + " | ".join([escaped(label), escaped(status), str(row["sample_count"]), mean, *cells]) + " |")
            lines += ["", f"Lowest observed eligible mean (descriptive only): `{shape['descriptive_lowest_mean_candidate']}`. Among new arms: `{shape['descriptive_lowest_mean_new_candidate']}`. These are not confirmed winner declarations.", ""]
        lines += ["### Failures and ineligible cases, both scopes", ""]
        if not run["failures_and_ineligible_cases"]:
            lines += ["None recorded.", ""]
        else:
            lines += ["| Shape | Arm | Scope | Status | Terminal/raw n | Detail |", "|---|---|---|---|---:|---|"]
            for r in run["failures_and_ineligible_cases"]:
                detail = r["error_message"] or ("Numerical or timing eligibility failed; see JSON correctness metrics.")
                lines.append("| " + " | ".join(map(escaped, [r["shape_id"], r["candidate_id"], r["scope"], r["status"], f"{r['terminal_sample_count']}/{r['raw_sample_count']}", detail])) + " |")
            lines.append("")
        if run["error_events"]:
            lines += ["Recorded error events:", ""]
            for e in run["error_events"]:
                lines.append(f"- {escaped(e.get('shape_id', 'run'))} / {escaped(e.get('candidate_id', 'run'))} / {escaped(e.get('scope', 'all'))}: {escaped(e.get('status'))}: {escaped(e.get('message'))}")
            lines.append("")
        lines += [run["comparison_interval_audit"], "", run["correctness_limit"], ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--confirmation", "--confirm", dest="confirmation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.screen and not args.confirmation:
        parser.error("Supply --screen and/or --confirmation")
    screen = Evidence(args.screen, "screen") if args.screen else None
    confirmation = Evidence(args.confirmation, "confirm") if args.confirmation else None
    bind_phases(screen, confirmation)
    report = {"version": VERSION, "interpretation": INTERPRETATION,
              "summarizer_sha256": sha256(__file__),
              "runs": [summarize_run(e) for e in (screen, confirmation) if e is not None]}
    rendered = markdown(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "findings.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    manifest = {"sha256": {name: sha256(args.output_dir / name) for name in ("findings.json", "summary.md")},
                "summarizer_sha256": report["summarizer_sha256"], "inputs": [r["provenance"] for r in report["runs"]]}
    with (args.output_dir / "artifact_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote separate phase summaries to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
