"""Read-only status panel with actual, source-linked benchmark result rows.

Version 007 adds finalized, sealed N9 model/policy rows and preserves the earlier agent/run/commit views. Final result files
take precedence over live mirrors; one authoritative file is read per run.
No benchmark, execution metadata, or earlier status implementation is changed.
"""
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import math
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]


def records(path):
    """Ignore incomplete final lines while a live mirror is being extended."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def results_path(directory):
    # The remote archive is retained under remote/<run_id>/ by the controller.
    # Its final artifacts have the same priority as a directly extracted one.
    choices = (directory / "artifacts/results.jsonl",
               directory / "remote" / directory.name / "artifacts/results.jsonl",
               directory / "results.jsonl")
    return next((path for path in choices if path.is_file()), None)


def counts_for_display(counts):
    classified = ("ok", "numerical_failure", "compile_error", "oom")
    return {"total": sum(counts.values()), "ok": counts["ok"],
            "numerical_failure": counts["numerical_failure"],
            "compile_error": counts["compile_error"], "oom": counts["oom"],
            "other": sum(value for key, value in counts.items() if key not in classified),
            "by_status": dict(counts)}


def compact_result(record, metadata, path, completion):
    kernel = record.get("kernel_metadata") or {}
    correctness = record.get("correctness") or record.get("quality") or {}
    timing = record.get("timing") or {}
    phase = record.get("phase", "unknown")
    comparison_list = record.get("comparisons") or []
    selected = [item for item in comparison_list
                if item.get("reference_arm") in ("native_xla", "native_joint_graph")
                or item.get("reference_arm", "") == "cubic_selected"
                or item.get("reference_arm", "").startswith("cubic_full")
                or item.get("reference_arm", "").startswith("cubic_quadrant")
                or item.get("reference_arm", "").startswith("c_")]
    if not selected:
        selected = [item for item in comparison_list if item.get("reference_arm") == "cubic_basic"]
    selected.sort(key=lambda item: (item.get("reference_arm") not in ("native_xla", "native_joint_graph"),
                                    item.get("reference_arm", "")))
    group_id = record.get("group_id", "")
    shape_id = record.get("shape_id") or record.get("model_id") or group_id.split("__", 1)[0] or "unknown"
    tile = kernel.get("tile_bm_bn_bk")
    if not kernel and "__" in group_id:
        tile = group_id.split("__", 1)[1]
    live = completion is None
    return {
        "run_id": metadata["run_id"], "phase": phase, "shape_id": shape_id,
        "shape_mkn": kernel.get("shape_mkn"), "arm_id": record.get("arm_id", "unknown"),
        "algorithm": record.get("algorithm") or kernel.get("algorithm"),
        "tile_bm_bn_bk": tile, "status": record.get("status", "unknown"),
        "scope": record.get("scope", "unknown"), "mean_ms": timing.get("mean_ms"),
        "sample_count": timing.get("sample_count", 0), "relative_l2": correctness.get("relative_l2"),
        "reference_scope": correctness.get("reference_scope"),
        "comparisons": [{key: item.get(key) for key in (
            "reference_arm", "speedup_ratio_of_means", "speedup_ci95", "valid_numerical_comparison"
        )} for item in selected[:2]],
        "eligible_for_speedup_claim": record.get("eligible_for_speedup_claim", False),
        "distribution": record.get("distribution"), "seed": record.get("seed"),
        "utc": record.get("utc", metadata.get("created_utc", "")),
        "sequence": record.get("sequence", 0), "live": live,
        "evidence_label": "live preliminary" if live else "archived execution",
        "screen": phase in ("N2", "N5-screen", "N7-screen"), "accuracy_only": phase == "N4",
        "source_path": str(path.relative_to(ROOT)),
    }



def digest_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_member(directory, relative, expected):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Missing or escaping sealed artifact: " + relative)
    if isinstance(expected, dict):
        valid = expected.get("bytes") == path.stat().st_size and expected.get("sha256") == digest_file(path)
    else:
        valid = isinstance(expected, str) and expected == digest_file(path)
    if not valid:
        raise ValueError("Artifact hash mismatch: " + relative)
    return path


def finite_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def n9_model_results(directory, completion):
    """Read only finalized canonical summaries; validate every displayed source."""
    artifacts = directory / "artifacts"
    summary_path = artifacts / "summary.json"
    summary = read_json(summary_path)
    if not summary or summary.get("phase") != "N9":
        return [], None
    try:
        if not completion or completion.get("remote_may_still_be_running") is not False:
            raise ValueError("Final remote completion has not been verified")
        outer = read_json(directory / "artifact-manifest.json")
        if not outer:
            raise ValueError("Final execution seal is not available yet")
        verify_member(directory, "completion.json", outer.get("completion.json"))
        verify_member(directory, "artifacts/artifact_manifest.json", outer.get("artifacts/artifact_manifest.json"))
        seal = read_json(artifacts / "artifact_manifest.json")
        hashes = (seal or {}).get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError("Canonical N9 seal has no SHA256 map")
        def canonical(relative):
            path = verify_member(artifacts, relative, hashes.get(relative))
            value = read_json(path)
            if value is None:
                raise ValueError("Invalid canonical JSON: " + relative)
            return path, value
        summary_path, summary = canonical("summary.json")
        _, campaign = canonical("effective_campaign.json")
        _, environment = canonical("environment.json")
        models = summary.get("results")
        if not isinstance(models, list):
            raise ValueError("Final N9 summary has no model result list")
        rows = []
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                raise ValueError("Invalid model summary entry")
            relative = "model-%02d/model_summary.json" % index
            source = summary_path
            if (artifacts / relative).is_file():
                source, child = canonical(relative)
                if child != model:
                    raise ValueError("Model summary disagrees with final N9 summary")
            quality = model.get("quality") or {}
            failures = model.get("failures") or {}
            resident = model.get("resident_layer_mean_ms") or {}
            streamed = model.get("streamed_forward_mean_ms") or {}
            samples = model.get("streamed_forward_samples_ms") or {}
            coverage = model.get("resident_sample_coverage") or {}
            eligible = model.get("eligible_for_speedup_claim")
            eligibility = eligible if isinstance(eligible, dict) else {}
            policies = set(quality) | set(failures) | set(resident) | set(streamed) | set(samples) | set(eligibility)
            ordered = sorted(policies, key=lambda name: (name != "native", name)) if policies else [None]
            for policy in ordered:
                q = quality.get(policy) or {}
                failure = failures.get(policy) or {}
                resident_coverage = coverage.get(policy) or {}
                qualification = model.get("qualification") or {}
                policy_status = failure.get("status") or model.get("status", "unknown")
                if q.get("passed") is False and not failure:
                    policy_status = "failed_quality"
                rows.append({
                    "run_id": directory.name, "model_id": model.get("model_id", "unknown"),
                    "revision": model.get("revision"), "arm_id": policy,
                    "status": policy_status, "model_status": model.get("status"),
                    "reason": model.get("reason") or model.get("error") or failure.get("message"),
                    "native_qualification_pass": qualification.get("passed"),
                    "native_qualification_positions": qualification.get("positions"),
                    "quality_pass": q.get("passed"), "quality_measured": model.get("quality_measured", False),
                    "nll_delta": finite_number(q.get("nll_delta")),
                    "mean_kl": finite_number(q.get("mean_kl")),
                    "top1_agreement": finite_number(q.get("top1_agreement")),
                    "quality_positions": q.get("positions"),
                    "resident_layer_mean_ms": finite_number(resident.get(policy)),
                    "resident_sample_count": resident_coverage.get("observed"),
                    "resident_expected_sample_count": resident_coverage.get("expected"),
                    "resident_complete": resident_coverage.get("complete"),
                    "resident_repeats_per_layer": (campaign.get("resident_timing") or {}).get("repeats"),
                    "streamed_forward_mean_ms": finite_number(streamed.get(policy)),
                    "streamed_sample_count": len(samples.get(policy) or []),
                    "streamed_expected_repeats": campaign.get("streamed_repeats"),
                    "eligible_for_speedup_claim": eligibility.get(policy) is True,
                    "allocation_id": (environment.get("identity") or {}).get("allocation_id"),
                    "run_status": completion.get("status"), "finished_utc": summary.get("finished_utc"),
                    "evidence_label": "finalized · source hashes verified",
                    "source_path": str(source.relative_to(ROOT)),
                    "run_summary_path": str(summary_path.relative_to(ROOT)),
                    "quality_scope": campaign.get("quality_scope"),
                    "timing_scope": campaign.get("timing_scope"),
                    "uncertainty_scope": model.get("uncertainty_scope"),
                    "model_scope": model.get("scope") or campaign.get("model_scope"),
                })
        return rows, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [], {"run_id": directory.name, "message": str(error),
                    "source_path": str(summary_path.relative_to(ROOT)),
                    "evidence_label": "Not displayed until final summary verification succeeds"}


def state():
    events = list(records(ROOT / "status/events.jsonl"))
    agents = {}
    for entry in events:
        if entry.get("kind") == "agent" and entry.get("name"):
            agents[entry["name"]] = entry
    runs, result_rows = [], []
    n9_rows, n9_issues = [], []
    total_counts = Counter()
    for path in sorted((ROOT / "runs").glob("*/execution.json")):
        metadata = read_json(path)
        if metadata is None:
            continue
        directory = path.parent
        metadata.setdefault("run_id", directory.name)
        completion = read_json(directory / "completion.json")
        metadata["completion"] = completion
        source = results_path(directory)
        count = 0
        last_record = None
        statuses = Counter()
        recent = deque(maxlen=20)
        seen_cases = set()
        if source:
            for row in records(source):
                count += 1
                last_record = {key: row.get(key) for key in (
                    "event", "kind", "shape_id", "arm_id", "group_id", "phase", "status", "utc"
                )}
                if row.get("event") != "case_result":
                    continue
                # Defensive deduplication if a live mirror repeats a boundary
                # line. Independent runs still remain independent evidence.
                identity = (row.get("case_id") or row.get("group_id"),
                            row.get("distribution"), row.get("seed"),
                            row.get("arm_id"), row.get("scope"))
                if identity in seen_cases:
                    continue
                seen_cases.add(identity)
                statuses[row.get("status", "unknown")] += 1
                if metadata.get("scope") == "tpu_benchmark" or row.get("phase") in (
                    "smoke", "N1", "N2", "N3", "N4"
                ):
                    recent.append(compact_result(row, metadata, source, completion))
        metadata["record_count"] = count
        metadata["last_record"] = last_record
        metadata["completed_cases"] = sum(statuses.values())
        metadata["failed_cases"] = sum(value for key, value in statuses.items() if key != "ok")
        metadata["case_counts"] = counts_for_display(statuses)
        metadata["results_source"] = str(source.relative_to(ROOT)) if source else None
        total_counts.update(statuses)
        result_rows.extend(recent)
        runs.append(metadata)
        model_rows, issue = n9_model_results(directory, completion)
        n9_rows.extend(model_rows)
        if issue:
            n9_issues.append(issue)
    result_rows.sort(key=lambda row: (row["utc"], row["run_id"], row["sequence"]), reverse=True)
    try:
        commits = subprocess.run(["git", "log", "-8", "--format=%h %s"], cwd=ROOT,
                                 text=True, capture_output=True, timeout=5).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        commits = []
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "agents": list(agents.values()), "events": events[-18:],
            "runs": runs, "commits": commits, "recent_results": result_rows[:20],
            "result_totals": counts_for_display(total_counts),
            "n9_model_results": n9_rows, "n9_verification_issues": n9_issues,
            "n9_evidence_scope": "Final canonical summaries only; hashes checked for displayed sources, not a full scientific audit."}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            data = json.dumps(state(), allow_nan=False).encode()
            mime = "application/json"
        elif url.path == "/":
            data = (ROOT / "status/index_v007.html").read_bytes()
            mime = "text/html; charset=utf-8"
        elif url.path.startswith("/files/"):
            target = (ROOT / unquote(url.path[7:])).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file() or any(
                part.startswith(".") for part in target.relative_to(ROOT).parts
            ):
                self.send_error(404)
                return
            data = target.read_bytes()
            mime = "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print("Status v007: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

