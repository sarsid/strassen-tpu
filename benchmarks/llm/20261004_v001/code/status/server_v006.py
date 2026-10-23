"""Read-only status panel with actual, source-linked benchmark result rows.

Version 006 preserves the earlier agent/run/commit views. Final result files
take precedence over live mirrors; one authoritative file is read per run.
No benchmark, execution metadata, or earlier status implementation is changed.
"""
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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


def state():
    events = list(records(ROOT / "status/events.jsonl"))
    agents = {}
    for entry in events:
        if entry.get("kind") == "agent" and entry.get("name"):
            agents[entry["name"]] = entry
    runs, result_rows = [], []
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
    result_rows.sort(key=lambda row: (row["utc"], row["run_id"], row["sequence"]), reverse=True)
    try:
        commits = subprocess.run(["git", "log", "-8", "--format=%h %s"], cwd=ROOT,
                                 text=True, capture_output=True, timeout=5).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        commits = []
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "agents": list(agents.values()), "events": events[-18:],
            "runs": runs, "commits": commits, "recent_results": result_rows[:20],
            "result_totals": counts_for_display(total_counts)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            data = json.dumps(state(), allow_nan=False).encode()
            mime = "application/json"
        elif url.path == "/":
            data = (ROOT / "status/index_v006.html").read_bytes()
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
    print("Status v006: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
