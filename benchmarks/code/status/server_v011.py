"""Read-only campaign progress: explicit measured attempts and fresh evidence.

Serve on a new loopback port, preserving the historical progress server. The
controller owns the atomic status snapshot and append-only execution journals.
This module does not run jobs, probe processes, update status, or contact TPUs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import server_v008 as original

BASE_STATE = original.state
FEED = "status/campaign_progress_v001.json"
UI = Path(__file__).with_name("progress_v003.html")
FRESH_SECONDS = 120
STATES = {"unknown", "not_started", "pending", "preparing", "running", "waiting",
          "succeeded", "completed", "failed", "attention", "blocked", "cancelled"}
ACTIVE = {"preparing", "running", "waiting"}
METHODS = ("native_default", "native_tuned", "cubic", "strassen1", "strassen2")
COUNTING_UNIT = "Distinct (group_id, seed, arm_id) terminal call-scope outcomes; prepared scope is not counted twice."


def invalid_constant(value):
    raise ValueError("Non-finite JSON value in campaign snapshot: " + value)


def elapsed(value, now):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        seconds = (now - parsed).total_seconds()
        return max(0, seconds) if seconds >= -30 else None
    except (ValueError, TypeError, AttributeError):
        return None


def feed_path(root, value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Campaign status path is missing")
    root = Path(root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("Campaign status must be a JSON file inside the project")
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        raise ValueError("Private status paths are not exposed")
    return path


def links(root, items):
    result = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        name = item["path"]
        if Path(name).is_absolute():
            try:
                name = str(Path(name).resolve().relative_to(root))
            except ValueError:
                continue
        if original.public_file(root, name) is not None:
            result.append({"label": str(item.get("label") or Path(name).name), "path": name})
    return result


def normalized_state(value):
    return value if isinstance(value, str) and value in STATES else "unknown"


def freshness(item, now):
    item["reported_state"] = item["state"]
    item["heartbeat_age_seconds"] = elapsed(item.get("heartbeat_utc"), now)
    item["worker_data_age_seconds"] = elapsed(item.get("worker_last_data_utc"), now)
    item["heartbeat_stale"] = (item["state"] in ACTIVE and
                               (item["heartbeat_age_seconds"] is None or item["heartbeat_age_seconds"] > FRESH_SECONDS))
    item["worker_data_quiet"] = (item["state"] in ACTIVE and
                                 (item["worker_data_age_seconds"] is None or item["worker_data_age_seconds"] > FRESH_SECONDS))
    if item["heartbeat_stale"]:
        item["state"] = "stale"
    return item


def stage_record(row, root, now, parent, warnings):
    kind = row.get("kind") if row.get("kind") in ("preparation", "measurement", "analysis") else "unknown"
    item = {"id": str(row.get("id", "unknown")), "label": str(row.get("label") or row.get("id") or "Unnamed stage"),
            "kind": kind, "state": normalized_state(row.get("state")), "detail": str(row.get("detail") or ""),
            "current_candidate": row.get("current_candidate"),
            "heartbeat_utc": row.get("heartbeat_utc") or parent.get("heartbeat_utc"),
            "worker_last_data_utc": row.get("worker_last_data_utc"),
            "evidence": links(root, row.get("evidence"))}
    for key in ("completed", "expected", "succeeded", "failed", "measured"):
        value = row.get(key)
        item[key] = value if kind == "measurement" and type(value) is int and value >= 0 else None
        if kind == "measurement" and value is not None and item[key] is None:
            warnings.append(item["id"] + ": invalid " + key)
    if kind == "measurement":
        done, total = item["completed"], item["expected"]
        invalid = (done is not None and total is not None and done > total)
        invalid |= (done is not None and item["measured"] is not None and item["measured"] > done)
        if all(item[k] is not None for k in ("completed", "succeeded", "failed")):
            invalid |= item["succeeded"] + item["failed"] != done
        if invalid:
            warnings.append(item["id"] + ": inconsistent outcome counters; progress hidden")
            for key in ("completed", "expected", "succeeded", "failed", "measured"):
                item[key] = None
            item["state"] = "attention"
        if item["state"] in ("succeeded", "completed") and (done is None or total is None or done != total):
            warnings.append(item["id"] + ": terminal success without complete coverage")
            item["state"] = "attention"
    return freshness(item, now)


def selected_record(row, root):
    geometry = row.get("shape_mkn")
    if (row.get("method") not in METHODS or row.get("scope") not in ("call", "prepared_kernel")
            or not isinstance(row.get("shape_id"), str) or not isinstance(geometry, list)
            or len(geometry) != 3 or any(type(x) is not int or x <= 0 for x in geometry)):
        return None
    mean = row.get("mean_ms")
    valid_mean = type(mean) in (float, int) and math.isfinite(mean) and mean > 0
    return {"shape_id": row["shape_id"], "shape_mkn": geometry, "stage": str(row.get("stage") or ""),
            "scope": row["scope"], "selection_status": str(row.get("selection_status") or "unknown"),
            "method": row["method"], "candidate_id": str(row.get("candidate_id") or "unavailable"),
            "tile": row.get("tile"), "mean_ms": mean if valid_mean else None,
            "eligible": row.get("eligible") if type(row.get("eligible")) is bool else None,
            "status": str(row.get("status") or ("eligible" if row.get("eligible") is True else "unavailable")),
            "evidence": links(root, row.get("evidence"))}


def campaign_state(root, now, name=FEED):
    empty = {"schema_version": 1, "campaign_id": None, "title": "New MM campaign", "state": "unknown",
             "stage": None, "detail": "No campaign measurement snapshot is available.", "next_step": None,
             "stages": [], "selected_results": [], "warnings": [], "evidence": [],
             "counting_unit": COUNTING_UNIT, "completed": None, "expected": None, "measured": None,
             "heartbeat_utc": None, "worker_last_data_utc": None}
    try:
        path = feed_path(root, name)
        if path.stat().st_size > 8 * 1024 ** 2:
            raise ValueError("Status snapshot exceeds 8 MiB")
        payload = json.loads(path.read_text(), parse_constant=invalid_constant)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported campaign status schema")
    except (OSError, ValueError) as error:
        if not isinstance(error, FileNotFoundError):
            empty["warnings"].append(str(error))
        return freshness(empty, now)
    result = {**empty, **{key: payload.get(key) for key in ("campaign_id", "title", "cohort_path", "stage", "detail",
                                                          "next_step", "updated_utc", "heartbeat_utc", "worker_last_data_utc")},
              "state": normalized_state(payload.get("state")), "warnings": []}
    result["title"] = str(result.get("title") or "MM campaign")
    result["evidence"] = links(root, payload.get("evidence")) + [original.evidence("Controller status snapshot", str(path.relative_to(root)))]
    stage_rows = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    result["stages"] = [stage_record(row, root, now, payload, result["warnings"]) for row in stage_rows if isinstance(row, dict)]
    ids = [row["id"] for row in result["stages"]]
    if len(ids) != len(set(ids)):
        result["warnings"].append("Duplicate stage IDs; overall progress hidden")
    else:
        measured_stages = [row for row in result["stages"] if row["kind"] == "measurement"]
        for key in ("completed", "expected", "measured"):
            result[key] = sum(row[key] for row in measured_stages) if measured_stages and all(row[key] is not None for row in measured_stages) else None
    selected = payload.get("selected_results") if isinstance(payload.get("selected_results"), list) else []
    result["selected_results"] = []
    for row in selected:
        clean = selected_record(row, root) if isinstance(row, dict) else None
        if clean is not None:
            result["selected_results"].append(clean)
        else:
            result["warnings"].append("An invalid selected-result row was omitted")
    if result["state"] in ("succeeded", "completed") and (
            result["completed"] is None or result["expected"] is None or result["completed"] != result["expected"]):
        result["warnings"].append("Campaign success is reported without complete measurement coverage")
        result["state"] = "attention"
    return freshness(result, now)


def agent_records(rows, now):
    result = []
    for source in rows:
        item = dict(source)
        seconds = elapsed(item.get("utc"), now)
        item["age_seconds"] = seconds
        item["reported_state"] = item.get("state", "unknown")
        active = str(item.get("state", "")).lower() in {"working", "running", "active", "reviewing", "preparing", "validating"}
        item["stale"] = active and (seconds is None or seconds > FRESH_SECONDS)
        if item["stale"]:
            item["state"] = "stale"
        result.append(item)
    return result


def state(root=original.ROOT, now=None, campaign_file=None):
    root = Path(root).resolve()
    now = now or datetime.now(timezone.utc)
    payload = BASE_STATE(root, now)
    payload["campaign"] = campaign_state(root, now, campaign_file or FEED)
    payload["agents"] = agent_records(payload.get("agents", []), now)
    payload["dashboard_version"] = "v011"
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--campaign-state", default=FEED)
    args = parser.parse_args()
    FEED = feed_path(original.ROOT, args.campaign_state)
    original.state = state
    original.UI = UI
    print(f"MM campaign progress v011: http://127.0.0.1:{args.port}", flush=True)
    original.ThreadingHTTPServer(("127.0.0.1", args.port), original.Handler).serve_forever()
