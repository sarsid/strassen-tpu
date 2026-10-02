"""Local, read-only progress view. Counts come from execution journals.

No job control, remote polling, fabricated percentage, or automatic scheduling.
The dashboard refreshes filesystem evidence; human work is explicitly reported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import unquote, urlparse

ROOT = Path(os.environ.get("STRASSEN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
UI = Path(__file__).with_name("progress_v001.html")
JOURNAL = "status/work_progress_v001.jsonl"
COHORT = "runs/20260920-region-grid-cohort-v001"
GEMMA = "runs/20260920T214756Z-Gemma-real-qualification-v001-8d3ef5"


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def rows(path):
    try:
        with path.open() as handle:
            for line in handle:
                if not line.endswith("\n"):
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                except ValueError:
                    continue
    except OSError:
        return


def age(utc, now):
    try:
        return max(0, (now - datetime.fromisoformat(utc.replace("Z", "+00:00"))).total_seconds())
    except (ValueError, TypeError, AttributeError):
        return None


def evidence(label, path):
    return {"label": label, "path": path}


class JournalReader:
    """Incrementally read complete lines; do not consume a partial append."""

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def read(self, path):
        with self.lock:
            try:
                stat = path.stat()
            except OSError:
                return {}
            key = str(path)
            item = self.cache.get(key)
            if item is None or item["inode"] != stat.st_ino or stat.st_size < item["offset"]:
                item = {"inode": stat.st_ino, "offset": 0, "completed": 0, "total": None}
                self.cache[key] = item
            with path.open("rb") as handle:
                handle.seek(item["offset"])
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    item["offset"] = handle.tell()
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    if row.get("utc"):
                        item["updated_utc"] = row["utc"]
                    event = row.get("event")
                    if event in ("group_start", "group_complete"):
                        if type(row.get("total")) is int and row["total"] > 0:
                            item["total"] = row["total"]
                        item["current_group"] = row.get("group_id")
                    if event == "group_complete" and type(row.get("index")) is int:
                        item["completed"] = max(item["completed"], row["index"])
                    if event in ("run_error", "error"):
                        item["error"] = row.get("type") or row.get("message") or event
            return dict(item)


READER = JournalReader()


def grid_state(root, now):
    folder = root / COHORT
    phase = next((name for name in ("GRID-confirm", "GRID-screen", "GRID-smoke")
                  if (folder / (name + "-started.json")).exists()), None)
    if phase is None:
        return {"state": "unknown", "phase": "No execution found", "completed": None,
                "total": None, "detail": "No grid execution record is available.", "evidence": []}
    started_path = folder / (phase + "-started.json")
    started = read_json(started_path)
    pointer = started.get("run")
    if not isinstance(pointer, str) or not pointer.strip():
        return {"state": "unknown", "phase": phase, "completed": None, "total": None,
                "detail": "Execution has no valid run pointer.", "evidence": []}
    run = Path(pointer)
    if not run.is_absolute():
        run = root / run
    run = run.resolve()
    if not run.is_relative_to(root.resolve()) or not run.is_dir():
        return {"state": "unknown", "phase": phase, "completed": None, "total": None,
                "detail": "Execution path is missing or outside this project.", "evidence": []}
    journal = run / "results.jsonl"
    data = READER.read(journal)
    finish = read_json(folder / (phase + "-finished.json")) or read_json(run / "completion.json")
    utc = data.get("updated_utc") or started.get("started_utc")
    seconds = age(utc, now)
    if finish:
        state = "complete" if finish.get("status") == "completed" else "attention"
    elif data.get("error"):
        state = "attention"
    elif not data.get("updated_utc"):
        state = "unknown"
    else:
        state = "running" if seconds is not None and seconds <= 120 else "stale"
    detail = "Separate MM grid study; these groups are not the three-model mini experiment."
    if state == "stale":
        detail += " No recent journal update; this does not prove the process stopped."
    return {"state": state, "phase": phase, "completed": data.get("completed", 0),
            "total": data.get("total"), "updated_utc": utc, "age_seconds": seconds,
            "current_group": data.get("current_group"), "detail": detail,
            "evidence": [evidence("Live execution journal", str(journal.relative_to(root))),
                         evidence("Controller log", COHORT + "/" + phase + "-controller.log")]}


def gemma_state(root):
    path = GEMMA + "/artifacts/summary.json"
    result = read_json(root / path)
    checks = result.get("checks", {})
    rows_out = []
    descriptions = (
        ("final_hidden", "relative_l2", "Final hidden-state relative L2", 0.02),
        ("full_logits", "relative_l2", "Full logit relative L2", 0.03),
        ("next_token_logits", "top1_agreement", "Next-token agreement", 0.90),
    )
    for name, key, label, limit in descriptions:
        row = checks.get(name, {})
        rows_out.append({"name": label, "value": row.get(key), "limit": limit,
                         "passed": row.get("passed"), "direction": ">=" if key == "top1_agreement" else "<="})
    return {"state": "complete" if result.get("passed") else "attention" if result else "unknown",
            "passed": sum(row.get("passed") is True for row in checks.values()),
            "total": len(checks), "checks": rows_out,
            "evidence": [evidence("Results and decisions", "reports/gemma_support_v001/README.md"),
                         evidence("Recorded qualification", path),
                         evidence("Numerical diagnostics", "reports/gemma_support_v001/intermediate_diagnostics.md")]}


def state(root=ROOT, now=None):
    now = now or datetime.now(timezone.utc)
    events = list(rows(root / JOURNAL))
    latest = {}
    for row in events:
        latest[(row.get("kind"), row.get("id", ""))] = row
    models = []
    for name in ("Qwen", "Mistral", "Gemma"):
        item = dict(latest.get(("model", name.lower()), {}))
        item.setdefault("name", name)
        item.setdefault("state", "unknown")
        item.setdefault("detail", "No mini-experiment execution status recorded.")
        item.setdefault("completed", None)
        item.setdefault("total", None)
        item.setdefault("evidence", [])
        models.append(item)
    activity = latest.get(("activity", "current"), {"title": "No current activity reported",
                         "state": "unknown", "detail": "Work updates are recorded when actions change."})
    agents = [row for (kind, _), row in latest.items() if kind == "agent"]
    timeline = [row for row in events if row.get("kind") in ("activity", "milestone", "decision", "model")]
    try:
        result = subprocess.run(["git", "log", "-6", "--format=%h\t%s"], cwd=root,
                                capture_output=True, text=True, timeout=3, check=True)
        commits = [dict(zip(("hash", "subject"), line.split("\t", 1)))
                   for line in result.stdout.splitlines() if "\t" in line]
    except (OSError, subprocess.SubprocessError):
        commits = []
    return {"generated_utc": now.isoformat(), "activity": activity, "models": models,
            "grid": grid_state(root, now), "gemma": gemma_state(root), "agents": agents,
            "events": sorted(timeline, key=lambda row: row.get("utc", ""), reverse=True)[:100],
            "commits": commits}


def public_file(root, relative):
    """Only expose study evidence, never private caches or hidden paths."""
    if ".." in Path(relative).parts or Path(relative).is_absolute():
        return None
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        return None
    parts = path.relative_to(root).parts
    if not parts or parts[0] not in ("runs", "reports", "plans", "status"):
        return None
    if any(part.startswith(".") for part in parts) or path.suffix not in (".md", ".txt", ".json", ".jsonl", ".log"):
        return None
    return path if path.is_file() else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/progress":
            data, mime = json.dumps(state(), allow_nan=False).encode(), "application/json"
        elif url.path == "/":
            data, mime = UI.read_bytes(), "text/html; charset=utf-8"
        elif url.path.startswith("/files/"):
            path = public_file(ROOT, unquote(url.path[7:]))
            if path is None:
                self.send_error(404)
                return
            # Large live journals are tailed to keep a click responsive.
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - 2 * 1024 ** 2))
                data = handle.read()
            if size > len(data):
                data = b"[Showing the last 2 MiB of this live evidence file]\n" + data
            mime = "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"Live project progress: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
